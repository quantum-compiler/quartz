# this file is under mypy's checking
from __future__ import annotations
import os
from typing import List, Dict, cast, OrderedDict
import warnings
from functools import partial
import time
import copy
import math
from numpy import str0
from tqdm import tqdm  # type: ignore
from natsort import natsorted

import qtz

import torch
from torch.distributions import Categorical
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.nn.parallel import DistributedDataParallel as DDP
# import quartz # type: ignore

import hydra
import wandb

from config.config import *
from ds import *
from utils import *
from model.actor_critic import ActorCritic
from actor import PPOAgent
from tester import Tester

from IPython import embed  # type: ignore
from icecream import ic  # type: ignore


class PPOMod:
    def __init__(self, cfg: BaseConfig, output_dir: str) -> None:
        self.cfg: BaseConfig = cfg
        self.output_dir = output_dir
        self.wandb_mode = 'online'
        if cfg.wandb.offline:
            self.wandb_mode = 'offline'
        elif cfg.wandb.en is False:
            self.wandb_mode = 'disabled'
        wandb.require("service")
        wandb.setup()
        self.print_cfg()
        """init quartz"""
        self.init_quartz_context_func = partial(
            qtz.init_quartz_context,
            cfg.gate_set,
            cfg.ecc_file,
            cfg.no_increase,
            cfg.include_nop,
        )
        self.input_graphs: List[Dict[str, str]] = []
        for input_graph in cfg.input_graphs:
            with open(input_graph.path) as f:
                self.input_graphs.append({
                    'name': input_graph.name,
                    'qasm': f.read(),
                })
        self.num_gate_types: int = cfg.num_gate_types

    def print_cfg(self) -> None:
        print('================ Configs ================')
        print(OmegaConf.to_yaml(self.cfg))
        # for k, v in self.cfg.items():
        #     print(f'{k} : {v}')
        print(f'output_dir : {self.output_dir}')
        print('=========================================')

    def init_process(self, rank: int, ddp_processes: int,
                     obs_processes: int) -> None:
        seed_all(self.cfg.seed + rank)
        """init Quartz for each process"""
        self.init_quartz_context_func()
        """set num of OMP threads to avoid blasting the machine"""
        if self.cfg.omp_num_threads != 0:
            os.environ['OMP_NUM_THREADS'] = str(self.cfg.omp_num_threads)
        # otherwise we don't limit it
        """RPC and DDP initialization"""
        # Ref: https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html
        self.rank = rank
        self.ddp_processes = ddp_processes
        tot_processes = ddp_processes + obs_processes
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=f'tcp://localhost:{self.cfg.ddp_port + 1}',
            rpc_timeout=0,
        )

        if rank < ddp_processes:
            """init agent processes"""
            agent_name = get_agent_name(rank)
            rpc.init_rpc(
                name=agent_name,
                rank=rank,
                world_size=tot_processes,
                rpc_backend_options=rpc_backend_options,
            )
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://localhost:{self.cfg.ddp_port}',
                rank=rank,
                world_size=ddp_processes,
            )
            self.train()
        else:
            """init observer processes"""
            obs_rank = rank - ddp_processes
            agent_rref_id = obs_rank // self.cfg.obs_per_agent
            obs_in_agent_rank = obs_rank % self.cfg.obs_per_agent
            obs_name = get_obs_name(agent_rref_id, obs_in_agent_rank)
            rpc.init_rpc(
                name=obs_name,
                rank=rank,
                world_size=tot_processes,
                rpc_backend_options=rpc_backend_options,
            )
        # block until all rpcs finish
        rpc.shutdown()

    def train(self) -> None:
        """init agent and network"""
        if self.cfg.gpus is None or len(self.cfg.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.cfg.gpus[self.rank]}')
        torch.cuda.set_device(self.device)
        self.ac_net: ActorCritic = ActorCritic(
            gnn_type=self.cfg.gnn_type,
            num_gate_types=self.num_gate_types,
            gate_type_embed_dim=self.cfg.gate_type_embed_dim,
            gnn_num_layers=self.cfg.gnn_num_layers,
            gnn_hidden_dim=self.cfg.gnn_hidden_dim,
            gnn_output_dim=self.cfg.gnn_output_dim,
            gnn_readout_feature_dim=self.cfg.gnn_readout_feature_dim,
            gin_num_mlp_layers=self.cfg.gin_num_mlp_layers,
            gin_learn_eps=self.cfg.gin_learn_eps,
            gin_neighbor_pooling_type=self.cfg.gin_neighbor_pooling_type,
            gin_graph_pooling_type=self.cfg.gin_graph_pooling_type,
            actor_hidden_size=self.cfg.actor_hidden_size,
            critic_hidden_size=self.cfg.critic_hidden_size,
            action_dim=qtz.quartz_context.num_xfers,
            device=self.device,
        ).to(self.device)
        self.ac_net = cast(
            ActorCritic, nn.SyncBatchNorm.convert_sync_batchnorm(self.ac_net))
        self.ac_net_old = copy.deepcopy(self.ac_net)
        self.agent = PPOAgent(
            agent_id=self.rank,
            num_agents=self.ddp_processes,
            num_observers=self.cfg.obs_per_agent,
            device=self.device,
            batch_inference=self.cfg.batch_inference,
            invalid_reward=self.cfg.invalid_reward,
            cost_type=CostType.from_str(self.cfg.cost_type),
            ac_net=self.ac_net_old,
            input_graphs=self.input_graphs,
            dyn_eps_len=self.cfg.dyn_eps_len,
            max_eps_len=self.cfg.max_eps_len,
            min_eps_len=self.cfg.min_eps_len,
            output_full_seq=self.cfg.output_full_seq,
            output_dir=self.output_dir,
        )
        """use a holder class for convenience but split the model into 5 modules
        to avoid issue with BN + DDP 
        (https://github.com/pytorch/pytorch/issues/66504)
        """
        self.ddp_ac_net = self.ac_net.ddp_model()
        self.ddp_ac_net.eval()
        self.optimizer = torch.optim.Adam([{
            'params':
            self.ddp_ac_net.gnn.parameters(),
            'lr':
            self.cfg.lr_gnn,
        }, {
            'params':
            self.ddp_ac_net.actor_node.parameters(),
            'lr':
            self.cfg.lr_actor,
        }, {
            'params':
            self.ddp_ac_net.actor_xfer.parameters(),
            'lr':
            self.cfg.lr_actor,
        }, {
            'params':
            self.ddp_ac_net.actor_readout.parameters(),
            'lr':
            self.cfg.lr_actor,
        }, {
            'params':
            self.ddp_ac_net.critic.parameters(),
            'lr':
            self.cfg.lr_critic,
        }])
        if self.rank == 0:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                mode=self.wandb_mode,
                config=self.cfg,  # type: ignore
            )
        printfl(
            f'rank {self.rank} / {self.ddp_processes} on {self.device} initialized'
        )

        max_iterations = int(self.cfg.max_iterations)
        self.i_iter: int = 0
        self.tot_exps_collected: int = 0
        if self.cfg.resume:
            self.load_ckpt(self.cfg.ckpt_path)
        """train loop"""
        limited_time_budget: bool = len(self.cfg.time_budget) > 0
        if limited_time_budget:
            sec_budget: float = hms_to_sec(self.cfg.time_budget)
            printfl(
                f'rank {self.rank}: Time budget {self.cfg.time_budget} ( {sec_budget} sec ) is set.'
            )
        self.start_time_sec = time.time()
        while self.i_iter < max_iterations:
            self.train_iter()
            if self.i_iter % self.cfg.update_policy_interval == 0:
                self.ac_net_old.load_state_dict(self.ac_net.state_dict())
            if self.i_iter % self.cfg.save_ckpt_interval == 0:
                self.save_ckpt(f'iter_{self.i_iter}.pt')
            self.i_iter += 1
            used_sec = time.time() - self.start_time_sec
            if limited_time_budget and used_sec > sec_budget:
                printfl(
                    f'rank {self.rank}: Run out of time budget {used_sec} sec / {self.cfg.time_budget} ({used_sec} sec). Breaking training loop...'
                )
                break

    def train_iter(self) -> None:
        """collect batched data in dgl or tensor format"""
        s_time_collect = get_time_ns()
        self.agent.perpare_buf_for_next_iter()
        exp_list: ExperienceList
        # printfl(f'Agent {self.rank} : start collecting data for iter {self.i_iter}')
        if self.cfg.agent_collect is True:
            exp_list = self.agent.collect_data_by_self(
                self.cfg.num_eps_per_iter // self.ddp_processes,
                self.cfg.agent_batch_size, self.cfg.max_cost_ratio,
                self.cfg.nop_stop, self.cfg.greedy_sample)
        else:  # use observers to collect data
            collect_fn = partial(self.agent.collect_data,
                                 self.cfg.max_cost_ratio, self.cfg.nop_stop,
                                 self.cfg.greedy_sample)
            exp_list = collect_fn()
            # support the case that (self.agent_batch_size > self.cfg.obs_per_agent)
            for i in range(self.cfg.num_eps_per_iter //
                           (self.ddp_processes * self.cfg.obs_per_agent) - 1):
                exp_list += collect_fn()
        e_time_collect = get_time_ns()
        dur_s_collect = dur_ms(e_time_collect, s_time_collect) / 1e3
        self.tot_exps_collected += len(exp_list)
        # printfl(f'Agent {self.rank} : finish collecting data for iter {self.i_iter} in {dur_s_collect} s. |exp_list| = {len(exp_list)}')
        """log info about data collection"""
        # Each agent has different data, so it is DDP training
        if self.rank == 0:
            other_info_dict = self.agent.other_info_dict()
            collect_info = {
                **other_info_dict,  # type: ignore
                'iter':
                self.i_iter,
                'num_exps':
                len(exp_list),
                'tot_exps_collected_all_rank':
                self.tot_exps_collected * self.ddp_processes,
            }
            printfl(
                f'\n  Data for iter {self.i_iter} collected in {dur_s_collect} s .'
            )
            logprintfl(
                f'\n  Training lasted {sec_to_hms(time.time() - self.start_time_sec)} .'
            )
            for k, v in collect_info.items():
                printfl(f'    {k} : {v}')
            wandb.log(collect_info)
            pbar = tqdm(
                total=self.cfg.k_epochs *
                math.ceil(len(exp_list) / self.cfg.mini_batch_size),
                desc=f'Iter {self.i_iter}',
                bar_format='{desc} : {n}/{total} |{bar}| {elapsed} {postfix}',
            )
        """evaluate, compute loss, and update (DDP)"""
        """compute target values with sample returns"""
        """compute advantages by substracting values from sample returns"""
        all_target_values: List[float] = []
        all_advs: List[float] = []
        target_value: float = 0.0
        for reward, game_over in zip(reversed(exp_list.reward),
                                     reversed(exp_list.game_over)):
            if game_over:
                target_value = 0.0
            target_value = reward + self.cfg.gamma * target_value
            all_target_values.insert(0, target_value)
        all_advs = (torch.tensor(all_target_values) -
                    torch.tensor(exp_list.state_value)).tolist()
        train_exp_list = TrainExpList(*exp_list, all_target_values,
                                      all_advs)  # type: ignore
        """update the network for K epochs"""
        self.ddp_ac_net.train()
        for k_epoch in range(self.cfg.k_epochs):
            train_exp_list.shuffle()
            for i_step, exps in enumerate(
                    ExperienceListIterator(train_exp_list,
                                           self.cfg.mini_batch_size,
                                           self.device)):
                assert isinstance(exps, TrainBatchExp)
                self.optimizer.zero_grad()
                """evaluate states by Critic"""
                num_nodes: torch.LongTensor = exps.state.batch_num_nodes()
                b_node_embeds, b_readouts = self.ddp_ac_net(
                    exps.state, ActorCritic.gnn_name())
                state_values: torch.Tensor = self.ddp_ac_net(
                    b_readouts, ActorCritic.critic_name())
                """get node dist by Actor"""
                node_logprob_new_list: List[torch.Tensor] = []
                node_entropy_list: List[torch.Tensor] = []
                node_embed_list: List[torch.Tensor] = torch.split(
                    b_node_embeds, num_nodes.tolist(), dim=0)
                """concat node embeddings and global readouts"""
                new_node_embed_list: List[torch.Tensor] = []
                b_readout_features: torch.Tensor = self.ddp_ac_net(
                    b_readouts, ActorCritic.actor_readout_name())
                for node_embeds, num_node, readout_features in zip(
                        node_embed_list, num_nodes, b_readout_features):
                    readout_featuers_reaped: torch.Tensor = readout_features.repeat(
                        num_node, 1)
                    new_node_embed_list.append(
                        torch.cat([node_embeds, readout_featuers_reaped],
                                  dim=1))
                b_new_node_embeds = torch.cat(new_node_embed_list, dim=0)
                for node_embeds, node in zip(new_node_embed_list,
                                             exps.action[:, 0]):
                    node_logits: torch.Tensor = self.ddp_ac_net(
                        node_embeds, ActorCritic.actor_node_name()).squeeze(1)
                    node_probs: torch.Tensor = F.softmax(node_logits, dim=0)
                    node_dist = Categorical(probs=node_probs)
                    node_logprob_new_list.append(node_dist.log_prob(node))
                    node_entropy_list.append(node_dist.entropy())
                node_logprobs_new: torch.Tensor = torch.stack(
                    node_logprob_new_list, dim=0)
                node_entropy: torch.Tensor = torch.stack(node_entropy_list)
                """get selected node embeds"""
                nodes_offset: torch.LongTensor = torch.LongTensor(
                    [0] * num_nodes.shape[0]).to(self.device)  # type: ignore
                nodes_offset[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
                selected_nodes = exps.action[:, 0] + nodes_offset
                selected_node_embeds = b_new_node_embeds[selected_nodes]
                """get xfer dist by Actor"""
                # (batch_num_graphs, action_dim)
                xfer_logits: torch.Tensor = self.ddp_ac_net(
                    selected_node_embeds, ActorCritic.actor_xfer_name())
                softmax_xfer = masked_softmax(xfer_logits, exps.xfer_mask)
                xfer_dists = Categorical(softmax_xfer)
                # (batch_num_graphs, )
                xfer_logprobs_new: torch.Tensor = xfer_dists.log_prob(
                    exps.action[:, 1])
                xfer_entropys = xfer_dists.entropy()
                """compute loss for Actor (policy_net, theta)"""
                # prob ratio = (pi_theta / pi_theta__old)
                ratios = torch.exp(node_logprobs_new + xfer_logprobs_new -
                                   (exps.xfer_logprob + exps.node_logprob))
                surr1 = ratios * exps.advantages
                surr2 = torch.clamp(
                    ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip
                ) * exps.advantages  # NOTE: use fixed advantages
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                """compute loss for Critic (value_net, phi)"""
                critic_loss = torch.mean(
                    (exps.target_values - state_values)**2)
                """compute loss for Entropy"""
                xfer_entropy = torch.mean(xfer_entropys)
                node_entropy = torch.mean(node_entropy)
                """compute overall loss"""
                loss = actor_loss + 0.5 * critic_loss - self.cfg.entropy_coeff * (
                    xfer_entropy + node_entropy)
                """update"""
                loss.backward()
                for param in self.ddp_ac_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                """logging"""
                if self.rank == 0:
                    pbar.update(1)
                    log_dict = {
                        'actor_loss': float(actor_loss),
                        'critic_loss': float(critic_loss),
                        'xfer_entropy': float(xfer_entropy),
                        'node_entropy': float(node_entropy),
                        'loss': float(loss),
                    }
                    pbar.set_postfix({
                        **log_dict,
                    })
                    pbar.refresh()
                    wandb.log({
                        'actor_loss': actor_loss,
                        'critic_loss': critic_loss,
                        'xfer_entropy': xfer_entropy,
                        'node_entropy': node_entropy,
                        'loss': loss,
                    })
            # end for i_step
            # exps = None
            # torch.cuda.empty_cache()
        # end for k_epochs
        self.ddp_ac_net.eval()
        self.agent.sync_best_graph()

    def save_ckpt(self, ckpt_name: str, only_rank_zero: bool = True) -> None:
        # TODO save top-k model
        ckpt_dir = os.path.join(self.output_dir, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not only_rank_zero or self.rank == 0:
            torch.save(
                {
                    'i_iter': self.i_iter,
                    'model_state_dict': self.ac_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'loss': LOSS,
                },
                ckpt_path)
            printfl(f'saved "{ckpt_path}"!')

    def load_ckpt(self, ckpt_path: str) -> None:
        """load model and optimizer"""
        ckpt = torch.load(ckpt_path, map_location=self.agent.device)
        self.i_iter = int(ckpt['i_iter']) + 1
        model_state_dict = cast(OrderedDict[str, torch.Tensor],
                                ckpt['model_state_dict'])
        if self.cfg.load_non_ddp_ckpt:
            self.ac_net.load_state_dict(model_state_dict)
            self.ac_net_old.load_state_dict(model_state_dict)
            # the weights of self.ddp_ac_net should also be changed
            # ddp_v = list(self.ddp_ac_net.state_dict().values())
            # v = list(model_state_dict.values())
            # assert len(ddp_v) == len(v)
            # for i in range(len(v)):
            #     if not torch.equal(ddp_v[i], v[i]):
            #         ic(ddp_v[i])
            #         ic(v[i])
            #         raise ValueError
            #     else:
            #         printfl(f'{i} equal')
        else:
            self.ddp_ac_net.load_state_dict(model_state_dict)
            self.ac_net_old.load_state_dict(self.ac_net.state_dict())
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        printfl(f'resumed from "{ckpt_path}"!')
        """load best graph info"""
        if self.cfg.load_best_info:
            info_files = os.listdir(self.cfg.best_info_dir)
            for info_file in info_files:
                self.agent.load_best_info(
                    os.path.join(self.cfg.best_info_dir, info_file))

    def init_ddp_processes(self, rank: int, world_size: int) -> None:
        seed_all(self.cfg.seed)
        """init Quartz and other things"""
        if self.cfg.gpus is None or len(self.cfg.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.cfg.gpus[rank]}')
        torch.cuda.set_device(self.device)
        printfl(f'rank {rank} / {world_size} use {self.device}')
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{self.cfg.ddp_port}',
            rank=rank,
            world_size=world_size,
        )
        self.init_quartz_context_func()
        if self.cfg.omp_num_threads != 0:
            os.environ['OMP_NUM_THREADS'] = str(self.cfg.omp_num_threads)
        self.ac_net: ActorCritic = ActorCritic(
            gnn_type=self.cfg.gnn_type,
            num_gate_types=self.num_gate_types,
            gate_type_embed_dim=self.cfg.gate_type_embed_dim,
            gnn_num_layers=self.cfg.gnn_num_layers,
            gnn_hidden_dim=self.cfg.gnn_hidden_dim,
            gnn_output_dim=self.cfg.gnn_output_dim,
            gnn_readout_feature_dim=self.cfg.gnn_readout_feature_dim,
            gin_num_mlp_layers=self.cfg.gin_num_mlp_layers,
            gin_learn_eps=self.cfg.gin_learn_eps,
            gin_neighbor_pooling_type=self.cfg.gin_neighbor_pooling_type,
            gin_graph_pooling_type=self.cfg.gin_graph_pooling_type,
            actor_hidden_size=self.cfg.actor_hidden_size,
            critic_hidden_size=self.cfg.critic_hidden_size,
            action_dim=qtz.quartz_context.num_xfers,
            device=self.device,
        ).to(self.device)
        self.ddp_ac_net = self.ac_net.ddp_model()

    @torch.no_grad()
    def test(self, rank: int, world_size: int) -> None:
        self.cfg = cast(TestConfig, self.cfg)
        self.init_ddp_processes(rank, world_size)
        """load ckpt"""
        if self.cfg.resume:
            ckpt_path = self.cfg.ckpt_path
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model_state_dict = ckpt['model_state_dict']
            if self.cfg.load_non_ddp_ckpt:
                self.ac_net.load_state_dict(model_state_dict)
            else:
                self.ddp_ac_net.load_state_dict(model_state_dict)
            printfl(f'rank {rank} resumed from "{ckpt_path}"!')

        tester = Tester(
            cost_type=CostType.from_str(self.cfg.cost_type),
            ac_net=self.ac_net,  # type: ignore
            device=self.device,
            output_dir=self.output_dir,
            rank=rank,
        )
        """get input graphs"""
        input_graphs: List[InputGraph] = []
        if len(self.cfg.input_graphs) > 0:
            input_graphs = self.cfg.input_graphs
        else:
            files: List[str] = cast(
                List[str], natsorted(os.listdir(self.cfg.input_graph_dir)))
            for file in files:
                qasm_path = os.path.join(self.cfg.input_graph_dir, file)
                name = file.split('.')[0]
                input_graphs.append(InputGraph(name, qasm_path))
        n_graph_each_rank = math.ceil(len(input_graphs) / world_size)
        input_graphs = input_graphs[n_graph_each_rank *
                                    rank:n_graph_each_rank * (rank + 1)]

        info_list: List[str] = []
        for input_graph in input_graphs:
            with open(input_graph.path) as f:
                qasm_str = f.read()
            graph = qtz.qasm_to_graph(qasm_str)
            printfl(f'rank {rank} Testing {input_graph.name}...')
            best_cost, best_time = tester.beam_search(graph, self.cfg.topk,
                                                      self.cfg.max_eps_len,
                                                      input_graph.name,
                                                      self.cfg.budget)
            info = f'{input_graph.name} : {best_cost}  {best_time} seconds ({sec_to_hms(best_time)})'
            printfl(f'rank {rank} Test Result: {info}')
            info_list.append(info)
        printfl(f'\n\n rank {rank} All Test Results:\n')
        for info in info_list:
            printfl(info)

    def convert(self, rank: int) -> None:
        self.cfg = cast(ConvertConfig, self.cfg)
        self.init_ddp_processes(rank, 2)
        if rank == 0:
            """load ckpt"""
            ckpt_path = self.cfg.ckpt_path
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model_state_dict = ckpt['model_state_dict']
            self.ddp_ac_net.load_state_dict(model_state_dict)
            printfl(f'"{ckpt_path}" is loaded!')

            out_path: str
            if self.cfg.ckpt_output_path == '':
                ckpt_fname = os.path.basename(ckpt_path)
                os.makedirs(self.cfg.ckpt_output_dir, exist_ok=True)
                out_path = os.path.join(self.cfg.ckpt_output_dir,
                                        f'converted_{ckpt_fname}')
            else:
                out_path = self.cfg.ckpt_output_path
            if os.path.exists(out_path):
                for i in range(int(1e8)):
                    posb_path = f'{out_path}.{i}'
                    if not os.path.exists(posb_path):
                        out_path = posb_path
                        break
            torch.save(self.ac_net.state_dict(), out_path)
            printfl(f'saved "{out_path}"!')


@hydra.main(config_path='config', config_name='config')
def main(config: Config) -> None:
    output_dir = os.path.abspath(os.curdir)  # get hydra output dir
    os.chdir(
        hydra.utils.get_original_cwd())  # set working dir to the original one

    cfg: BaseConfig = config.c
    warnings.simplefilter('ignore')

    ppo_mod = PPOMod(cfg, output_dir)

    ddp_processes = 1
    if len(cfg.gpus) > 1:
        ddp_processes = len(cfg.gpus)
    mp.set_start_method(cfg.mp_start_method)
    if cfg.mode == 'train':
        obs_processes = ddp_processes * cfg.obs_per_agent
        tot_processes = ddp_processes + obs_processes
        print(f'spawning {tot_processes} processes...')
        mp.spawn(
            fn=ppo_mod.init_process,
            args=(
                ddp_processes,
                obs_processes,
            ),
            nprocs=tot_processes,
            join=True,
        )
    elif cfg.mode == 'test':
        mp.spawn(
            fn=ppo_mod.test,
            args=(ddp_processes, ),
            nprocs=ddp_processes,
            join=True,
        )
    elif cfg.mode == 'convert':
        mp.spawn(
            fn=ppo_mod.convert,
            args=(),
            nprocs=2,
            join=True,
        )
    else:
        raise NotImplementedError(f'Unexpected mode {cfg.mode}')


if __name__ == '__main__':
    main()
