# this file is under mypy's checking
from __future__ import annotations
import os
from typing import List, Dict
import warnings
from functools import partial
import time
import copy
import math
from tqdm import tqdm # type: ignore

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
from model import ActorCritic
from actor import PPOAgent

from IPython import embed # type: ignore
from icecream import ic # type: ignore

class PPOMod:
    
    def __init__(
        self, cfg: BaseConfig, output_dir: str
    ) -> None:
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
        self.num_gate_type: int = cfg.num_gate_type
        
    def print_cfg(self) -> None:
        print('================ Configs ================')
        print(OmegaConf.to_yaml(self.cfg))
        # for k, v in self.cfg.items():
        #     print(f'{k} : {v}')
        print(f'output_dir : {self.output_dir}')
        print('=========================================')
    
    def init_process(self, rank: int, ddp_processes: int, obs_processes: int) -> None:
        seed_all(self.cfg.seed + rank)
        """init Quartz for each process"""
        global quartz_context
        global quartz_parser
        self.init_quartz_context_func()
        
        """set num of OMP threads to avoid blasting the machine"""
        if self.cfg.omp_num_threads != 0:
            os.environ["OMP_NUM_THREADS"] = str(self.cfg.omp_num_threads)
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
                name=agent_name, rank=rank, world_size=tot_processes,
                rpc_backend_options=rpc_backend_options,
            )
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://localhost:{self.cfg.ddp_port}',
                rank=rank, world_size=ddp_processes,
            )
            self.train()
        else:
            """init observer processes"""
            obs_rank = rank - ddp_processes
            agent_rref_id = obs_rank // self.cfg.obs_per_agent
            obs_in_agent_rank = obs_rank % self.cfg.obs_per_agent
            obs_name = get_obs_name(agent_rref_id, obs_in_agent_rank)
            rpc.init_rpc(
                name=obs_name, rank=rank, world_size=tot_processes,
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
        self.ac_net = ActorCritic(
            num_gate_type=self.num_gate_type,
            graph_embed_size=self.cfg.graph_embed_size,
            actor_hidden_size=self.cfg.actor_hidden_size,
            critic_hidden_size=self.cfg.critic_hidden_size,
            action_dim=qtz.quartz_context.num_xfers,
            device=self.device,
        ).to(self.device)
        self.ac_net_old = copy.deepcopy(self.ac_net)
        # NOTE should not use self.ac_net later
        self.agent = PPOAgent(
            agent_id=self.rank,
            num_agents=self.ddp_processes,
            num_observers=self.cfg.obs_per_agent,
            device=self.device,
            batch_inference=self.cfg.batch_inference,
            invalid_reward=self.cfg.invalid_reward,
            ac_net=self.ac_net_old,
            input_graphs=self.input_graphs,
            softmax_temp_en=self.cfg.softmax_temp_en,
            hit_rate=self.cfg.hit_rate,
            dyn_eps_len=self.cfg.dyn_eps_len,
            max_eps_len=self.cfg.max_eps_len,
            min_eps_len=self.cfg.min_eps_len,
            output_dir=self.output_dir,
        )
        self.ddp_ac_net = DDP(self.ac_net, device_ids=[self.device])
        self.optimizer = torch.optim.Adam([
            {
                'params': self.ddp_ac_net.module.graph_embedding.parameters(), # type: ignore
                'lr': self.cfg.lr_graph_embedding,
            },
            {
                'params': self.ddp_ac_net.module.actor.parameters(), # type: ignore
                'lr': self.cfg.lr_actor,
            },
            {
                'params': self.ddp_ac_net.module.critic.parameters(), # type: ignore
                'lr': self.cfg.lr_critic,
            }
        ])
        if self.rank == 0:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                mode=self.wandb_mode,
                config=self.cfg, # type: ignore
            )
        printfl(f'rank {self.rank} on {self.device} initialized')
        
        max_iterations = int(self.cfg.max_iterations)
        self.i_iter = 0
        if self.cfg.resume:
            self.load_ckpt(self.cfg.ckpt_path)
        """train loop"""
        self.start_time_sec = time.time()
        while self.i_iter < max_iterations:
            self.train_iter()
            if self.i_iter % self.cfg.update_policy_interval == 0:
                self.ac_net_old.load_state_dict(self.ddp_ac_net.module.state_dict())
            if self.i_iter % self.cfg.save_ckpt_interval == 0:
                self.save_ckpt(f'iter_{self.i_iter}.pt')
            self.i_iter += 1

        
    def train_iter(self) -> None:
        """collect batched data in dgl or tensor format"""
        s_time_collect = get_time_ns()
        self.agent.perpare_buf_for_next_iter()
        if self.cfg.agent_collect is True:
            collect_fn = self.agent.collect_data_self
        else: # use observers to collect data
            collect_fn = self.agent.collect_data
        printfl(f'Agent {self.rank} : start collecting data for iter {self.i_iter}')
        exp_list: ExperienceList = collect_fn(self.cfg.max_gate_count_ratio, self.cfg.nop_stop)
        # support the case that (self.agent_batch_size > self.cfg.obs_per_agent)
        for i in range(self.cfg.num_eps_per_iter // self.cfg.obs_per_agent - 1):
            exp_list += collect_fn(self.cfg.max_gate_count_ratio, self.cfg.nop_stop)
        e_time_collect = get_time_ns()
        dur_s_collect = dur_ms(e_time_collect, s_time_collect) / 1e3
        printfl(f'Agent {self.rank} : finish collecting data for iter {self.i_iter} in {dur_s_collect} s. |exp_list| = {len(exp_list)}')
        """evaluate, compute loss, and update (DDP)"""
        # Each agent has different data, so it is DDP training
        if self.rank == 0:
            other_info_dict = self.agent.other_info_dict()
            collect_info = {
                **other_info_dict, # type: ignore
                'num_exps': len(exp_list),
            }
            printfl(f'\n  Data for iter {self.i_iter} collected in {dur_s_collect} s .')
            logprintfl(f'\n  Training lasted {sec_to_hms(time.time() - self.start_time_sec)} .')
            for k, v in collect_info.items():
                printfl(f'    {k} : {v}')
            wandb.log(other_info_dict)
            pbar = tqdm(
                total=self.cfg.k_epochs * math.ceil(len(exp_list) / self.cfg.mini_batch_size),
                desc=f'Iter {self.i_iter}',
                bar_format='{desc} : {n}/{total} |{bar}| {elapsed} {postfix}',
            )
        
        for epoch_k in range(self.cfg.k_epochs):
            for i_step, exps in enumerate(
                ExperienceListIterator(exp_list, self.cfg.mini_batch_size, self.device)
            ):
                self.optimizer.zero_grad()
                """get embeds of seleted nodes and evaluate them by Critic"""
                num_nodes: torch.LongTensor = exps.state.batch_num_nodes()
                # (batch_num_nodes, embed_dim)
                b_graph_embeds: torch.Tensor = self.ddp_ac_net(exps.state, ActorCritic.graph_embedding_name())
                nodes_offset: torch.LongTensor = torch.LongTensor([0] * num_nodes.shape[0]).to(self.device) # type: ignore
                nodes_offset[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
                selected_nodes = exps.action[:, 0] + nodes_offset
                selected_node_embeds = b_graph_embeds[selected_nodes]
                selected_node_values: torch.Tensor = self.ddp_ac_net(selected_node_embeds, ActorCritic.critic_name()).squeeze()
                """get xfer dist by Actor"""
                # (batch_num_graphs, action_dim)
                xfer_logits: torch.Tensor = self.ddp_ac_net(selected_node_embeds, ActorCritic.actor_name())
                softmax_xfer = masked_softmax(xfer_logits, exps.xfer_mask)
                xfer_dists = Categorical(softmax_xfer)
                # (batch_num_graphs, )
                xfer_logprobs: torch.Tensor = xfer_dists.log_prob(exps.action[:, 1])
                xfer_entropys = xfer_dists.entropy()
                """get embeds of next nodes and evaluate them by Critic without grad"""
                with torch.no_grad():
                    # (num_next_graphs, )
                    next_num_nodes: torch.LongTensor = exps.next_state.batch_num_nodes()
                    """get embeds"""
                    # (batch_next_graphs_nodes, embed_dim)
                    b_next_graph_embeds: torch.Tensor = self.ddp_ac_net(exps.next_state, ActorCritic.graph_embedding_name())
                    next_graph_embeds_list: List[torch.Tensor] = torch.split(b_next_graph_embeds, next_num_nodes.tolist())
                    """select embeds"""
                    # ( sum(num_next_nodes), embed_dim )
                    next_node_embeds: torch.Tensor = torch.cat([
                        graph_embed[next_node_ids]
                        for (next_node_ids, graph_embed) in zip(exps.next_nodes, next_graph_embeds_list)
                    ])
                    """evaluate"""
                    # ( sum(num_next_nodes), )
                    next_node_values: torch.Tensor = self.ddp_ac_net(next_node_embeds, ActorCritic.critic_name()).squeeze()
                    num_next_nodes = list(map(len, exps.next_nodes))
                    next_node_values_list: List[torch.Tensor] = torch.split(next_node_values, num_next_nodes)
                    """get max next value for each graph"""
                    max_next_values_list: List[torch.Tensor] = []
                    for i in range(len(exps)):
                        max_next_value: torch.Tensor
                        # invalid xfer, gate count exceeds limit, NOP
                        if next_node_values_list[i].shape[0] == 0 or \
                            exps.game_over[i] or \
                            qtz.is_nop(int(exps.action[i, 1])) and self.cfg.nop_stop:
                            max_next_value = torch.zeros(1).to(self.device)
                        else:
                            max_next_value, _ = torch.max(next_node_values_list[i], dim=0, keepdim=True)
                        max_next_values_list.append(max_next_value)
                    max_next_values = torch.cat(max_next_values_list)
                # end with
                """compute loss for Actor (policy_net, theta)"""
                # prob ratio = (pi_theta / pi_theta__old)
                ratios = torch.exp(xfer_logprobs - exps.xfer_logprob)
                advantages = exps.reward + self.cfg.gamma * max_next_values - selected_node_values
                surr1 = ratios * advantages.detach()
                surr2 = torch.clamp(
                    ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip
                ) * advantages.detach()
                actor_loss = - torch.sum(torch.min(surr1, surr2)) / len(exp_list)
                """compute loss for Critic (value_net, phi)"""
                critic_loss = torch.sum(advantages ** 2) / len(exp_list)
                xfer_entropy = torch.sum(xfer_entropys) / len(exp_list)
                """compute overall loss"""
                loss = actor_loss + 0.5 * critic_loss - self.cfg.entropy_coeff * xfer_entropy
                """update"""
                loss.backward()
                for param in self.ddp_ac_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step() # confirm
                """logging"""
                if self.rank == 0:
                    pbar.update(1)
                    log_dict = {
                        'actor_loss': float(actor_loss),
                        'critic_loss': float(critic_loss),
                        'xfer_entropy': float(xfer_entropy),
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
                        'loss': loss,
                    })
            # end for i_step
        # end for k_epochs
        self.agent.sync_best_graph()
        
    def save_ckpt(self, ckpt_name: str, only_rank_zero: bool = True) -> None:
        # TODO save top-k model
        ckpt_dir = os.path.join(self.output_dir, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not only_rank_zero or self.rank == 0:
            torch.save({
                'i_iter': self.i_iter,
                'model_state_dict': self.ddp_ac_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'loss': LOSS,
            }, ckpt_path)
            printfl(f'saved "{ckpt_path}"!')
        
    def load_ckpt(self, ckpt_path: str) -> None:
        """load model and optimizer"""
        ckpt = torch.load(ckpt_path, map_location=self.agent.device)
        self.i_iter = ckpt['i_iter']
        model_state_dict = ckpt['model_state_dict']
        if self.cfg.load_non_ddp_ckpt:
            self.ddp_ac_net.module.load_state_dict(model_state_dict)
            self.ac_net_old.load_state_dict(model_state_dict)
        else:
            self.ddp_ac_net.load_state_dict(model_state_dict)
            self.ac_net_old.load_state_dict(self.ddp_ac_net.module.state_dict())
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        printfl(f'resumed from "{ckpt_path}"!')
        """load best graph info"""
        if self.cfg.load_best_info:
            info_files = os.listdir(self.cfg.best_info_dir)
            for info_file in info_files:
                self.agent.load_best_info(
                    os.path.join(self.cfg.best_info_dir, info_file)
                )

@hydra.main(config_path='config', config_name='config')
def main(config: Config) -> None:
    output_dir = os.path.abspath(os.curdir) # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd()) # set working dir to the original one
    
    cfg: BaseConfig = config.c
    warnings.simplefilter('ignore')
    
    ppo_mod = PPOMod(cfg, output_dir)
    
    mp.set_start_method(cfg.mp_start_method)
    ddp_processes = 1
    if len(cfg.gpus) > 1:
        ddp_processes = len(cfg.gpus)
    obs_processes = ddp_processes * cfg.obs_per_agent
    tot_processes = ddp_processes + obs_processes
    print(f'spawning {tot_processes} processes...')
    mp.spawn(
        fn=ppo_mod.init_process,
        args=(ddp_processes, obs_processes,),
        nprocs=tot_processes,
        join=True,
    )
    # TODO profiling; some parts of this code are slow
    
if __name__ == '__main__':
    main()
