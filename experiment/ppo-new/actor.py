# this file is under mypy's checking
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import threading
from typing import Any, Dict, List, Tuple

import dgl  # type: ignore
import qtz
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import wandb
from ds import *
from icecream import ic  # type: ignore
from IPython import embed  # type: ignore
from model.actor_critic import ActorCritic
from omegaconf.dictconfig import DictConfig
from torch.distributions import Categorical
from torch.futures import Future
from utils import *

# import quartz # type: ignore


class Observer:
    def __init__(
        self,
        obs_id: int,
        agent_id: int,
        batch_inference: bool,
        invalid_reward: float,
        cost_type: CostType,
    ) -> None:
        self.id = obs_id
        self.agent_id = agent_id
        self.batch_inference = batch_inference
        self.invalid_reward = invalid_reward
        self.cost_type = cost_type

    def run_episode(
        self,
        agent_rref: rpc.RRef[PPOAgent],
        init_state_str: str,
        original_state_str: str,  # input graph, used to limit the cost
        max_eps_len: int,
        max_extra_cost: int,
        nop_stop: bool,
    ) -> List[SerializableExperience]:
        """Interact with env for many steps to collect data.
        If `batch_inference` is `True`, we call `select_action_batch` of the agent,
        so we have to run a fixed number of steps.
        If an episode stops in advance, we start a new one to continue running.

        If `batch_inference` is `False`, we can run just one episode.

        Returns:
            List[SerializableExperience]: a list of experiences collected in this function
        """
        init_graph = qtz.qasm_to_graph(init_state_str)
        original_graph = qtz.qasm_to_graph(original_state_str)
        original_cost = get_cost(init_graph, self.cost_type)
        exp_list: List[SerializableExperience] = []

        graph = init_graph
        graph_str = init_state_str
        info: Dict[str, Any] = {'start': True}
        last_eps_end: int = -1
        for i_step in range(max_eps_len):
            """get action (action_node, xfer_dist) from agent"""
            actmp: ActionTmp
            if self.batch_inference:
                actmp = agent_rref.rpc_sync().select_action_batch(
                    self.id,
                    graph.to_qasm_str(),
                )
            else:
                actmp = agent_rref.rpc_sync().select_action(
                    self.id,
                    graph.to_qasm_str(),
                )
            """sample action_xfer with mask"""
            av_xfers = graph.available_xfers_parallel(
                context=qtz.quartz_context, node=graph.get_node_from_id(id=actmp.node)
            )
            av_xfer_mask = torch.BoolTensor([0] * qtz.quartz_context.num_xfers)
            av_xfer_mask[av_xfers] = True
            # (action_dim, )  only sample from available xfers
            softmax_xfer = masked_softmax(actmp.xfer_dist, av_xfer_mask)
            xfer_dist = Categorical(softmax_xfer)
            action_xfer = xfer_dist.sample()
            action_xfer_logp: torch.Tensor = xfer_dist.log_prob(action_xfer)

            """apply action"""
            action = Action(actmp.node, action_xfer.item())
            next_nodes: List[int]
            next_graph, next_nodes = graph.apply_xfer_with_local_state_tracking(
                xfer=qtz.quartz_context.get_xfer_from_id(id=action.xfer),
                node=graph.get_node_from_id(id=action.node),
                eliminate_rotation=qtz.has_parameterized_gate,
            )
            """parse result, compute reward"""
            if next_graph is None:
                reward = self.invalid_reward
                game_over = True
                next_graph_str = graph_str  # CONFIRM placeholder?
            elif qtz.is_nop(action.xfer):
                reward = 0
                game_over = nop_stop
                next_graph_str = graph_str  # unchanged
                next_nodes = [action.node]  # CONFIRM
            else:
                graph_cost = get_cost(graph, self.cost_type)
                next_graph_cost = get_cost(next_graph, self.cost_type)
                reward = graph_cost - next_graph_cost
                game_over = graph_cost > original_cost + max_extra_cost  # or (
                #     graph.gate_count > original_graph.gate_count + max_extra_cost
                # )
                next_graph_str = next_graph.to_qasm_str()

            exp = SerializableExperience(
                graph_str,
                action,
                reward,
                next_graph_str,
                game_over,
                actmp.node_value,
                next_nodes,
                av_xfer_mask,
                action_xfer_logp.item(),
                copy.deepcopy(info),
            )
            exp_list.append(exp)
            info['start'] = False
            if game_over:
                if self.batch_inference:
                    graph = init_graph
                    graph_str = init_state_str
                    info['start'] = True
                    last_eps_end = i_step
                else:
                    break
            else:
                graph = next_graph
                graph_str = next_graph_str
        # end for
        if i_step - last_eps_end >= max_eps_len:
            exp_list[-1].game_over = True  # eps len exceeds limit
        return exp_list


class PPOAgent:
    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        num_observers: int,
        device: torch.device,
        batch_inference: bool,
        invalid_reward: float,
        limit_total_gate_count: bool,
        cost_type: CostType,
        ac_net: ActorCritic,
        input_graphs: List[Dict[str, str]],
        softmax_temp_en: bool,
        hit_rate: float,
        dyn_eps_len: bool,
        max_eps_len: int,
        min_eps_len: int,
        subgraph_opt: bool,
        xfer_pred_layers: int,
        output_full_seq: bool,
        output_dir: str,
        vmem_perct_limit: float,
    ) -> None:
        self.id = agent_id
        self.num_agents = num_agents
        self.device = device
        self.output_full_seq = output_full_seq
        self.output_dir = output_dir

        self.dyn_eps_len = dyn_eps_len
        self.max_eps_len = max_eps_len
        self.min_eps_len = min_eps_len
        self.cost_type = cost_type
        self.invalid_reward = invalid_reward
        self.limit_total_gate_count = limit_total_gate_count
        self.subgraph_opt = subgraph_opt
        self.xfer_pred_layers = xfer_pred_layers

        """networks related"""
        self.ac_net = ac_net  # NOTE: just a ref
        self.softmax_temp_en = softmax_temp_en
        self.hit_rate = hit_rate

        """init Observers on the other processes and hold the refs to them"""
        self.obs_rrefs: List[rpc.RRef] = []
        for obs_rank in range(0, num_observers):
            ob_info = rpc.get_worker_info(get_obs_name(self.id, obs_rank))
            self.obs_rrefs.append(
                rpc.remote(
                    ob_info,
                    Observer,
                    args=(
                        obs_rank,
                        self.id,
                        batch_inference,
                        invalid_reward,
                        self.cost_type,
                    ),
                )
            )

        self.graph_buffers: List[GraphBuffer] = [
            GraphBuffer(
                input_graph['name'],
                input_graph['qasm'],
                self.cost_type,
                self.device,
                vmem_perct_limit=vmem_perct_limit,
            )
            for input_graph in input_graphs
        ]
        self.init_buffer_turn: int = 0

        """helper vars for select_action"""
        self.future_actions: Future[List[ActionTmp]] = Future()
        self.pending_states = len(self.obs_rrefs)
        self.states_buf: List[dgl.DGLGraph] = [None] * len(self.obs_rrefs)
        self.lock = threading.Lock()

    @torch.no_grad()
    def select_action(self, obs_id: int, state_str: str) -> ActionTmp:
        self.ac_net.eval()
        """respond to a single query"""
        pygraph: quartz.PyGraph = qtz.qasm_to_graph(state_str)
        dgl_graph: dgl.DGLGraph = pygraph.to_dgl_graph().to(self.device)
        num_nodes: int = dgl_graph.num_nodes()
        """compute embeds and use Critic to evaluate each node"""
        node_embeds: torch.Tensor = self.ac_net.gnn(dgl_graph)
        node_values: torch.Tensor = self.ac_net.critic(node_embeds).squeeze()
        temperature: float
        if not self.softmax_temp_en:
            temperature = 1.0
        else:
            temperature = 1 / (
                math.log(self.hit_rate * (num_nodes - 1) / (1 - self.hit_rate))
            )
        softmax_node_values = F.softmax(node_values / temperature, dim=0)
        action_node = int(torch.multinomial(softmax_node_values, 1))
        """use Actor to evaluate xfers for the sampled node"""
        action_node_embed = node_embeds[action_node]
        xfer_logits: torch.Tensor = self.ac_net.actor(action_node_embed).cpu()

        return ActionTmp(action_node, float(node_values[action_node]), xfer_logits)

    @rpc.functions.async_execution
    @torch.no_grad()
    def select_action_batch(self, obs_id: int, state_str: str) -> Future[ActionTmp]:
        """inference a batch of queries queried by all of the observers at once"""
        future_action: Future[ActionTmp] = self.future_actions.then(
            lambda future_actions: future_actions.wait()[obs_id]
        )  # this single action is returned for the obs that calls this function
        # It is available after self.future_actions is set
        pygraph: quartz.PyGraph = qtz.qasm_to_graph(state_str)
        dgl_graph: dgl.DGLGraph = pygraph.to_dgl_graph().to(self.device)
        if self.states_buf[obs_id] is None:
            self.states_buf[obs_id] = dgl_graph
        else:
            raise Exception(
                f'Unexpected: self.states_buf[{obs_id}] is not None! Duplicate assignment occurs!'
            )

        with self.lock:  # avoid data race on self.pending_states
            self.pending_states -= 1
            if self.pending_states == 0:
                self.ac_net.eval()
                """collected a batch, start batch inference"""
                b_state: dgl.DGLGraph = dgl.batch(self.states_buf)
                num_nodes: torch.Tensor = (
                    b_state.batch_num_nodes()
                )  # (num_graphs, ) assert each elem > 0
                """compute embeds and use Critic to evaluate each node"""
                # (batch_num_nodes, embed_dim)
                b_node_embeds: torch.Tensor = self.ac_net.gnn(b_state)
                # (batch_num_nodes, )
                b_node_values: torch.Tensor = self.ac_net.critic(
                    b_node_embeds
                ).squeeze()
                # list with length num_graphs; each member is a tensor of node values in a graph
                node_values_list: List[torch.Tensor] = torch.split(
                    b_node_values, num_nodes.tolist()
                )
                """sample node by softmax with temperature for each graph as a batch"""
                # (num_graphs, max_num_nodes)
                b_node_values_pad = nn.utils.rnn.pad_sequence(
                    node_values_list, batch_first=True, padding_value=-math.inf
                )
                # (num_graphs, )
                temperature: torch.Tensor
                if not self.softmax_temp_en:
                    temperature = torch.ones(1).to(self.device)
                else:
                    temperature = 1 / (
                        torch.log(self.hit_rate * (num_nodes - 1) / (1 - self.hit_rate))
                    )

                b_softmax_node_values_pad = F.softmax(
                    b_node_values_pad / temperature.unsqueeze(1), dim=-1
                )
                b_sampled_nodes = torch.multinomial(
                    b_softmax_node_values_pad, 1
                ).flatten()
                """collect embeddings of sampled nodes"""
                # (num_graphs, )
                node_offsets = torch.zeros(
                    b_sampled_nodes.shape[0], dtype=torch.long
                ).to(self.device)
                node_offsets[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
                sampled_node_ids = b_sampled_nodes + node_offsets
                # (num_graphs, embed_dim)
                sampled_node_embeds = b_node_embeds[sampled_node_ids]
                """use Actor to evaluate xfers for sampled nodes"""
                # (num_graphs, action_dim)
                xfer_logits: torch.Tensor = self.ac_net.actor(sampled_node_embeds).cpu()
                # return the xfer dist. to observers who are responsible for sample xfer with masks
                actions = [
                    ActionTmp(
                        node=int(b_sampled_nodes[i]),
                        node_value=float(node_values_list[i][b_sampled_nodes[i]]),
                        xfer_dist=xfer_logits[i],
                    )
                    for i in range(len(self.obs_rrefs))
                ]
                this_future_actions = (
                    self.future_actions
                )  # NOTE something magic to avoid duplicate assignment
                """re-init"""
                self.pending_states = len(self.obs_rrefs)
                self.future_actions = Future()
                self.states_buf = [None] * len(self.obs_rrefs)

                this_future_actions.set_result(actions)
        # end with
        return future_action  # return a future

    def perpare_buf_for_next_iter(self) -> None:
        for buffer in self.graph_buffers:
            buffer.prepare_for_next_iter()

    def other_info_dict(self) -> Dict[str, float | int]:
        info_dict: Dict[str, float | int] = {}
        for buffer in self.graph_buffers:
            best_graph_info: Dict[str, int] = {
                'cost': get_cost(buffer.best_graph, self.cost_type),
                'gate_count': buffer.best_graph.gate_count,
                'cx_count': buffer.best_graph.cx_count,
                'depth': buffer.best_graph.depth,
            }
            for metric_name, value in best_graph_info.items():
                info_dict[f'{buffer.name}_best_graph_{metric_name}'] = value

            eps_len_info = buffer.eps_len_info()
            for k, v in eps_len_info.items():
                info_dict[f'{buffer.name}_{k}'] = v

            rewards_info = buffer.rewards_info()
            for k, v in rewards_info.items():
                info_dict[f'{buffer.name}_{k}'] = v

            cost_info = buffer.cost_info()
            for k, v in cost_info.items():
                info_dict[f'{buffer.name}_{k}'] = v

            basic_info = buffer.basic_info()
            for k, v in basic_info.items():
                info_dict[f'{buffer.name}_{k}'] = v

        # end for

        return info_dict

    def output_opt_path(
        self,
        name: str,
        best_gc: int,
        exp_seq: List[Tuple[SerializableExperience, quartz.PyGraph, quartz.PyGraph]],
    ) -> str:
        output_dir = os.path.join(self.output_dir, name, f'{best_gc}_{self.id}')
        os.makedirs(output_dir)
        """make a s_exp to output the starting graph"""
        first_s_exp = SerializableExperience.new_empty()
        first_s_exp.action = Action(0, 0)
        first_s_exp.reward = 0.0
        first_s_exp.next_state = exp_seq[0][0].state
        exp_seq = [(first_s_exp, None, exp_seq[0][1])] + exp_seq
        """output the seq"""
        for i_step, (s_exp, graph, next_graph) in enumerate(exp_seq):
            fname = (
                f'{i_step}_{get_cost(next_graph, self.cost_type)}_{int(s_exp.reward)}_'
                f'{s_exp.action.node}_{s_exp.action.xfer}.qasm'
            )
            with open(os.path.join(output_dir, fname), 'w') as f:
                if not isinstance(s_exp.next_state, str):
                    s_exp.next_state = s_exp.next_state.to_qasm_str()

                f.write(s_exp.next_state)
        return output_dir

    @torch.no_grad()
    def collect_data(
        self,
        max_extra_cost: int,
        nop_stop: bool,
        greedy_sample: bool,
    ) -> ExperienceList:
        """collect experiences from observers"""
        future_exp_lists: List[Future[List[SerializableExperience]]] = []
        init_buffer_ids: List[int] = []
        init_graph_qasm_list: List[str] = []
        original_graph_qasm_list: List[str] = []
        max_eps_len_for_all: int = 0
        for obs_rref in self.obs_rrefs:
            """sample init state"""
            graph_buffer = self.graph_buffers[self.init_buffer_turn]
            init_graph: quartz.PyGraph = graph_buffer.sample(greedy_sample)
            init_buffer_ids.append(self.init_buffer_turn)
            self.init_buffer_turn = (self.init_buffer_turn + 1) % len(
                self.graph_buffers
            )
            if self.dyn_eps_len:
                max_len = graph_buffer.max_eps_length
                if max_len < 40:
                    max_eps_len = math.ceil(max_len * 1.5)
                else:
                    max_eps_len = math.ceil(max_len * 1.2)
                max_eps_len = max(max_eps_len, self.min_eps_len)
                max_eps_len = min(max_eps_len, self.max_eps_len)
            else:
                max_eps_len = self.max_eps_len
            init_graph_qasm_list.append(init_graph.to_qasm_str())
            original_graph_qasm_list.append(graph_buffer.original_graph.to_qasm_str())
            max_eps_len_for_all = max(max_eps_len_for_all, max_eps_len)
        # end for
        """communicate with others to get max of max_eps_len_for_all"""
        max_eps_len_all_ranks = torch.zeros(self.num_agents).to(self.device)
        max_eps_len_all_ranks[self.id] = max_eps_len_for_all
        for r in range(self.num_agents):
            dist.broadcast(max_eps_len_all_ranks[r], r)
        max_eps_len_for_all = int(max_eps_len_all_ranks.max())
        """make async RPC to kick off an episode on observers"""
        for obs_rref, init_qasm, orig_qasm in zip(
            self.obs_rrefs, init_graph_qasm_list, original_graph_qasm_list
        ):
            future_exp_lists.append(
                obs_rref.rpc_async().run_episode(
                    rpc.RRef(self),
                    init_qasm,
                    orig_qasm,
                    max_eps_len_for_all,  # make sure all observers have the same max_eps_len
                    max_extra_cost,
                    nop_stop,
                )
            )

        # wait until all obervers have finished their episode
        s_exp_lists: List[List[SerializableExperience]] = torch.futures.wait_all(
            future_exp_lists
        )
        """convert graph and maintain infos in graph_buffer"""
        state_dgl_list: List[dgl.DGLGraph] = []
        next_state_dgl_list: List[dgl.DGLGraph] = []

        for buffer_id, obs_res in zip(init_buffer_ids, s_exp_lists):
            """for each observer's results (several episodes)"""
            graph_buffer = self.graph_buffers[buffer_id]
            init_graph = None
            init_graph_cost: int
            exp_seq: List[
                Tuple[SerializableExperience, quartz.PyGraph, quartz.PyGraph]
            ] = []  # for output optimization path
            for s_exp in obs_res:
                """for each experience"""
                graph = qtz.qasm_to_graph(s_exp.state)
                next_graph = qtz.qasm_to_graph(s_exp.next_state)
                state_dgl_list.append(graph.to_dgl_graph())
                next_state_dgl_list.append(next_graph.to_dgl_graph())
                graph_cost = get_cost(graph, self.cost_type)
                next_graph_cost = get_cost(next_graph, self.cost_type)
                """collect info"""
                if s_exp.info['start']:
                    init_graph = graph
                    init_graph_cost = graph_cost
                    exp_seq = []
                    i_step = 0
                    graph_buffer.rewards.append([])
                    graph_buffer.append_init_costs_from_graph(init_graph)
                    graph_buffer.append_costs_from_graph(init_graph)
                exp_seq.append((s_exp, graph, next_graph))

                """add graphs into buffer"""
                if (
                    not s_exp.game_over
                    and not qtz.is_nop(s_exp.action.xfer)
                    and next_graph_cost <= init_graph_cost
                ):  # NOTE: only add graphs with less or equal gate count
                    graph_buffer.push_back(next_graph)

                graph_buffer.rewards[-1].append(s_exp.reward)
                graph_buffer.append_costs_from_graph(next_graph)
                """best graph maintenance"""
                cur_best_cost = get_cost(graph_buffer.best_graph, self.cost_type)
                if next_graph_cost < cur_best_cost:
                    seq_path = self.output_opt_path(
                        graph_buffer.name, next_graph_cost, exp_seq
                    )
                    msg = f'Agent {self.id} : {graph_buffer.name}: {cur_best_cost} -> {next_graph_cost} ! Seq saved to {seq_path} .'
                    printfl(f'\n{msg}\n')
                    if (
                        self.id == 0 and os.getenv('ALERT_BETTER') == '1'
                    ):  # TODO(not going to do) Colin multi-processing logging
                        wandb.alert(
                            title='Better graph is found!',
                            text=msg,
                            level=wandb.AlertLevel.INFO,
                            wait_duration=0,
                        )  # send alert to slack
                    # end if
                    graph_buffer.best_graph = next_graph
                # end if better
                i_step += 1
                if s_exp.game_over:
                    graph_buffer.eps_lengths.append(i_step)
            # end for s_exp
            if len(obs_res) > 0 and not obs_res[-1].game_over:
                graph_buffer.eps_lengths.append(i_step)
        # end for obs_res

        # end for obs
        """collect experiences together"""
        s_exps: List[SerializableExperience] = list(itertools.chain(*s_exp_lists))

        s_exps_zip = ExperienceList(*map(list, zip(*s_exps)))  # type: ignore
        s_exps_zip.state = state_dgl_list
        s_exps_zip.next_state = next_state_dgl_list
        return s_exps_zip

    def sync_best_graph(self) -> None:
        """broadcast the best graph of each buffer to other agents"""
        sync_dir = os.path.join(self.output_dir, 'sync_dir')
        os.makedirs(sync_dir, exist_ok=True)
        best_info = [
            {
                'name': buffer.name,
                'best_cost': get_cost(buffer.best_graph, self.cost_type),
                'qasm': buffer.best_graph.to_qasm_str(),
            }
            for buffer in self.graph_buffers
        ]
        with open(os.path.join(sync_dir, f'best_info_{self.id}.json'), 'w') as f:
            json.dump(best_info, fp=f, indent=2)
        # printfl(f'Agent {self.id} : waiting for others to sync')
        dist.barrier()
        # printfl(f'Agent {self.id} : finish waiting')
        """read in other agents' results"""
        for r in range(self.num_agents):
            if r != self.id:
                self.load_best_info(os.path.join(sync_dir, f'best_info_{r}.json'))
            # end if r
        # end for r

    def load_best_info(self, best_info_path: str) -> None:
        best_info: List[Dict[str, Any]]
        with open(best_info_path) as f:
            best_info = json.load(f)
        for i in range(len(self.graph_buffers)):
            buffer = self.graph_buffers[i]
            info = best_info[i]
            assert buffer.name == info['name'], f'{buffer.name = }, {info["name"] = }'
            if info['best_cost'] < get_cost(buffer.best_graph, self.cost_type):
                graph = qtz.qasm_to_graph(info['qasm'])
                buffer.push_back(graph)
                buffer.best_graph = graph
                printfl(
                    f'  Agent {self.id} : read in new best graph ({get_cost(buffer.best_graph, self.cost_type)}) from {best_info_path}'
                )
                if self.output_full_seq:
                    buffer.all_graphs = {
                        buffer.best_graph: AllGraphDictValue(
                            0,
                            get_cost(buffer.best_graph, buffer.cost_type),
                            None,
                            Action(0, 0),
                        ),
                    }
            # end if
        # end for i

    def output_best_graph(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for buffer in self.graph_buffers:
            best_g = buffer.best_graph
            best_cost = get_cost(best_g, self.cost_type)
            best_qasm = best_g.to_qasm_str()
            best_path = os.path.join(output_dir, f'{buffer.name}.qasm')
            with open(best_path, 'w') as f:
                f.write(best_qasm)

    @torch.no_grad()
    def select_action_for_self(
        self,
        cur_graphs: List[quartz.PyGraph],
    ):
        self.ac_net.eval()
        num_eps = len(cur_graphs)
        """compute embeds and use Critic to evaluate each node"""
        dgl_graphs: List[dgl.DGLGraph] = [g.to_dgl_graph() for g in cur_graphs]
        b_state: dgl.DGLGraph = dgl.batch(dgl_graphs).to(self.device)
        num_nodes: torch.LongTensor = (
            b_state.batch_num_nodes()
        )  # (num_graphs, ) assert each elem > 0
        # (batch_num_nodes, embed_dim)
        b_node_embeds: torch.Tensor = self.ac_net.gnn(b_state)
        # (batch_num_nodes, )
        b_node_values: torch.Tensor = self.ac_net.critic(b_node_embeds).squeeze()
        # list with length num_graphs; each member is a tensor of node values in a graph
        node_values_list: List[torch.Tensor] = torch.split(
            b_node_values, num_nodes.tolist()
        )
        """sample node by softmax with temperature for each graph as a batch"""
        # (num_graphs, max_num_nodes)
        b_node_values_pad = nn.utils.rnn.pad_sequence(
            node_values_list,
            batch_first=True,
            padding_value=-math.inf,
        )
        # (num_graphs, )
        temperature: torch.Tensor
        if not self.softmax_temp_en:
            temperature = torch.ones(1).to(self.device)
        else:
            temperature = 1 / (
                torch.log(self.hit_rate * (num_nodes - 1) / (1 - self.hit_rate))
            )
        b_softmax_node_values_pad = F.softmax(
            b_node_values_pad / temperature.unsqueeze(1), dim=-1
        )
        # b_softmax_node_values_pad = torch.isclose(
        #     b_node_values_pad,
        #     torch.max(b_node_values_pad, dim=-1, keepdim=True).values
        # ).float()
        b_sampled_nodes = torch.multinomial(b_softmax_node_values_pad, 1).flatten()
        action_nodes: List[int] = b_sampled_nodes.tolist()
        """collect embeddings of sampled nodes"""
        # (num_graphs, )
        node_offsets = torch.zeros(b_sampled_nodes.shape[0], dtype=torch.long).to(
            self.device
        )
        node_offsets[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
        sampled_node_b_ids = b_sampled_nodes + node_offsets
        # (num_graphs, embed_dim)
        sampled_node_embeds = b_node_embeds[sampled_node_b_ids]
        """use Actor to evaluate xfers for sampled nodes"""
        # (num_graphs, action_dim)
        xfer_logits: torch.Tensor = self.ac_net.actor(sampled_node_embeds)
        """sample action_xfer with mask"""
        av_xfer_masks = torch.zeros_like(
            xfer_logits, dtype=torch.bool
        )  # device is the same with xfer_logits
        av_xfer_masks = cast(torch.BoolTensor, av_xfer_masks)
        for i_eps in range(num_eps):
            graph = cur_graphs[i_eps]
            av_xfers = graph.available_xfers_parallel(
                context=qtz.quartz_context,
                node=graph.get_node_from_id(id=action_nodes[i_eps]),
            )
            av_xfer_masks[i_eps][av_xfers] = True
        # end for
        softmax_xfer_logits = masked_softmax(xfer_logits, av_xfer_masks)
        # softmax_xfer_logits = torch.isclose(
        #     softmax_xfer_logits,
        #     torch.max(softmax_xfer_logits, dim=-1, keepdim=True).values
        # ).float()
        xfer_dists = Categorical(softmax_xfer_logits)
        action_xfers = xfer_dists.sample()
        action_xfer_logps: torch.Tensor = xfer_dists.log_prob(action_xfers)
        action_node_values: List[float] = b_node_values_pad[
            list(range(num_eps)), action_nodes
        ].tolist()
        return (
            dgl_graphs,
            action_nodes,
            action_xfers.tolist(),
            action_node_values,
            action_xfer_logps.tolist(),
            av_xfer_masks.cpu(),
        )

    def output_seq(
        self, name: str, best_cost: int, seq: List[Tuple[quartz.PyGraph, Action, float]]
    ) -> str:
        output_dir = os.path.join(self.output_dir, name, f'{best_cost}_{self.id}')
        os.makedirs(output_dir)
        for i_step, (graph, action, reward) in enumerate(seq):
            # NOTE: action here is what action to apply on this graph to get the next graph
            fname = (
                f'{i_step}_{get_cost(graph, self.cost_type)}_{int(reward)}_'
                f'{action.node}_{action.xfer}.qasm'
            )
            qasm = graph.to_qasm_str()
            with open(os.path.join(output_dir, fname), 'w') as f:
                f.write(qasm)
        return output_dir

    def output_full_opt_seq(
        self,
        all_graphs: Dict[quartz.PyGraph, AllGraphDictValue],
        name: str,
        best_graph: quartz.PyGraph,
        best_cost: int,
    ) -> str:
        output_dir = os.path.join(
            self.output_dir, name, f'fullseq_{best_cost}_{self.id}'
        )
        os.makedirs(output_dir)
        graph = best_graph
        while graph is not None:
            info: AllGraphDictValue = all_graphs[graph]
            # NOTE: action here is how this graph is got from its predecessors
            fname = (
                f'{info.dist}_{info.cost}_{info.action.node}_{info.action.xfer}.qasm'
            )
            qasm = graph.to_qasm_str()
            with open(os.path.join(output_dir, fname), 'w') as f:
                f.write(qasm)
            graph = info.pre_graph
        return output_dir

    @torch.no_grad()
    def collect_data_by_self(
        self,
        num_eps: int,
        agent_batch_size: int,
        max_extra_cost: int,
        nop_stop: bool,
        greedy_sample: bool,
    ) -> ExperienceList:
        eps_lists: List[ExperienceList] = [
            ExperienceList.new_empty() for _ in range(num_eps)
        ]

        """sample init state"""
        buffer_idx_list: List[int] = []
        init_graph_list: List[quartz.PyGraph] = []
        original_graph_list: List[quartz.PyGraph] = []
        max_eps_len_for_all: int = 0
        for i_eps in range(num_eps):
            graph_buffer = self.graph_buffers[self.init_buffer_turn]
            init_graph: quartz.PyGraph = graph_buffer.sample(greedy_sample)
            buffer_idx_list.append(self.init_buffer_turn)
            self.init_buffer_turn = (self.init_buffer_turn + 1) % len(
                self.graph_buffers
            )
            init_graph_list.append(init_graph)
            original_graph_list.append(graph_buffer.original_graph)
            """compute max eps len"""
            if self.dyn_eps_len:
                max_len = graph_buffer.max_eps_length
                if max_len < 40:
                    max_eps_len = math.ceil(max_len * 1.5)
                else:
                    max_eps_len = math.ceil(max_len * 1.2)
                max_eps_len = max(max_eps_len, self.min_eps_len)
                max_eps_len = min(max_eps_len, self.max_eps_len)
            else:
                max_eps_len = self.max_eps_len
            max_eps_len_for_all = max(max_eps_len_for_all, max_eps_len)
        # end for
        """communicate with other ranks to get max of max_eps_len_for_all"""
        max_eps_len_all_ranks = torch.zeros(self.num_agents).to(self.device)
        max_eps_len_all_ranks[self.id] = max_eps_len_for_all
        for r in range(self.num_agents):
            dist.broadcast(max_eps_len_all_ranks[r], r)
        max_eps_len_for_all = int(max_eps_len_all_ranks.max())

        """run episodes"""
        original_costs: List[int] = [
            get_cost(orig_g, self.cost_type) for orig_g in original_graph_list
        ]
        cur_graphs: List[quartz.PyGraph] = init_graph_list.copy()  # shallow copy
        graph_seqs: List[List[quartz.PyGraph]] = [[] for _ in range(num_eps)]
        last_eps_ends: List[int] = [-1 for _ in range(num_eps)]
        for i_step in range(max_eps_len_for_all):
            """inference by mini-batches"""
            dgl_graphs: List[dgl.DGLGraph] = []
            action_nodes: List[int] = []
            action_xfers: List[int] = []
            action_node_values: List[float] = []
            action_xfer_logps: List[float] = []
            av_xfer_masks: torch.BoolTensor = cast(
                torch.BoolTensor, torch.zeros(0, dtype=torch.bool)
            )
            for inf_batch_start in range(0, num_eps, agent_batch_size):
                (
                    mb_dgl_graphs,
                    mb_action_nodes,
                    mb_action_xfers,
                    mb_node_values,
                    mb_logps,
                    mb_av_xfer_masks,
                ) = self.select_action_for_self(
                    cur_graphs[inf_batch_start : inf_batch_start + agent_batch_size]
                )
                dgl_graphs += mb_dgl_graphs
                action_nodes += mb_action_nodes
                action_xfers += mb_action_xfers
                action_node_values += mb_node_values
                action_xfer_logps += mb_logps
                av_xfer_masks = cast(
                    torch.BoolTensor, torch.cat([av_xfer_masks, mb_av_xfer_masks])
                )
            # end for infer_batch
            """apply actions"""
            for i_eps in range(num_eps):
                eps_list = eps_lists[i_eps]
                last_eps_end = last_eps_ends[i_eps]
                graph_seq = graph_seqs[i_eps]
                graph = cur_graphs[i_eps]
                graph_buffer = self.graph_buffers[buffer_idx_list[i_eps]]
                cur_best_cost = get_cost(graph_buffer.best_graph, self.cost_type)

                graph_seq.append(graph)
                action = Action(action_nodes[i_eps], action_xfers[i_eps])
                next_graph, next_nodes = graph.apply_xfer_with_local_state_tracking(
                    node=graph.get_node_from_id(id=action.node),
                    xfer=qtz.quartz_context.get_xfer_from_id(id=action.xfer),
                    eliminate_rotation=qtz.has_parameterized_gate,
                    predecessor_layers=self.xfer_pred_layers,
                )
                """parse result, compute reward"""
                reward: float = 0.0
                game_over: bool = False
                if next_graph is None:
                    reward = self.invalid_reward
                    game_over = True
                    next_graph = graph
                elif qtz.is_nop(action.xfer):
                    reward = 0.0
                    game_over = nop_stop
                    next_nodes = [action.node]
                else:
                    graph_cost = get_cost(graph, self.cost_type)
                    next_graph_cost = get_cost(next_graph, self.cost_type)
                    reward = graph_cost - next_graph_cost
                    # game_over = graph_cost > original_costs[i_eps] * max_extra_cost
                    game_over = graph_cost > cur_best_cost + max_extra_cost
                    if self.limit_total_gate_count:
                        game_over |= (
                            graph.gate_count
                            > original_graph_list[i_eps].gate_count * max_extra_cost
                        )
                        # NOTE: limit the gate count according to the original graph
                        # (the one inputted by file) rather than the starting graph
                if i_step - last_eps_end >= max_eps_len_for_all:
                    game_over = True  # exceed len limit

                """collect data"""

                # collect next_graph info
                if i_step - last_eps_end > 1:
                    # not the first step of a trajactory; cur_graph is next_graph of last step
                    if self.subgraph_opt:
                        last_next_nodes = eps_list.next_nodes[-1]
                        sub_graph, new_indices = self.khop_subgraph(
                            dgl_graphs[i_eps], last_next_nodes
                        )
                        eps_list.next_nodes[-1] = new_indices
                        eps_list.next_state.append(sub_graph)
                    else:
                        eps_list.next_state.append(dgl_graphs[i_eps])

                if game_over or i_step == max_eps_len_for_all - 1:
                    # the last step of this trajectory; add next_graph info for itself
                    next_dgl_graph = next_graph.to_dgl_graph()
                    if self.subgraph_opt:
                        next_dgl_graph, next_nodes = self.khop_subgraph(
                            next_dgl_graph, next_nodes
                        )
                    eps_list.next_state.append(next_dgl_graph)
                # add next_nodes in all steps
                eps_list.next_nodes.append(next_nodes)

                # collect cur_graph info
                if self.subgraph_opt:
                    dgl_graphs[i_eps], new_indices = dgl.khop_out_subgraph(
                        dgl_graphs[i_eps],
                        action.node,
                        k=self.ac_net.gnn_num_layers,
                    )
                    action.node = new_indices[0]
                eps_list.state.append(dgl_graphs[i_eps])
                eps_list.action.append(action)

                # collect other info
                eps_list.reward.append(reward)
                eps_list.game_over.append(game_over)
                eps_list.node_value.append(action_node_values[i_eps])
                eps_list.xfer_mask.append(cast(torch.BoolTensor, av_xfer_masks[i_eps]))
                eps_list.xfer_logprob.append(action_xfer_logps[i_eps])
                eps_list.info.append({})

                """collect info for graph buffer"""

                if i_step == last_eps_end + 1:
                    # the first step in an episode
                    graph_buffer.rewards.append([])
                    graph_buffer.append_init_costs_from_graph(graph)
                    graph_buffer.append_costs_from_graph(graph)

                graph_buffer.rewards[-1].append(reward)
                graph_buffer.append_costs_from_graph(next_graph)
                if next_graph is not graph and self.output_full_seq:
                    graph_buffer.push_back_all_graphs(
                        next_graph, next_graph_cost, graph, action
                    )
                if (
                    not game_over
                    and not qtz.is_nop(action.xfer)
                    and next_graph_cost <= get_cost(graph_seq[0], self.cost_type)
                ):
                    if not greedy_sample or next_graph_cost < cur_best_cost + 6:
                        graph_buffer.push_back(next_graph)
                    """best graph maintenance"""
                    if next_graph_cost < cur_best_cost:
                        # create a list of tuple: (graph, action taken on this graph, reward of action)
                        seq: List[Tuple[quartz.PyGraph, Action, float]] = []
                        for i_glob_step in range(last_eps_end + 1, i_step + 1):
                            i_eps_step = i_glob_step - (last_eps_end + 1)
                            seq.append(
                                (
                                    graph_seq[i_eps_step],
                                    eps_list.action[i_glob_step],
                                    eps_list.reward[i_glob_step],
                                )
                            )
                        seq.append((next_graph, Action(0, 0), 0))
                        # output the seq and log info
                        seq_path = self.output_seq(
                            graph_buffer.name, next_graph_cost, seq
                        )
                        msg = f'Agent {self.id} : {graph_buffer.name}: {cur_best_cost} -> {next_graph_cost} ! Seq saved to {seq_path} .'
                        printfl(f'\n{msg}\n')
                        if (
                            self.id == 0
                        ):  # TODO(not going to do) multi-processing logging
                            wandb.alert(
                                title='Better graph is found!',
                                text=msg,
                                level=wandb.AlertLevel.INFO,
                                wait_duration=0,
                            )  # send alert to slack
                        graph_buffer.best_graph = next_graph
                        if self.output_full_seq:
                            printfl(f'Agent {self.id}: saving full seq...')
                            full_seq_path = self.output_full_opt_seq(
                                graph_buffer.all_graphs,
                                graph_buffer.name,
                                next_graph,
                                next_graph_cost,
                            )
                            printfl(
                                f'Agent {self.id}: full seq saved to {full_seq_path}'
                            )
                    # end if better
                # end if

                if i_step == max_eps_len_for_all - 1:
                    # the last step; the loop goes to end
                    if not game_over:
                        graph_buffer.eps_lengths.append(i_step - last_eps_end)

                if game_over:
                    # batch inference, restart
                    init_graph = graph_buffer.sample(greedy_sample)
                    cur_graphs[i_eps] = init_graph
                    graph_buffer.eps_lengths.append(i_step - last_eps_end)
                    graph_seq.clear()
                    last_eps_ends[i_eps] = i_step
                else:
                    cur_graphs[i_eps] = next_graph
            # end for i_eps
        # end for i_step in an episode
        eps_list_cat = ExperienceList.new_empty()
        for eps_list in eps_lists:
            eps_list_cat += eps_list
        eps_list_cat.sanity_check()
        return eps_list_cat

    def khop_subgraph(self, g: dgl.DGLGraph, nodes: List[int]) -> dgl.DGLGraph:
        subgraph, new_indices = dgl.khop_out_subgraph(
            g, nodes, k=self.ac_net.gnn_num_layers
        )
        return subgraph, cast(torch.Tensor, new_indices).tolist()
