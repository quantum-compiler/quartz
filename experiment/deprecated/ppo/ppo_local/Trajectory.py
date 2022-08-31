import torch
from PPO import PPO, RolloutBuffer

from quartz import PyGraph, QuartzContext


class Trajectory:
    def __init__(
        self,
        tid: int,
        init_circ_name: str,
        init_circ: PyGraph,
        init_circ_original_cnt: int,
        init_circ_best_possible_cnt: int,
        max_seq_len: int,
        invalid_reward: int,
    ) -> None:
        self.tid = tid
        self.init_circ_name = init_circ_name
        self.init_circ = init_circ
        self.init_gate_cnt = init_circ_original_cnt
        self.best_possible_cnt = init_circ_best_possible_cnt
        self.max_seq_len = max_seq_len
        self.invalid_reward = invalid_reward
        self.rollout_buffer = RolloutBuffer()

        self.current_circ = init_circ

        self.done = False
        self.is_nop = False
        self.in_local = False

        self.node_range = []

        self.t_reward = 0
        self.t_best_gate_cnt = self.init_gate_cnt
        self.t_gate_cnt = self.init_gate_cnt
        self.t_len = 0

        self.intermediate_circ_buffer = {}
        self.intermediate_circ_buffer['circ'] = []
        self.intermediate_circ_buffer['hash'] = []

    def reset_nop(self):
        self.is_nop = False

    def not_done(self):
        return not self.done

    def apply_action_and_record(
        self,
        context: QuartzContext,
        node: int,
        xfer: int,
        xfer_logprob: torch.Tensor,
        mask: torch.Tensor,
    ):
        next_circ, next_nodes = self.current_circ.apply_xfer_with_local_state_tracking(
            xfer=context.get_xfer_from_id(id=xfer),
            node=self.current_circ.get_node_from_id(id=node),
        )

        # Update states
        # Update t_len
        self.t_len += 1

        # Update is_nop
        if context.get_xfer_from_id(id=xfer).is_nop:
            self.is_nop = True
        else:
            self.is_nop = False

        # Update done
        # Invalid xfers
        if next_circ == None:
            self.done = True
        # Upper limit for sequence length is reached
        elif self.t_len == self.max_seq_len:
            self.done = True
        # NOP when not in local
        elif not self.in_local and self.is_nop:
            self.done = True
        # Gate count too large
        elif next_circ.gate_count > self.init_gate_cnt * 1.1:
            self.done = True

        # Update in_local
        if self.is_nop:
            self.in_local = False
        elif next_nodes == []:
            self.in_local = False
        else:
            self.in_local = True

        # Update node_range
        if self.in_local:
            node_range = torch.tensor(next_nodes, dtype=torch.int64)
            if context.get_xfer_from_id(id=xfer).dst_gate_count != 0:
                src_node_ids, _, edge_ids = next_circ.to_dgl_graph().in_edges(
                    node_range, form='all'
                )
                edge_mask = next_circ.to_dgl_graph().edata['reversed'][edge_ids] == 0
                node_range = torch.cat((node_range, src_node_ids[edge_mask]))

            # TODO: 2 hop predecessors?
            src_node_ids, _, edge_ids = next_circ.to_dgl_graph().in_edges(
                node_range, form='all'
            )
            edge_mask = next_circ.to_dgl_graph().edata['reversed'][edge_ids] == 0
            node_range = torch.cat((node_range, src_node_ids[edge_mask]))

        else:
            self.node_range = []

        # Reward
        reward: int = 0
        if self.current_circ == None:
            reward = self.invalid_reward
        elif self.is_nop:
            reward = 0
        else:
            reward = (self.current_circ.gate_count - next_circ.gate_count) * 4
        self.t_reward += reward

        # Update rollout_buffer
        self.rollout_buffer.graphs.append(self.current_circ)
        self.rollout_buffer.nodes.append(node)
        self.rollout_buffer.xfers.append(xfer)
        self.rollout_buffer.xfer_logprobs.append(xfer_logprob)  # torch.Tensor
        self.rollout_buffer.masks.append(mask)  # torch.Tensor
        self.rollout_buffer.next_graphs.append(next_circ)
        self.rollout_buffer.next_nodes.append(next_nodes)
        self.rollout_buffer.rewards.append(reward)
        self.rollout_buffer.is_nops.append(self.is_nop)
        self.rollout_buffer.is_terminals.append(self.done)
        self.rollout_buffer.is_start_point.append(self.t_len == 0)

        # Update intermediate graphs buffer
        if self.t_reward > 0:
            self.intermediate_circ_buffer['circ'].append(next_circ)
            self.intermediate_circ_buffer['hash'].append(next_circ.hash())

        # Update trajectory best gate count
        self.t_best_gate_cnt = min(self.t_best_gate_cnt, next_circ.gate_count)

        # Update trajectory gate count
        if self.done:
            if next_circ != None:
                self.t_gate_cnt = next_circ.gate_count
            else:
                self.t_gate_cnt = self.current_circ.gate_count

        # Update current_circ
        self.current_circ = next_circ


def sample_init_circs(
    circ_info: dict,
    circ_dataset: dict,
    circ_names: list[str],
    num_each_circ: int,
    keep_origin: bool = True,
) -> list[tuple[str, PyGraph, int, int]]:

    pass


def get_trajectory_batch(
    ppo_agent: PPO,
    context: QuartzContext,
    sampled_init_circs: list[tuple[str, PyGraph, int, int]],
    max_seq_len: int,
    invalid_reward: int,
):
    trajectory_list: list[Trajectory] = []
    for i, circ_info in enumerate(sampled_init_circs):
        trajectory_list.append(Trajectory(i, *circ_info, max_seq_len, invalid_reward))

    for _ in range(max_seq_len):
        undone_ts = {}
        undone_ts['id'] = []
        undone_ts['curr_circ'] = []
        undone_ts['node_range'] = []

        # Collect circuits and node_range from each trajectories
        for trajectory in trajectory_list:
            if trajectory.not_done():
                # Reset nop in the beginning of every timestep
                trajectory.reset_nop()

                undone_ts['id'].append(trajectory.tid)
                undone_ts['curr_circ'].append(trajectory.current_circ)
                undone_ts['node_range'].append(trajectory.node_range)

        if len(undone_ts['id']) == 0:
            break
        else:
            nodes, xfers, xfer_logprobs, masks = ppo_agent.select_actions(
                undone_ts['curr_circ'], undone_ts['node_range']
            )

            for i, id in enumerate(undone_ts['id']):
                trajectory_list[id].apply_action_and_record(
                    context, nodes[i], xfers[i], xfer_logprobs[i], masks[i].clone()
                )

                # print(masks[i].shape)

    intermediate_circs = {}
    trajectory_infos = {}

    for trajectory in trajectory_list:
        # PPO rollout buffer
        ppo_agent.buffer += trajectory.rollout_buffer

        # Aggregate trajectory infos
        if trajectory.init_circ_name not in trajectory_infos:
            trajectory_infos[trajectory.init_circ_name] = {}
            trajectory_infos[trajectory.init_circ_name]['num'] = 1
            trajectory_infos[trajectory.init_circ_name]['seq_len'] = trajectory.t_len
            trajectory_infos[trajectory.init_circ_name]['reward'] = trajectory.t_reward
            trajectory_infos[trajectory.init_circ_name]['possible_reward'] = (
                trajectory.init_gate_cnt - trajectory.best_possible_cnt
            ) * 4
            trajectory_infos[trajectory.init_circ_name][
                'best_reward'
            ] = trajectory.t_reward
            trajectory_infos[trajectory.init_circ_name][
                'best_gate_cnt'
            ] = trajectory.t_best_gate_cnt
            trajectory_infos[trajectory.init_circ_name][
                'best_final_gate_cnt'
            ] = trajectory.t_gate_cnt
        else:
            trajectory_infos[trajectory.init_circ_name]['num'] += 1
            trajectory_infos[trajectory.init_circ_name]['seq_len'] += trajectory.t_len
            trajectory_infos[trajectory.init_circ_name]['reward'] += trajectory.t_reward
            trajectory_infos[trajectory.init_circ_name]['possible_reward'] += (
                trajectory.init_gate_cnt - trajectory.best_possible_cnt
            ) * 4
            trajectory_infos[trajectory.init_circ_name]['best_reward'] = max(
                trajectory.t_reward,
                trajectory_infos[trajectory.init_circ_name]['best_reward'],
            )
            trajectory_infos[trajectory.init_circ_name]['best_gate_cnt'] = min(
                trajectory.t_best_gate_cnt,
                trajectory_infos[trajectory.init_circ_name]['best_gate_cnt'],
            )
            trajectory_infos[trajectory.init_circ_name]['best_final_gate_cnt'] = min(
                trajectory.t_gate_cnt,
                trajectory_infos[trajectory.init_circ_name]['best_final_gate_cnt'],
            )

        # Gather intermediate circuits
        if trajectory.init_circ_name not in intermediate_circs:
            intermediate_circs[trajectory.init_circ_name] = {}
            intermediate_circs[trajectory.init_circ_name][
                'circ'
            ] = trajectory.intermediate_circ_buffer['circ']
            intermediate_circs[trajectory.init_circ_name][
                'hash'
            ] = trajectory.intermediate_circ_buffer['hash']
        else:
            intermediate_circs[trajectory.init_circ_name][
                'circ'
            ] += trajectory.intermediate_circ_buffer['circ']
            intermediate_circs[trajectory.init_circ_name][
                'hash'
            ] += trajectory.intermediate_circ_buffer['hash']

    for circ in trajectory_infos:
        trajectory_infos[circ]['real_rate'] = (
            trajectory_infos[circ]['reward'] / trajectory_infos[circ]['possible_reward']
        )
        trajectory_infos[circ]['avg_seq_len'] = (
            trajectory_infos[circ]['seq_len'] / trajectory_infos[circ]['num']
        )
        trajectory_infos[circ]['avg_reward'] = (
            trajectory_infos[circ]['reward'] / trajectory_infos[circ]['num']
        )

    return intermediate_circs, trajectory_infos
