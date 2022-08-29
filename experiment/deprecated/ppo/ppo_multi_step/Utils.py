import torch
import torch.nn.functional as F


def masked_softmax(logits, mask):
    mask = torch.ones_like(mask, dtype=torch.bool) ^ mask
    logits[mask] -= 1.0e10
    return F.softmax(logits, dim=-1)


def get_trajectory(ppo_agent, context, init_state, max_seq_len, invalid_reward):
    graph = init_state

    done = False
    is_nop = False
    in_local = False

    trajectory_reward = 0
    trajectory_len = 0
    trajectory_best_gate_count = init_state.gate_count
    intermediate_graphs = []

    node_range = []

    for t in range(max_seq_len):
        is_nop = False

        if node_range == []:
            in_local = False

        if not done:
            node, xfer = ppo_agent.select_action(graph, node_range)

            next_graph, next_nodes = graph.apply_xfer_with_local_state_tracking(
                xfer=context.get_xfer_from_id(id=xfer),
                node=graph.get_node_from_id(id=node),
            )

            # Invalid xfer
            if next_graph == None:
                reward = invalid_reward
                done = True
                next_graph = graph
            # Nop
            elif context.get_xfer_from_id(id=xfer).is_nop:
                reward = 0
                is_nop = True
                next_nodes = []
                node_range = []

                # If a node is chosen and it chooses NOP immediately
                # Stop the trajectory
                if not in_local:
                    done = True
                in_local = False
            else:
                reward = (graph.gate_count - next_graph.gate_count) * 4
                in_local = True

                # Get new node_range
                node_range = torch.tensor(next_nodes, dtype=torch.int64)
                if context.get_xfer_from_id(id=xfer).dst_gate_count != 0:
                    src_node_ids, _, edge_ids = next_graph.to_dgl_graph().in_edges(
                        node_range, form='all'
                    )
                    mask = next_graph.to_dgl_graph().edata['reversed'][edge_ids] == 0
                    node_range = torch.cat((node_range, src_node_ids[mask]))

            trajectory_reward += reward

            if trajectory_reward > 0:
                intermediate_graphs.append(next_graph)
            # Add a mark of terminal if the limit is reached
            if t == max_seq_len:
                done = True
            # Upper limit for circuit gate count
            if graph.gate_count > init_state.gate_count * 1.1:
                done = True

            reward = torch.tensor(reward, dtype=torch.float)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(
                torch.tensor(t == max_seq_len - 1, dtype=torch.bool)
            )
            ppo_agent.buffer.next_graphs.append(next_graph)
            ppo_agent.buffer.next_nodes.append(node_range)
            ppo_agent.buffer.is_start_point.append(t == 0)
            ppo_agent.buffer.is_nops.append(is_nop)
            graph = next_graph

        else:
            trajectory_len = t
            break

    trajectory_best_gate_count = min(graph.gate_count, trajectory_best_gate_count)

    if trajectory_len == 0:
        trajectory_len = max_seq_len

    return (
        trajectory_reward,
        trajectory_best_gate_count,
        trajectory_len,
        intermediate_graphs,
    )
