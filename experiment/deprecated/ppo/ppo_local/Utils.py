import torch
import torch.nn.functional as F


def masked_softmax(logits, mask):
    mask = torch.ones_like(mask, dtype=torch.bool) ^ mask
    logits[mask] -= 1.0e10
    return F.softmax(logits, dim=-1)


def get_trajectory(ppo_agent, context, init_state, max_seq_len, invalid_reward):
    graph = init_state

    done = False
    nop_stop = False
    is_nop = False
    consective_nop_cnt = 0
    nop_stop_limit = 2

    trajectory_reward = 0
    trajectory_len = 0
    trajectory_best_gate_count = init_state.gate_count
    intermediate_graphs = []

    for t in range(max_seq_len):
        is_nop = False

        if not done:
            # t_0 = time.time()
            node, xfer = ppo_agent.select_action(graph)
            # t_1 = time.time()
            # print(f'time network: {t_1 - t_0}')
            next_graph, next_nodes = graph.apply_xfer_with_local_state_tracking(
                xfer=context.get_xfer_from_id(id=xfer),
                node=graph.get_node_from_id(id=node),
            )
            # t_2 = time.time()
            # print(f'time apply xfer: {t_2 - t_1}')

            if next_graph == None:
                reward = invalid_reward
                done = True
                next_graph = graph  # ???
            elif context.get_xfer_from_id(id=xfer).is_nop:
                # TODO: use this when include increase
                reward = 0
                # nop_stop = True
                is_nop = True
                next_nodes = []
                # consective_nop_cnt += 1
                # if consective_nop_cnt >= nop_stop_limit:
                #     done = True
                done = True
                # continue

                # TODO: use this when no increase
                # reward = 0
                # done = True
                # nop_stop = True
                # next_nodes = [node]
            else:
                reward = (graph.gate_count - next_graph.gate_count) * 4
                # consective_nop_cnt = 0

            trajectory_reward += reward

            if trajectory_reward > 0:
                intermediate_graphs.append(next_graph)

            reward = torch.tensor(reward, dtype=torch.float)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(torch.tensor(done, dtype=torch.bool))
            ppo_agent.buffer.next_graphs.append(next_graph)
            ppo_agent.buffer.next_nodes.append(next_nodes)
            ppo_agent.buffer.is_start_point.append(t == 0)
            ppo_agent.buffer.is_nops.append(is_nop)
            graph = next_graph

            # Upper limit for circuit gate count
            # if graph.gate_count > init_state.gate_count * 1.1:
            #     done = True
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
