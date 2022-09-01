import time

import dgl
import torch
import torch.nn as nn
import wandb
from ActorCritic import ActorCritic

from quartz import PyGraph


class RolloutBuffer:
    def __init__(self):
        self.graphs = []
        self.nodes = []
        self.xfers = []
        self.next_graphs = []
        self.next_nodes = []
        self.xfer_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.is_start_point = []
        self.is_nops = []
        self.masks = []
        self.values = []
        self.next_values = []

    def clear(self):
        self.__init__()

    def __iadd__(self, other):
        self.graphs += other.graphs
        self.nodes += other.nodes
        self.xfers += other.xfers
        self.next_graphs += other.next_graphs
        self.next_nodes += other.next_nodes
        self.xfer_logprobs += other.xfer_logprobs
        self.rewards += other.rewards
        self.is_terminals += other.is_terminals
        self.is_start_point += other.is_start_point
        self.is_nops += other.is_nops
        self.masks += other.masks
        self.values += other.values
        self.next_values += other.next_values
        return self


class PPO:
    def __init__(
        self,
        num_gate_type,
        context,
        gnn_layers,
        graph_embed_size,
        actor_hidden_size,
        critic_hidden_size,
        action_dim,
        lr_graph_embedding,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        entropy_coefficient,
        mini_batch_size,
        log_file_handle,
        device_get_trajectory,
        device_update,
    ):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_cofficient = entropy_coefficient
        self.mini_batch_size = mini_batch_size
        self.device_get_trajectory = device_get_trajectory
        self.device_update = device_update

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            gnn_layers,
            num_gate_type,
            graph_embed_size,
            actor_hidden_size,
            critic_hidden_size,
            action_dim,
            self.device_update,
        ).to(self.device_update)
        self.optimizer = torch.optim.Adam(
            [
                {
                    'params': self.policy.graph_embedding.parameters(),
                    'lr': lr_graph_embedding,
                },
                {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            ]
        )
        wandb.watch(self.policy, log='all', log_freq=10)

        self.policy_old = ActorCritic(
            gnn_layers,
            num_gate_type,
            graph_embed_size,
            actor_hidden_size,
            critic_hidden_size,
            action_dim,
            self.device_get_trajectory,
        ).to(self.device_get_trajectory)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.context = context

        self.log_file_handle = log_file_handle

    def select_action(self, graph, node_range):
        # Use the old policy network to select an action
        # No gradient needed
        with torch.no_grad():
            node, xfer, xfer_logprob, mask = self.policy_old.act(
                self.context, graph, node_range
            )

        self.buffer.graphs.append(graph)
        self.buffer.nodes.append(node)
        self.buffer.xfers.append(xfer)
        self.buffer.xfer_logprobs.append(xfer_logprob)
        self.buffer.masks.append(mask)

        return node.item(), xfer.item()

    def select_actions(self, graphs: list[PyGraph]):
        with torch.no_grad():
            nodes, xfers, xfer_logprobs, masks = self.policy_old.act_batch(
                self.context, graphs
            )

        return nodes, xfers, xfer_logprobs, masks

    def select_actions_beam(self, undone_ts: dict, k: int):
        graphs = []
        for gs in undone_ts['curr_circs']:
            graphs += gs
        with torch.no_grad():
            graphs, nodes, xfers, xfer_logprobs, masks = self.policy_old.act_batch_beam(
                self.context, graphs, k
            )

        return_dict = {}
        begin, end = 0, 0
        for id, num in zip(undone_ts['id'], undone_ts['curr_circ_num']):
            end = begin + num * k
            return_dict[id] = (
                graphs[begin:end],
                nodes[begin:end],
                xfers[begin:end],
                xfer_logprobs[begin:end],
                masks[begin:end],
            )
            begin = end
        return return_dict

    def update(self):
        # start = time.time()

        masks = torch.stack(self.buffer.masks)

        gs = [g.to_dgl_graph() for g in self.buffer.graphs]
        batched_dgl_gs = dgl.batch(gs).to(self.device_update)

        dgl_next_gs = [g.to_dgl_graph() for g in self.buffer.next_graphs]
        batched_dgl_next_gs = dgl.batch(dgl_next_gs).to(self.device_update)

        node_nums = batched_dgl_gs.batch_num_nodes().tolist()
        next_node_nums = batched_dgl_next_gs.batch_num_nodes().tolist()

        next_node_lists = self.buffer.next_nodes

        old_xfer_logprobs = (
            torch.squeeze(torch.stack(self.buffer.xfer_logprobs, dim=0))
            .detach()
            .to(self.device_update)
        )

        for _ in range(self.K_epochs):
            values, next_values, xfer_logprobs, xfer_entropys = self.policy.evaluate(
                batched_dgl_gs,
                self.buffer.nodes,
                self.buffer.xfers,
                batched_dgl_next_gs,
                next_node_lists,
                self.buffer.is_nops,
                self.buffer.is_terminals,
                masks,
                node_nums,
                next_node_nums,
            )

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(xfer_logprobs - old_xfer_logprobs.detach())
            # ratios_nop_masked = ratios[nop_mask]

            # Finding Surrogate Loss
            rewards = torch.stack(self.buffer.rewards).to(self.device_update)
            advantages = rewards + next_values * self.gamma - values
            surr1 = ratios * advantages.clone().detach()
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages.clone().detach()
            )

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = advantages.pow(2).mean()
            xfer_entropy = xfer_entropys.mean()

            wandb.log(
                {
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
                    'xfer_entropy': xfer_entropy,
                }
            )

            # final loss of clipped objective PPO
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
            #     state_values, rewards) - 0.01 * (node_entropy + xfer_entropy)
            loss = (
                actor_loss + 0.5 * critic_loss - self.entropy_cofficient * xfer_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        cumulated_reward = 0
        for i in range(len(self.buffer.graphs)):
            if self.buffer.is_start_point[i]:
                cumulated_reward = 0
                self.log_file_handle.write(
                    f'initial gate count: {self.buffer.graphs[i].gate_count}\n'
                )
            message = f"node: {self.buffer.nodes[i]}\txfer: {self.buffer.xfers[i]}\treward: {self.buffer.rewards[i]}\tvalue: {values[i]:.3f}\tnext values: {next_values[i]:.3f}"
            if self.buffer.rewards[i] > 0:
                cumulated_reward += self.buffer.rewards[i]
                message += "\tReduced!!!"
                message += f'\n{masks[i].nonzero()}'
                message += f'\n{torch.exp(old_xfer_logprobs[i])}'
                message += f'\n{torch.exp(xfer_logprobs[i])}'
            elif self.buffer.rewards[i] < 0:
                cumulated_reward += self.buffer.rewards[i]
                message += "\tIncreased..."
                message += f'\n{masks[i].nonzero()}'
                message += f'\n{torch.exp(old_xfer_logprobs[i])}'
                message += f'\n{torch.exp(xfer_logprobs[i])}'
            elif self.buffer.is_nops[i]:
                message += "\tNOP."
                message += f'\n{masks[i].nonzero()}'
                message += f'\n{torch.exp(old_xfer_logprobs[i])}'
                message += f'\n{torch.exp(xfer_logprobs[i])}'

            # print(message)
            self.log_file_handle.write(message + '\n')
            self.log_file_handle.flush()
            if self.buffer.is_terminals[i]:
                self.log_file_handle.write('terminated\n')
                self.log_file_handle.write(f'{masks[i].nonzero()}\n')
                self.log_file_handle.write(f'trajectory reward: {cumulated_reward}\n')

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        # print(f'evaluation time: {time.time() - t_0}')
        # print(f"update time: {time.time() - start}")

    def update_mini_batch(self):
        def evaluate_generator(buffer: RolloutBuffer):
            dp_num = len(buffer.graphs)
            mbs = self.mini_batch_size
            for i in range(0, dp_num, mbs):
                gs = buffer.graphs[i : i + mbs]
                dgl_gs = [g.to_dgl_graph() for g in gs]
                batched_dgl_gs = dgl.batch(dgl_gs).to(self.device_update)
                node_nums = batched_dgl_gs.batch_num_nodes().tolist()

                next_gs = buffer.next_graphs[i : i + mbs]
                dgl_next_gs = [g.to_dgl_graph() for g in next_gs]
                batched_dgl_next_gs = dgl.batch(dgl_next_gs).to(self.device_update)
                next_node_nums = batched_dgl_next_gs.batch_num_nodes().tolist()

                yield batched_dgl_gs, buffer.nodes[i : i + mbs], buffer.xfers[
                    i : i + mbs
                ], batched_dgl_next_gs, buffer.next_nodes[i : i + mbs], buffer.is_nops[
                    i : i + mbs
                ], buffer.is_terminals[
                    i : i + mbs
                ], torch.stack(
                    buffer.masks[i : i + mbs]
                ).to(
                    self.device_update
                ), node_nums, next_node_nums

        def log_prob_generator(buffer: RolloutBuffer):
            dp_num = len(buffer.graphs)
            mbs = self.mini_batch_size
            for i in range(0, dp_num, mbs):
                yield torch.squeeze(
                    torch.stack(buffer.xfer_logprobs[i : i + mbs], dim=0)
                ).detach().to(self.device_update)

        def reward_generator(buffer: RolloutBuffer):
            dp_num = len(buffer.graphs)
            mbs = self.mini_batch_size
            for i in range(0, dp_num, mbs):
                yield torch.stack(buffer.rewards[i : i + mbs]).to(self.device_update)

        advantage_0 = []
        for _ in range(self.K_epochs):
            dp_num = len(self.buffer.graphs)
            # print(dp_num)

            eva_gen = evaluate_generator(self.buffer)
            log_prob_gen = log_prob_generator(self.buffer)
            reward_gen = reward_generator(self.buffer)

            self.optimizer.zero_grad()

            ###############################################

            for i, (eva_data, old_log_probs, rewards) in enumerate(
                zip(eva_gen, log_prob_gen, reward_gen)
            ):
                # start = time.time()

                (
                    values,
                    next_values,
                    xfer_logprobs,
                    xfer_entropys,
                ) = self.policy.evaluate(*eva_data)

                # t_0 = time.time()
                # print(f'evaluate time: {t_0 - start}')

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(xfer_logprobs - old_log_probs)

                # Finding Surrogate Loss
                advantages = rewards + next_values * self.gamma - values

                if _ == 0:
                    advantage_0.append(advantages)

                # surr1 = ratios * advantages.clone().detach()
                # surr2 = torch.clamp(
                #     ratios, 1 - self.eps_clip,
                #     1 + self.eps_clip) * advantages.clone().detach()

                surr1 = ratios * advantage_0[i].clone().detach()
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantage_0[i].clone().detach()
                )

                actor_loss = -torch.min(surr1, surr2).sum() / dp_num
                critic_loss = advantages.pow(2).sum() / dp_num
                xfer_entropy = xfer_entropys.sum() / dp_num

                wandb.log(
                    {
                        'actor_loss': actor_loss,
                        'critic_loss': critic_loss,
                        'xfer_entropy': xfer_entropy,
                    }
                )

                # final loss of clipped objective PPO
                # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
                #     state_values, rewards) - 0.01 * (node_entropy + xfer_entropy)
                loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    - self.entropy_cofficient * xfer_entropy
                )

                # take gradient step
                loss.backward()
                for param in self.policy.parameters():
                    param.grad.data.clamp_(-1, 1)

                torch.cuda.empty_cache()

                # t_1 = time.time()
                # print(f'other time: {t_1 - t_0}')

            self.optimizer.step()

        cumulated_reward = 0
        for i in range(len(self.buffer.graphs)):
            # if self.buffer.is_start_point[i]:
            #     cumulated_reward = 0
            #     self.log_file_handle.write(
            #         f'initial gate count: {self.buffer.graphs[i].gate_count}\n'
            #     )
            # message = f"node: {self.buffer.nodes[i]}\txfer: {self.buffer.xfers[i]}\treward: {self.buffer.rewards[i]}\tvalue: {values[i]:.3f}\tnext values: {next_values[i]:.3f}"
            message = f"node: {self.buffer.nodes[i]}\txfer: {self.buffer.xfers[i]}\treward: {self.buffer.rewards[i]}\t"
            if self.buffer.rewards[i] > 0:
                cumulated_reward += self.buffer.rewards[i]
                message += "\tReduced!!!"
                # message += f'\n{masks[i].nonzero()}'
                # message += f'\n{torch.exp(old_xfer_logprobs[i])}'
                # message += f'\n{torch.exp(xfer_logprobs[i])}'
            elif self.buffer.rewards[i] < 0:
                cumulated_reward += self.buffer.rewards[i]
                message += "\tIncreased..."
                # message += f'\n{masks[i].nonzero()}'
                # message += f'\n{torch.exp(old_xfer_logprobs[i])}'
                # message += f'\n{torch.exp(xfer_logprobs[i])}'
            elif self.buffer.is_nops[i]:
                message += "\tNOP."
                # message += f'\n{masks[i].nonzero()}'
                # message += f'\n{torch.exp(old_xfer_logprobs[i])}'
                # message += f'\n{torch.exp(xfer_logprobs[i])}'

            # print(message)
            self.log_file_handle.write(message + '\n')
            self.log_file_handle.flush()
            if self.buffer.is_terminals[i]:
                self.log_file_handle.write('terminated\n')
                # self.log_file_handle.write(f'{masks[i].nonzero()}\n')
                self.log_file_handle.write(f'trajectory reward: {cumulated_reward}\n')

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
