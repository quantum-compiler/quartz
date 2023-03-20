# PPO

## Setup

- Follow [INSTALL.md](../../INSTALL.md) to setup a basic environment `quartz` .
- Activate the working environment by `conda activate quartz` .
- Install packages required for PPO by `conda env update --file env_ppo.yml` .

## Implementation Details

We use PyTorch RPC and DDP to accelerate the training process.

The multi-processing model is designed as below.

![Program Design](./program_design.png)

The execuation flow can be summerized as follows.

```pseudocode
main:
    mp.spawn processes
    PPOMod.init_process
        There are 2 kinds of processes.
        ddp process:
            PPOMod.train
                init self.ac_net : ActorCritic
                self.ac_net_old = self.ac_net.clone()
                init self.agent: Agent(self.ac_net_old) (each ddp process has an agent)
                Train loop:
                    PPOMod.train_iter
                        self.agent.collect_data
                            for each observer this agent manages:
                                select a graph_buffer according to self.init_buffer_turn in a Round-Robin way
                                init_graph = graph_buffer.sample()
                                remote async call: observer.run_episode , get a Future object
                            wait until all of the Future objects are ready (means all of the observers finish running an episode)
                                During waiting, Agent.select_action_batch may be called by its observers
                                    if there's a batch of observers waiting for actions:
                                        batch all of the current graphs of observers together
                                        inference by ac_net to compute action nodes and xfer distributions
                                        tell the observers that the results are ready
                            convert graph
                            maintain graph_buffer, best graph
                            batch colleted data
                        train the network
                            for k in k_epochs:
                                for mini-batch in all collected data (by list order)
                                    optimizer.zero_grad()
                                    node_embeddings in each graph = self.ac_net.graph_embedding(mini_batch.state)
                                    action_node_values = self.ac_net.critic(embeddings of mini_batch.action_node)
                                    xfer_dist = softmax( self.ac_net.actor(embeddings of mini_batch.action_node) )

                                    with no gradient environment:
                                        node_embeddings in each next_graph = self.ac_net.graph_embedding(mini_batch.next_state)
                                        next_node_values = self.ac_net.critic( embeddings of mini_batch.next_node )

                                    advantages = mini_batch.reward + gamma * next_node_values - action_node_values
                                    compute actor_loss, critic_loss
                                    compute total loss
                                    optimizer.step()

        observer process:
            sleep if there's no call
            When called by the agent (which manages it) to execute Observer.run_episode:
                for each step in an episode:
                    remote sync call: Agent.select_action_batch to get action's node and xfer's distribution (wo mask)
                    generate xfer mask and sample a xfer to get the complete action
                    apply action
                    save experience
                return experiences
```
