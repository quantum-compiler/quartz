2 gpus, 2 obs per agent
trajectory length = 80
global_batch_size: 128

32 * collect_data = 32 * 2 trajectories

PPOMod.iter_finished: 180768, 147973

Agent.inference_finished: 30 ms <~ Observer.action_got
Agent.data_collected: 3200 ms ≈ Observer.trajectory_finished

Agent.graph_converted: 720 ms
Agent.data_batched: 27 ms

Observer.action_got: 37 ms
Observer.action_applied: 2 ms
Observer.trajectory_finished: 3200 ms ≈ length * (action_got + action_applied)

Tow graph convered to dgl in 0.878165 ms.
Tow graph convered to dgl in 1.027023 ms.
Tow qasm graph convered in 2.398593 ms.
Tow qasm graph convered in 2.380831 ms.




