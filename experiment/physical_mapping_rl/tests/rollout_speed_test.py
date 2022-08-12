import random
import time

from quartz import PySimplePhysicalEnv


def main():
    env = PySimplePhysicalEnv(qasm_file_path="./rollout.qasm", backend_type_str="IBM_Q20_TOKYO")
    step_count = 0
    total_reward = 0
    max_reward = -1000
    is_finished = False
    start_time = time.time()
    while not is_finished:
        # get state and action space
        cur_state = env.get_state()
        total_gate_count = cur_state.circuit.number_of_nodes
        action_space = env.get_action_space()

        # apply action
        selected_action_id = random.randint(0, len(action_space) - 1)
        selected_action = action_space[selected_action_id]
        reward = env.step(action=selected_action)

        # check finished
        is_finished = env.is_finished()
        step_count += 1
        total_reward += reward
        max_reward = max(max_reward, reward)

        # log
        if step_count % 1000 == 0:
            used_time = time.time() - start_time
            print(f"Total gate count (before current action) is {total_gate_count}, select action {selected_action_id}")
            print(f"Selected Action: Type = {selected_action.type}, qubit_idx_0 = {selected_action.qubit_idx_0},"
                  f" qubit_idx_1 = {selected_action.qubit_idx_1}, reward = {reward},"
                  f" avg reward = {total_reward / step_count}, max reward = {max_reward}")
            print(f"Time since start = {used_time}, steps = {step_count},"
                  f" rollout speed = {int(step_count / used_time)} steps/s")

        if is_finished:
            used_time = time.time() - start_time
            print(f"Time since start = {used_time}, steps = {step_count},"
                  f" rollout speed = {int(step_count / used_time)} steps/s")
            print(f"total reward = {total_reward}, avg reward = {total_reward / step_count},"
                  f" max reward = {max_reward}")


if __name__ == '__main__':
    main()
