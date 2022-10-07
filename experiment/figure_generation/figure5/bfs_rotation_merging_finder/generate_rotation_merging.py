import copy
import random


class Game:
    def __init__(self, num_qubits):
        # num qubits
        self.num_qubits = num_qubits
        # state and state history
        self.state = []
        for i in range(num_qubits):
            self.state.append({i})
        self.state_history = [self.state.copy()]
        # action space and action history
        self.action_space = []
        for i in range(num_qubits):
            for j in range(num_qubits):
                if not i == j:
                    self.action_space.append([i, j])
        self.action_history = []

    def apply_action(self, control, target):
        new_game = copy.deepcopy(self)
        new_game.state[target] = self.state[target].symmetric_difference(self.state[control])
        new_game.state_history.append(new_game.state.copy())
        new_game.action_history.append((control, target))
        return new_game

    def check_equivalence(self, min_cx_count):
        # check equivalence
        is_equivalent = False
        t_pos, tdg_pos = (-1, -1), (-1, -1)
        m, i = 0, 0
        for j in range(self.num_qubits):
            if not i == j:
                for n in range(len(self.state_history)):
                    if self.state_history[m][i] == self.state_history[n][j] and n >= min_cx_count:
                        is_equivalent = True
                        t_pos, tdg_pos = (m, i), (n, j)
                        print(f"depth{n}, qubit{j}")

        # variety condition
        variety = len(set(self.action_history))

        # print history
        if is_equivalent and len(self.state_history) >= min_cx_count and variety > self.num_qubits:
            for state in self.state_history:
                print(state)
            while True:
                should_save = input("Save or not (y/n):")
                if should_save == "y":
                    filename = f"{random.randint(0, 10000000)}"
                    self.save_to_file(filename=filename,
                                      t_idx=t_pos, tdg_idx=tdg_pos)
                    break
                elif should_save == "n":
                    break
                else:
                    continue

    def save_to_file(self, filename, t_idx, tdg_idx):
        # decode for t and tdg
        m, i = t_idx[0], t_idx[1]
        n, j = tdg_idx[0], tdg_idx[1]
        step_count = 0
        with open(f"./dataset/{filename}.qasm", 'w') as handle:
            print("OPENQASM 2.0;", file=handle, flush=True)
            print("include \"qelib1.inc\";", file=handle, flush=True)
            print(f"qreg q[{self.num_qubits}];", file=handle, flush=True)
            for action_pair in self.action_history:
                if step_count == m:
                    print(f"t q[{i}];", file=handle, flush=True)
                if step_count == n:
                    print(f"tdg q[{j}];", file=handle, flush=True)
                step_count += 1
                control = action_pair[0]
                target = action_pair[1]
                if not step_count == m and not step_count == n:
                    print(f"t q[{target}];", file=handle, flush=True)
                print(f"cx q[{control}], q[{target}];", file=handle, flush=True)
            if step_count == n:
                print(f"tdg q[{j}];", file=handle, flush=True)


def main():
    # input parameter
    num_qubits = 4
    min_cx_count = 6

    # start search
    initial_game = Game(num_qubits=num_qubits)
    candidate_queue = [initial_game]
    searched_count = 0
    while True:
        cur_game = candidate_queue.pop(0)
        for action_pair in cur_game.action_space:
            control, target = action_pair[0], action_pair[1]
            new_game = cur_game.apply_action(control=control, target=target)
            new_game.check_equivalence(min_cx_count=min_cx_count)
            candidate_queue.append(new_game)
            # logging
            searched_count += 1
            if searched_count % 1000 == 0:
                print(f"Searched {searched_count} circuits.")


if __name__ == '__main__':
    main()
