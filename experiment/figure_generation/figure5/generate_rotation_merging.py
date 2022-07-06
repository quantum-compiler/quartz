import copy
import pickle


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
        new_game.action_history.append({"control": control, "target": target})
        return new_game

    def check_equivalence(self):
        # check equivalence
        is_equivalent = False
        t_pos, tdg_pos = -1, -1
        for i in range(num_qubits):
            for j in range(num_qubits):
                if not i == j and self.state[i] == self.state[j]:
                    is_equivalent = True
                    t_pos, tdg_pos = i, j

        # print history
        if is_equivalent:
            for state in self.state_history:
                print(state)
            while True:
                should_save = input("Save or not (y/n):")
                if should_save == "y":
                    self.save_to_file(t_idx=t_pos, tdg_idx=tdg_pos)
                elif should_save == "n":
                    break
                else:
                    continue

    def save_to_file(self, t_idx, tdg_idx):
        with open("./input.qasm", 'w') as handle:
            print("OPENQASM 2.0;", file=handle, flush=True)
            print("include \"qelib1.inc\";", file=handle, flush=True)
            print(f"qreg q[{self.num_qubits}];", file=handle, flush=True)
            for action_pair in self.action_history:
                control = action_pair["control"]
                target = action_pair["target"]
                print(f"cx q[{control}], q[{target}];", file=handle, flush=True)
            print(f"t q[{t_idx}];", file=handle, flush=True)
            print(f"tdg q[{tdg_idx}];", file=handle, flush=True)


def main():
    # input parameter
    num_qubits = 5

    # start search
    initial_game = Game(num_qubits=num_qubits)
    candidate_queue = [initial_game]
    while True:
        cur_game = candidate_queue.pop(0)
        for action_pair in cur_game.action_space:
            control, target = action_pair[0], action_pair[1]
            new_game = cur_game.apply_action(control=control, target=target)
            new_game.check_equivalence()
            candidate_queue.append(new_game)


if __name__ == '__main__':
    main()
