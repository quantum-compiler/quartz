import matplotlib.pyplot as plt
import os


def main():
    # input arguments
    input_path = "./input"
    use_x_percentage = True

    # read optimization history of each circuit
    circuit_name_list = os.listdir(input_path)
    print(circuit_name_list)
    circuit_opt_history_set = {}
    for circuit_name in circuit_name_list:
        circuit_path = os.path.join(input_path, circuit_name)
        opt_history_list = os.listdir(circuit_path)
        gate_count_list = [-1] * len(opt_history_list)
        for opt_history in opt_history_list:
            cur_step_idx = str.split(opt_history, "_")[0]
            cur_step_gate_count = str.split(opt_history, "_")[1]
            gate_count_list[int(cur_step_idx)] = int(cur_step_gate_count)
        circuit_opt_history_set[circuit_name] = gate_count_list

    # normalize data and plot
    plt.figure(num=1, figsize=(8, 6), dpi=300)
    plt.yticks(ticks=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0],
               labels=["100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%", "0%"])
    if use_x_percentage:
        plt.xticks(ticks=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0],
                   labels=["100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%", "0%"])
    plt.xlabel("optimization process (%)")
    plt.ylabel("remaining gates (%)")
    for circuit_name in circuit_name_list:
        # normalize gate count
        gate_cnt_list = circuit_opt_history_set[circuit_name]
        original_gate_cnt = gate_cnt_list[0]
        for idx in range(len(gate_cnt_list)):
            gate_cnt_list[idx] /= original_gate_cnt
        # normalize steps
        step_list = []
        for idx in range(len(gate_cnt_list)):
            if use_x_percentage:
                step_list.append(idx / (len(gate_cnt_list) - 1))
            else:
                step_list.append(idx)
        # plot
        plt.plot(step_list, gate_cnt_list, label=circuit_name, linewidth=2)
    plt.legend()
    plt.savefig("./figure3_final_figures/figure3.jpg")
    plt.show()


if __name__ == '__main__':
    main()
