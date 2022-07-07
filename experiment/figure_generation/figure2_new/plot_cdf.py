import os
import pickle
import argparse

import matplotlib.pyplot as plt


def get_data(folder_name):
    # initialize
    result_dict = {}
    for i in range(6):
        if i + 1 not in result_dict:
            result_dict[i + 1] = 0
    if -1 not in result_dict:
        result_dict[-1] = 0

    # load data:
    filename_list = os.listdir(f"./{folder_name}")
    for filename in filename_list:
        with open(f"./{folder_name}/{filename}", 'rb') as handle:
            tmp_result_dict = pickle.load(file=handle)
        print(tmp_result_dict)
        for key in tmp_result_dict:
            result_dict[key] += tmp_result_dict[key]

    # prepare y_list
    y_list = [0]
    cumulative_value = 0
    for i in range(max(result_dict)):
        cumulative_value += result_dict[i + 1]
        y_list.append(cumulative_value)
    y_list.append(cumulative_value + result_dict[-1])
    y_list = [y_list[idx] / y_list[-1] for idx in range(len(y_list))]
    print(y_list)

    # prepare x_list
    x_list = list(range(max(result_dict) + 2))
    x_list[-1] = f">{x_list[-2]}"
    print(x_list)

    # return
    return x_list, y_list


# This file plot cdf.
def main():
    # get data
    gate_count_x_list, gate_count_y_list = get_data(folder_name="gate_count_raw_results")

    # plot
    plt.figure(num=1, figsize=(8, 6), dpi=300)
    plt.xlabel("min #xfers")
    plt.ylabel("percentage (%)")
    plt.ylim((-0.02, 1.02))
    plt.xlim((-0.1, 7.1))
    plt.yticks(ticks=[1, 0.8, 0.6, 0.4, 0.2, 0],
               labels=["100%", "80%", "60%", "40%", "20%", "0%"])
    plt.plot(gate_count_x_list, gate_count_y_list, marker='o', label="gate count", linewidth=2)
    plt.legend()
    # plt.savefig(f"./figure2_new_final_figures/cdf.pdf")
    plt.show()


if __name__ == '__main__':
    main()
