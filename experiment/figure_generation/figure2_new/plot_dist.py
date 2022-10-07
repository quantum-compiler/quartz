import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


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
    y_list = []
    for i in range(max(result_dict)):
        y_list.append(result_dict[i + 1])
    y_list.append(result_dict[-1])
    total = sum(y_list)
    y_list = [y_list[idx] * 100 / total for idx in range(len(y_list))]
    print(total, y_list)

    # prepare x_list
    x_list = list(range(1, max(result_dict) + 2))
    for idx in range(len(x_list)):
        x_list[idx] = f"{x_list[idx]}"
    x_list[-1] = f">{x_list[-2]}"
    print(x_list)

    # return
    return x_list, y_list


# This file plots distribution
def main():
    # get data
    gate_count_x_list, gate_count_y_list = get_data(folder_name="gate_count_raw_results")
    _, circuit_depth_y_list = get_data(folder_name="circuit_depth_raw_results")

    # plot
    labels = gate_count_x_list
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig = plt.figure(num=1, figsize=(8, 6), dpi=300)
    ax = fig.subplots()
    rects1 = ax.bar(x, gate_count_y_list, width * 2, label='gate count')
    # rects1 = ax.bar(x - width / 2, gate_count_y_list, width, label='gate count')
    # rects2 = ax.bar(x + width / 2, circuit_depth_y_list, width, label='circuit depth')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("percentage (%)")
    ax.set_xlabel("min #xfers")
    ax.set_xticks(x, labels)
    ax.set_yticks(ticks=[25, 20, 15, 10, 5, 0],
                  labels=["25%", "20%", "15%", "10%", "5%", "0%"])
    ax.set_ylim((0, 25))
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt="%d%%")
    # ax.bar_label(rects2, padding=3)
    plt.savefig(f"./figure2_new_final_figures/dist.pdf")
    plt.show()


if __name__ == '__main__':
    main()
