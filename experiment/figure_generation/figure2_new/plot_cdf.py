import os
import pickle

import matplotlib.colors
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

    # prepare label list
    label_list = []
    for y in y_list:
        label_list.append("%.2f" % y)

    # return
    return x_list, y_list, label_list


# This file plot cdf.
def main():
    # get data
    gate_count_x_list, gate_count_y_list, label_list = get_data(folder_name="gate_count_raw_results")

    # figure
    plt.rcParams.update({'font.size': 20})
    plt.figure(num=1, figsize=(8, 6), dpi=300)
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.13)

    # axis
    plt.xlabel("Minimal Number of Transformations to Reduce Cost", fontsize=20, labelpad=10.0)
    plt.ylabel("CDF", fontsize=20, labelpad=10.0)
    plt.ylim((-0.03, 1.1))
    plt.xlim((-0.3, 7.3))
    plt.yticks(ticks=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    # curve
    plt.plot(gate_count_x_list, gate_count_y_list,
             marker='o', markersize=5, mew=7,
             label="gate count",
             linewidth=5, color=matplotlib.colors.CSS4_COLORS.get("slateblue"))
    plt.legend()

    # curve label
    plt.rcParams.update({'font.size': 15})
    for x, y, label in zip(gate_count_x_list, gate_count_y_list, label_list):
        plt.text(x, y + 0.02, label, ha="center", va="bottom")

    plt.savefig(f"./figure2_new_final_figures/cdf.pdf")
    plt.show()


if __name__ == '__main__':
    main()
