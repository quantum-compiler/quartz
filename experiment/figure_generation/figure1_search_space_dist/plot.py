import math
import pickle

import matplotlib.pyplot as plt


# This file plots distribution of search space
def main():
    # data for this figure
    raw_y_list = [18, 176, 1208, 6463, 28513, 107788, 360775, 1106946, 3247176]
    y_list = []
    for item in raw_y_list:
        item = math.log10(item)
        y_list.append(item)
    x_list = list(range(1, len(y_list) + 1))
    print(x_list)
    print(y_list)

    # basic figure setting
    plt.rcParams.update({'font.size': 20})
    plt.figure(num=1, figsize=(8, 6), dpi=300)
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.15)

    # axis
    plt.ylim((0, 7))
    plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7],
               labels=["0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6", "1e7"])
    plt.xticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel("#steps from original circuit", fontsize=30, labelpad=10.0)
    plt.ylabel("#circuits", fontsize=30, labelpad=10.0)

    # plot
    plt.rcParams.update({'font.size': 12})
    bars = plt.bar(x_list, y_list)
    plt.bar_label(bars, fmt="%d", labels=raw_y_list)

    # plot
    plt.savefig(f"./figure1_search_space_dist_final_figures/figure1_search_space_dist.pdf")
    plt.show()


if __name__ == '__main__':
    main()
