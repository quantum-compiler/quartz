import pickle

import matplotlib.pyplot as plt


# This file plots distribution of search space
def main():
    # x_list and y_list
    y_list = [18, 176, 1208, 6463, 28513, 107573, 356345]
    x_list = list(range(1, len(y_list) + 1))
    print(x_list)
    print(y_list)

    # plot
    fig = plt.figure(num=1, figsize=(8, 6), dpi=300)
    ax = fig.subplots()
    bars = ax.bar(x_list, y_list)
    ax.set_xlabel("#steps from original circuit")
    ax.set_ylabel("#circuits")
    ax.bar_label(bars)
    plt.savefig(f"search_space_dist.pdf")
    plt.show()


if __name__ == '__main__':
    main()
