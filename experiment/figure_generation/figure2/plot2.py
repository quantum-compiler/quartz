import pickle

import matplotlib.pyplot as plt


# This file plots distribution
def main():
    # input param
    gate_count = 46

    with open(f"./raw_results/final_results_{gate_count}.pickle", 'rb') as handle:
        result_dict = pickle.load(file=handle)
    for i in range(8):
        if i + 1 not in result_dict:
            result_dict[i + 1] = 0
    if -1 not in result_dict:
        result_dict[-1] = 0

    # prepare y_list
    y_list = []
    for i in range(max(result_dict)):
        y_list.append(result_dict[i + 1])
    y_list.append(result_dict[-1])
    total = sum(y_list)
    y_list = [y_list[idx] / total for idx in range(len(y_list))]
    print(total, y_list)

    # prepare x_list
    x_list = list(range(1, max(result_dict) + 2))
    for idx in range(len(x_list)):
        x_list[idx] = f"{x_list[idx]}"
    x_list[-1] = f">{x_list[-2]}"
    print(x_list)

    # plot
    plt.xlabel("min #xfers")
    plt.ylabel("percentage")
    plt.bar(x_list, y_list)
    plt.savefig(f"{gate_count}_dist.jpg")
    plt.show()


if __name__ == '__main__':
    main()
