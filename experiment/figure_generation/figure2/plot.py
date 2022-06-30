import pickle

import matplotlib.pyplot as plt


# This file plot cdf.
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

    # plot
    plt.xlabel("min #xfers")
    plt.ylabel("percentage")
    plt.plot(x_list, y_list, marker='o')
    plt.savefig(f"{gate_count}_cdf.jpg")
    plt.show()


if __name__ == '__main__':
    main()
