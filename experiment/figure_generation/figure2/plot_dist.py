import pickle
import argparse

import matplotlib.pyplot as plt


# This file plots distribution
def main():
    # input param
    parser = argparse.ArgumentParser(description='Generate figure for paper.')
    parser.add_argument('--gate_count', type=int, required=True)
    args = parser.parse_args()
    gate_count = f"{args.gate_count}"

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
    x_list[-1] = f"N/A"
    print(x_list)

    # plot
    plt.figure(num=1, figsize=(8, 6), dpi=300)
    plt.xlabel("min #xfers")
    plt.ylabel("percentage (%)")
    plt.yticks(ticks=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0],
               labels=["100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%", "0%"])
    plt.bar(x_list, y_list)
    plt.savefig(f"./figure2_final_figures/{gate_count}_dist.pdf")
    # plt.show()


if __name__ == '__main__':
    main()
