import pickle


# This file aggregates data into one file
def main():
    # read files
    file_suffix_list = [46, 48, 50, 51, 52, 53, 54, 56, 58]
    final_result_dict = {}
    for i in list(range(1, 9)) + [-1]:
        final_result_dict[i] = 0
    for file_suffix in file_suffix_list:
        with open(f"./raw_results/final_results_{file_suffix}.pickle", 'rb') as handle:
            result_dict = pickle.load(file=handle)
            for idx in result_dict:
                final_result_dict[idx] += result_dict[idx]

    # output
    print(final_result_dict)
    total = 0
    for idx in final_result_dict:
        total += final_result_dict[idx]
    print(total)
    with open(f"./raw_results/final_results_all.pickle", 'wb') as handle:
        pickle.dump(final_result_dict, handle)


if __name__ == '__main__':
    main()
