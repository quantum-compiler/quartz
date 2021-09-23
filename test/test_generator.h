#pragma once

#include "../generator/generator.h"

#include <chrono>

void test_generator(const std::vector<GateType> &support_gates,
                    int num_qubits,
                    int max_num_parameters,
                    int max_num_gates,
                    bool verbose,
                    const std::string &save_file_name) {
  Context ctx(support_gates);
  Generator generator(&ctx);
  Dataset dataset;
  auto start = std::chrono::steady_clock::now();
  generator.generate(num_qubits, max_num_parameters, max_num_gates, dataset);
  auto end = std::chrono::steady_clock::now();
  if (verbose) {
    for (auto &it : dataset.dataset) {
      std::cout << std::hex << it.first << ":" << std::endl;
      for (auto &dag : it.second) {
        std::cout << dag->to_string();
      }
    }
  }
  std::cout << std::dec << "Circuits with " << dataset.dataset.size()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  if (!save_file_name.empty()) {
    start = std::chrono::steady_clock::now();
    dataset.save_json(save_file_name);
    end = std::chrono::steady_clock::now();
    std::cout << "Json saved in "
              << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                  end - start).count() / 1000.0 << " seconds.";
  }
}
