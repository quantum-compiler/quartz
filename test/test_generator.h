#pragma once

#include "../generator/generator.h"

#include <chrono>

void test_generator(const std::vector<GateType> &support_gates,
                    int num_qubits,
                    int max_num_parameters,
                    int max_num_gates,
                    bool verbose) {
  Context ctx(support_gates);
  Generator generator(&ctx);
  Dataset dataset;
  auto start = std::chrono::steady_clock::now();
  generator.generate(num_qubits, max_num_parameters, max_num_gates, dataset);
  auto end = std::chrono::steady_clock::now();
  if (verbose) {
    std::cout << "{" << std::endl;
    bool start0 = true;
    for (auto &it : dataset.dataset) {
      if (start0)
        start0= false;
      else
        std::cout << ",";
      std::cout << "\"" << std::hex << it.first << "\": [" << std::endl;
      bool start = true;
      for (auto &dag : it.second) {
	if (start)
          start = false;
	else
          std::cout << ",";
        std::cout << dag->to_string();
      }
      std::cout << "]" << std::endl;
    }
    std::cout << "}" << std::endl;
  }
  //std::cout << std::dec << "Circuits with " << dataset.size()
  //          << " different hash values are found in "
  //          << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
  //              end - start).count() / 1000.0 << " seconds."
  //          << std::endl;
}
