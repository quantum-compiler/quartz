#pragma once

#include "quartz/context/context.h"
#include "quartz/generator/generator.h"

#include <chrono>

using namespace quartz;

void test_generator(const std::vector<GateType> &support_gates, int num_qubits,
                    int max_num_input_parameters, int max_num_gates,
                    bool verbose, const std::string &save_file_name,
                    bool count_minimal_representations = false) {
  ParamInfo param_info(/*num_input_symbolic_params=*/max_num_input_parameters);
  Context ctx(support_gates, num_qubits, &param_info);
  Generator generator(&ctx);
  Dataset dataset;
  auto start = std::chrono::steady_clock::now();
  EquivalenceSet equiv_set;
  generator.generate(num_qubits, max_num_gates, &dataset,
                     /*invoke_python_verifier=*/false, &equiv_set,
                     /*unique_parameters=*/false);
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
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;
  if (!save_file_name.empty()) {
    start = std::chrono::steady_clock::now();
    dataset.save_json(&ctx, save_file_name);
    end = std::chrono::steady_clock::now();
    std::cout << "Json saved in "
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                     end - start)
                         .count() /
                     1000.0
              << " seconds." << std::endl;
  }
  if (count_minimal_representations) {
    int num_different_minrep = 0;
    int num_missing_minrep = 0;
    std::unique_ptr<CircuitSeq> tmp_dag;
    for (auto &it : dataset.dataset) {
      bool has_minimal_representation = false;
      for (auto &dag : it.second) {
        bool result = dag->canonical_representation(&tmp_dag, &ctx);
        if (result) {
          has_minimal_representation = true;
        } else {
          if (dataset.dataset.count(tmp_dag->hash(&ctx)) == 0) {
            num_missing_minrep++;
          }
        }
      }
      if (has_minimal_representation) {
        num_different_minrep++;
      }
    }
    std::cout << "Found DAGs with " << num_different_minrep
              << " minimal circuit representations with different hash "
                 "values, among which "
              << num_missing_minrep << " minimal DAGs are missing."
              << std::endl;
  }
}
