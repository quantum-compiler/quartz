#pragma once

#include "quartz/dataset/dataset.h"
#include "quartz/dataset/equivalence_set.h"

#include <chrono>

using namespace quartz;

void test_equivalence_set(const std::vector<GateType> &support_gates,
                          const std::string &file_name,
                          const std::string &save_file_name) {
  ParamInfo param_info;
  Context ctx(support_gates, &param_info);
  EquivalenceSet eqs;
  auto start = std::chrono::steady_clock::now();
  if (!eqs.load_json(&ctx, file_name, /*from_verifier=*/false)) {
    std::cout << "Failed to load equivalence file." << std::endl;
    return;
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << std::dec << eqs.num_equivalence_classes()
            << " classes of equivalences with " << eqs.num_total_dags()
            << " DAGs are loaded in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;
  start = std::chrono::steady_clock::now();
  eqs.simplify(&ctx);
  end = std::chrono::steady_clock::now();
  std::cout << "After simplification in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds, " << eqs.num_equivalence_classes()
            << " classes of equivalences with " << eqs.num_total_dags()
            << " DAGs are found." << std::endl;
  if (!save_file_name.empty()) {
    start = std::chrono::steady_clock::now();
    eqs.save_json(&ctx, save_file_name);
    end = std::chrono::steady_clock::now();
    std::cout << "Json saved in "
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                     end - start)
                         .count() /
                     1000.0
              << " seconds." << std::endl;
  }
}
