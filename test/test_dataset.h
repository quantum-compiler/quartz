#pragma once

#include "../dataset/dataset.h"
#include "../dataset/equivalence_set.h"

#include <chrono>

void test_equivalence_set(const std::vector<GateType> &support_gates,
                          const std::string &file_name) {
  Context ctx(support_gates);
  EquivalenceSet eqs;
  auto start = std::chrono::steady_clock::now();
  if (!eqs.load_json(&ctx, file_name)) {
    std::cout << "Failed to load equivalence file." << std::endl;
    return;
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << std::dec << eqs.num_equivalence_classes()
            << " classes of equivalences with " << eqs.num_total_dags()
            << " DAGs are loaded in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
  start = std::chrono::steady_clock::now();
  eqs.normalize_to_minimal_representations(&ctx);
  end = std::chrono::steady_clock::now();
  std::cout << "After normalizing to minimal representations in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds, "
            << eqs.num_equivalence_classes()
            << " classes of equivalences with " << eqs.num_total_dags()
            << " DAGs are found."
            << std::endl;
}
