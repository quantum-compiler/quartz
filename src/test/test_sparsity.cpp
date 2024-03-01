#include "quartz/context/context.h"
#include "quartz/generator/generator.h"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>

using namespace quartz;

// Count number of non-zero entries.
int nnz(const std::vector<Vector> &mat, double eps) {
  const auto &sz = mat.size();
  int result = 0;
  for (auto &col : mat) {
    for (int i = 0; i < sz; i++) {
      if (std::abs(col[i]) > eps) {
        result++;
      }
    }
  }
  return result;
}

void test_sparsity(const std::vector<GateType> &supported_gates,
                   const std::string &file_prefix, int num_qubits,
                   int num_input_parameters, int max_num_quantum_gates) {
  ParamInfo param_info(/*num_input_symbolic_params=*/num_input_parameters);
  Context ctx(supported_gates, num_qubits, &param_info);
  Generator gen(&ctx);

  EquivalenceSet equiv_set;

  Dataset dataset;
  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();

  start = std::chrono::steady_clock::now();
  gen.generate(num_qubits, max_num_quantum_gates, &dataset, true, &equiv_set,
               /*unique_parameters=*/false, true);
  end = std::chrono::steady_clock::now();
  std::cout << std::dec << dataset.num_total_dags() << " circuits with "
            << dataset.num_hash_values()
            << " different hash values are found in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;

  start = std::chrono::steady_clock::now();
  dataset.save_json(&ctx, file_prefix + "unverified.json");
  end = std::chrono::steady_clock::now();
  std::cout << std::dec << "Json saved in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;

  dataset.clear();

  start = std::chrono::steady_clock::now();
  system(("python src/python/verifier/verify_equivalences.py " + file_prefix +
          "unverified.json " + file_prefix + "verified.json")
             .c_str());
  equiv_set.clear();
  equiv_set.load_json(&ctx, file_prefix + "verified.json",
                      /*from_verifier=*/true);
  equiv_set.normalize_to_canonical_representations(&ctx);
  end = std::chrono::steady_clock::now();
  std::cout << std::dec << "There are " << equiv_set.num_total_dags()
            << " circuits in " << equiv_set.num_equivalence_classes()
            << " equivalence classes after verification in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;

  auto eccs = equiv_set.get_all_equivalence_sets();
  std::map<int, int> nnz_distribution;
  std::map<int, int> nnz_distribution_non_singleton;
  std::map<int, int> nnz_distribution_most_sparse;

  constexpr double kNNZEPS = 1e-6;

  for (auto &ecc : eccs) {
    assert(!ecc.empty());
    auto mat = ecc[0]->get_matrix(&ctx);
    auto num = nnz(mat, kNNZEPS);
    nnz_distribution[num]++;
    if (ecc.size() > 1) {
      nnz_distribution_non_singleton[num]++;
      for (int i = 1; i < ecc.size(); i++) {
        auto num_i = nnz(ecc[i]->get_matrix(&ctx), kNNZEPS);
        if (num_i < num) {
          num = num_i;
        }
      }
      nnz_distribution_most_sparse[num]++;
    }
  }

  std::cout << "Number of non-zero entries in the " << (1 << num_qubits)
            << " by " << (1 << num_qubits)
            << " matrix representations of representative circuits:"
            << std::endl;
  for (auto &it : nnz_distribution) {
    std::cout << "nnz=" << it.first << ": " << it.second << " circuits."
              << std::endl;
  }

  std::cout << "Number of non-zero entries in the " << (1 << num_qubits)
            << " by " << (1 << num_qubits)
            << " matrix representations of representative circuits (singleton "
               "ECCs removed):"
            << std::endl;
  for (auto &it : nnz_distribution_non_singleton) {
    std::cout << "nnz=" << it.first << ": " << it.second << " circuits."
              << std::endl;
  }

  std::cout << "If we choose the most sparse ones as representatives:"
            << std::endl;
  for (auto &it : nnz_distribution_most_sparse) {
    std::cout << "nnz=" << it.first << ": " << it.second << " circuits."
              << std::endl;
  }

  equiv_set.simplify(&ctx);
  eccs = equiv_set.get_all_equivalence_sets();
  // <matrix size, nnz>
  std::map<std::pair<int, int>, int> nnz_distribution_pair;

  for (auto &ecc : eccs) {
    assert(!ecc.empty());
    auto mat = ecc[0]->get_matrix(&ctx);
    auto num = nnz(mat, kNNZEPS);
    nnz_distribution_pair[std::make_pair(mat.size(), num)]++;
  }
  std::cout << "After all optimizations on the equivalence set (note that many "
               "ECCs are pruned here):"
            << std::endl;
  for (auto &it : nnz_distribution_pair) {
    std::cout << "matrix size=" << it.first.first << "x" << it.first.first
              << ", nnz=" << it.first.second << ": " << it.second
              << " circuits." << std::endl;
  }
}

int main() {
  test_sparsity(
      {GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add},
      "nam_circuit_324_", 3, 2, 4);
  return 0;
}
