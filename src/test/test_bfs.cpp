#include "test_generator.h"

using namespace quartz;

int main() {
  const int num_qubits = 1;
  const int num_input_parameters = 0;
  const int max_num_quantum_gates = 2;
  const bool run_bfs_unverified = false;
  const bool run_bfs_verified = true;  // with representative pruning

  ParamInfo param_info(/*num_input_symbolic_params=*/num_input_parameters);
  Context ctx({GateType::h}, num_qubits, &param_info);
  Generator gen(&ctx);

  EquivalenceSet equiv_set;

  Dataset dataset1;
  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();

  if (run_bfs_unverified) {
    ctx.clear_representatives();

    Dataset dataset2;
    start = std::chrono::steady_clock::now();
    gen.generate(num_qubits, max_num_quantum_gates,
                 &dataset2,      /*verify_equivalences=*/
                 false, nullptr, /*unique_parameters=*/
                 false,          /*verbose=*/
                 true);
    end = std::chrono::steady_clock::now();
    std::cout << std::dec << "BFS unverified: " << dataset2.num_total_dags()
              << " Circuits with " << dataset2.num_hash_values()
              << " different hash values are found in "
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                     end - start)
                         .count() /
                     1000.0
              << " seconds." << std::endl;
    dataset2.save_json(&ctx, "bfs_unverified.json");

    start = std::chrono::steady_clock::now();
    equiv_set.clear();
    system("python src/python/verifier/verify_equivalences.py "
           "bfs_unverified.json "
           "bfs_unverified_verified.json");
    equiv_set.load_json(&ctx, "bfs_unverified_verified.json",
                        /*from_verifier=*/true);
    equiv_set.simplify(&ctx);
    equiv_set.save_json(&ctx, "bfs_unverified_simplified.json");
    end = std::chrono::steady_clock::now();
    std::cout
        << std::dec << "BFS unverified: there are "
        << equiv_set.num_total_dags() << " circuits in "
        << equiv_set.num_equivalence_classes()
        << " equivalence classes after verification and simplification in "
        << (double)std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                   .count() /
               1000.0
        << " seconds." << std::endl;

    ctx.clear_representatives();
  }

  if (run_bfs_verified) {
    Dataset dataset3;
    start = std::chrono::steady_clock::now();
    gen.generate(num_qubits, max_num_quantum_gates,
                 &dataset3,        /*verify_equivalences=*/
                 true, &equiv_set, /*unique_parameters=*/
                 false,            /*verbose=*/
                 true);
    end = std::chrono::steady_clock::now();
    std::cout << std::dec << "BFS verified: " << dataset3.num_total_dags()
              << " circuits with " << dataset3.num_hash_values()
              << " different hash values are found in "
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                     end - start)
                         .count() /
                     1000.0
              << " seconds." << std::endl;
    dataset3.save_json(&ctx, "tmp_before_verify.json");

    start = std::chrono::steady_clock::now();
    equiv_set.clear();
    system("python src/python/verifier/verify_equivalences.py "
           "tmp_before_verify.json "
           "bfs_verified.json");
    equiv_set.load_json(&ctx, "bfs_verified.json", /*from_verifier=*/true);
    equiv_set.simplify(&ctx);
    equiv_set.save_json(&ctx, "bfs_verified_simplified.json");
    end = std::chrono::steady_clock::now();
    std::cout
        << std::dec << "BFS verified: there are " << equiv_set.num_total_dags()
        << " circuits in " << equiv_set.num_equivalence_classes()
        << " equivalence classes after verification and simplification in "
        << (double)std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                   .count() /
               1000.0
        << " seconds." << std::endl;

    auto result = equiv_set.first_class_with_common_first_or_last_gates();
    if (result == -1) {
      std::cout << "No common first or last gates." << std::endl;
    } else {
      std::cout << "Found common first or last gates in "
                << equiv_set.get_class_id(result) << std::endl;
    }
  }
  return 0;
}
