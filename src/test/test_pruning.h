#pragma once

#include "quartz/context/context.h"
#include "quartz/generator/generator.h"

#include <chrono>
#include <fstream>

using namespace quartz;

void test_pruning(const std::vector<GateType> &supported_gates,
                  const std::string &file_prefix, int num_qubits,
                  int num_input_parameters, int max_num_quantum_gates,
                  bool use_generated_file_if_possible = false,
                  int max_num_param_gates = 1,
                  bool run_representative_pruning = true,
                  bool run_original = true,
                  bool run_original_unverified = false,
                  bool run_original_verified = true) {
  Context ctx(supported_gates);
  Generator gen(&ctx);

  EquivalenceSet equiv_set;

  Dataset dataset1;
  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();
  decltype(end - start) running_time_with_all_pruning_techniques{0};
  int num_singletons = 0;

  if (run_representative_pruning) {
    std::ifstream fin(file_prefix + "pruning_unverified.json");
    if (fin.is_open() && use_generated_file_if_possible) {
      std::cout << "Representative pruning: use generated file."
                << std::endl;
      fin.close();
    } else {
      if (fin.is_open()) {
        fin.close();
      }
      start = std::chrono::steady_clock::now();
      gen.generate(num_qubits, num_input_parameters,
                   max_num_quantum_gates, max_num_param_gates,
                   &dataset1,        /*verify_equivalences=*/
                   true, &equiv_set, /*unique_parameters=*/false, /*verbose=*/
                   true);
      end = std::chrono::steady_clock::now();
      running_time_with_all_pruning_techniques += end - start;
      std::cout << std::dec
                << "Representative pruning: " << dataset1.num_total_dags()
                << " circuits with " << dataset1.num_hash_values()
                << " different hash values are found in "
                << (double)std::chrono::duration_cast<
                    std::chrono::milliseconds>(end - start)
                    .count() /
                    1000.0
                << " seconds." << std::endl;

      start = std::chrono::steady_clock::now();
      num_singletons = dataset1.remove_singletons(&ctx);
      end = std::chrono::steady_clock::now();
      running_time_with_all_pruning_techniques += end - start;
      std::cout << num_singletons << " singletons removed." << std::endl;

      start = std::chrono::steady_clock::now();
      dataset1.save_json(&ctx, file_prefix + "pruning_unverified.json");
      end = std::chrono::steady_clock::now();
      running_time_with_all_pruning_techniques += end - start;
      std::cout << std::dec << "Representative pruning: json saved in "
                << (double)std::chrono::duration_cast<
                    std::chrono::milliseconds>(end - start)
                    .count() /
                    1000.0
                << " seconds." << std::endl;

      dataset1.clear();
    }

    equiv_set.clear();
    start = std::chrono::steady_clock::now();
    system(("python src/python/verifier/verify_equivalences.py " + file_prefix +
        "pruning_unverified.json " + file_prefix + "pruning.json")
               .c_str());
    equiv_set.load_json(&ctx, file_prefix + "pruning.json");
    end = std::chrono::steady_clock::now();
    running_time_with_all_pruning_techniques += end - start;
    std::cout
        << std::dec << "Representative pruning: there are "
        << equiv_set.num_total_dags() + num_singletons << " circuits in "
        << equiv_set.num_equivalence_classes() + num_singletons
        << " equivalence classes after verification in "
        << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start)
            .count() /
            1000.0
        << " seconds." << std::endl;

    start = std::chrono::steady_clock::now();
    equiv_set.simplify(&ctx,  /*normalize_to_minimal_circuit_representation=*/false, /*common_subcircuit_pruning=*/
                       false, /*other_simplification=*/
                       true);
    equiv_set.save_json(file_prefix + "pruning_other_simplification.json");
    end = std::chrono::steady_clock::now();
    std::cout
        << std::dec << "Representative pruning: there are "
        << equiv_set.num_total_dags() << " circuits in "
        << equiv_set.num_equivalence_classes()
        << " equivalence classes after other simplification in "
        << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start)
            .count() /
            1000.0
        << " seconds." << std::endl;

    std::cout << "*** " << file_prefix.substr(0, file_prefix.size() - 1) << " Representative: "
              << equiv_set.num_total_dags() << " ("
              << equiv_set.num_equivalence_classes() << ")" << std::endl;

    equiv_set.clear();
    equiv_set.load_json(&ctx, file_prefix + "pruning.json");
    start = std::chrono::steady_clock::now();
    equiv_set.simplify(&ctx, /*normalize_to_minimal_circuit_representation=*/false);
    equiv_set.save_json(file_prefix + "pruning_simplified.json");
    end = std::chrono::steady_clock::now();
    running_time_with_all_pruning_techniques += end - start;
    std::cout
        << std::dec << "Representative pruning: there are "
        << equiv_set.num_total_dags() << " circuits in "
        << equiv_set.num_equivalence_classes()
        << " equivalence classes after all simplification in "
        << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start)
            .count() /
            1000.0
        << " seconds." << std::endl;

    std::cout << "*** " << file_prefix.substr(0, file_prefix.size() - 1) << " Common Subcircuit: "
              << equiv_set.num_total_dags() << " ("
              << equiv_set.num_equivalence_classes() << ")" << std::endl;

    std::cout << "*** " << file_prefix.substr(0, file_prefix.size() - 1) << " Running Time (s): "
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                  running_time_with_all_pruning_techniques)
                  .count() /
                  1000.0 << std::endl;
  }

  if (run_original) {
    std::ifstream fin(file_prefix + "original_unverified.json");
    bool missing_num_singletons = false;
    if (fin.is_open() && use_generated_file_if_possible) {
      std::cout << "Original: use generated file." << std::endl;
      fin.close();
      num_singletons = 0;
      missing_num_singletons = true;
    } else {
      if (fin.is_open()) {
        fin.close();
      }
      if (use_generated_file_if_possible) {
        std::cout << (file_prefix + "original_unverified.json")
                  << " not found. Generating..." << std::endl;
      }
      start = std::chrono::steady_clock::now();
      gen.generate_dfs(num_qubits, num_input_parameters,
                       max_num_quantum_gates, max_num_param_gates,
                       dataset1, /*restrict_search_space=*/
                       false);
      end = std::chrono::steady_clock::now();
      std::cout << std::dec << "Original: " << dataset1.num_total_dags()
                << " circuits with " << dataset1.num_hash_values()
                << " different hash values are found in "
                << (double)std::chrono::duration_cast<
                    std::chrono::milliseconds>(end - start)
                    .count() /
                    1000.0
                << " seconds." << std::endl;

      num_singletons = dataset1.remove_singletons(&ctx);
      std::cout << num_singletons << " singletons removed." << std::endl;

      start = std::chrono::steady_clock::now();
      dataset1.save_json(&ctx, file_prefix + "original_unverified.json");
      end = std::chrono::steady_clock::now();
      std::cout << std::dec << "Original: json saved in "
                << (double)std::chrono::duration_cast<
                    std::chrono::milliseconds>(end - start)
                    .count() /
                    1000.0
                << " seconds." << std::endl;
    }

    ctx.clear_representatives();
    dataset1.clear();

    if (run_original_unverified) {
      equiv_set.clear();
      start = std::chrono::steady_clock::now();
      // Do not invoke SMT solver to save time at first.
      system(("python src/python/verifier/verify_equivalences.py " + file_prefix +
          "original_unverified.json " + file_prefix + "original.json -n")
                 .c_str());
      equiv_set.load_json(&ctx, file_prefix + "original.json");
      end = std::chrono::steady_clock::now();
      if (missing_num_singletons) {
        std::cout << "Warning: missing num_singletons. The following two lines are inaccurate." << std::endl;
      }
      std::cout
          << std::dec << "Original unverified: there are "
          << equiv_set.num_total_dags() + num_singletons << " circuits in "
          << equiv_set.num_equivalence_classes() + num_singletons
          << " equivalence classes after verification in "
          << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
              end - start)
              .count() /
              1000.0
          << " seconds." << std::endl;
      std::cout << "*** " << file_prefix.substr(0, file_prefix.size() - 1) << " Original (Unverified): "
                << equiv_set.num_total_dags() + num_singletons << " ("
                << equiv_set.num_equivalence_classes() + num_singletons << ")" << std::endl;

      start = std::chrono::steady_clock::now();
      equiv_set.simplify(&ctx,  /*normalize_to_minimal_circuit_representation=*/false, /*common_subcircuit_pruning=*/
                         false, /*other_simplification=*/
                         true);
      equiv_set.save_json(file_prefix + "original_other_simplification.json");
      end = std::chrono::steady_clock::now();
      std::cout
          << std::dec << "Original unverified: there are "
          << equiv_set.num_total_dags() << " circuits in "
          << equiv_set.num_equivalence_classes()
          << " equivalence classes after other simplification in "
          << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
              end - start)
              .count() /
              1000.0
          << " seconds." << std::endl;
      std::cout << "*** " << file_prefix.substr(0, file_prefix.size() - 1) << " ECC Simplification (Unverified): "
                << equiv_set.num_total_dags() << " ("
                << equiv_set.num_equivalence_classes() << ")" << std::endl;
    }

    if (run_original_verified) {
      equiv_set.clear();
      start = std::chrono::steady_clock::now();
      system(("python src/python/verifier/verify_equivalences.py " + file_prefix +
          "original_unverified.json " + file_prefix +
          "original_verified.json")
                 .c_str());
      equiv_set.load_json(&ctx, file_prefix + "original_verified.json");
      end = std::chrono::steady_clock::now();
      if (missing_num_singletons) {
        std::cout << "Warning: missing num_singletons. The following two lines are inaccurate." << std::endl;
      }
      std::cout
          << std::dec << "Original verified: there are "
          << equiv_set.num_total_dags() + num_singletons << " circuits in "
          << equiv_set.num_equivalence_classes() + num_singletons
          << " equivalence classes after verification in "
          << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
              end - start)
              .count() /
              1000.0
          << " seconds." << std::endl;
      std::cout << "*** " << file_prefix.substr(0, file_prefix.size() - 1) << " Original: "
                << equiv_set.num_total_dags() + num_singletons << " ("
                << equiv_set.num_equivalence_classes() + num_singletons << ")" << std::endl;

      start = std::chrono::steady_clock::now();
      equiv_set.simplify(&ctx,  /*normalize_to_minimal_circuit_representation=*/false, /*common_subcircuit_pruning=*/
                         false, /*other_simplification=*/
                         true);
      equiv_set.save_json(file_prefix +
          "original_verified_other_simplification.json");
      end = std::chrono::steady_clock::now();
      std::cout
          << std::dec << "Original verified: there are "
          << equiv_set.num_total_dags() << " circuits in "
          << equiv_set.num_equivalence_classes()
          << " equivalence classes after other simplification in "
          << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
              end - start)
              .count() /
              1000.0
          << " seconds." << std::endl;
      std::cout << "*** " << file_prefix.substr(0, file_prefix.size() - 1) << " ECC Simplification: "
                << equiv_set.num_total_dags() << " ("
                << equiv_set.num_equivalence_classes() << ")" << std::endl;
    }
  }
}
