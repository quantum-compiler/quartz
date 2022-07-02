#pragma once

#include "quartz/context/context.h"
#include "quartz/generator/generator.h"

namespace quartz {
void gen_ecc_set(const std::vector<GateType> &supported_gates,
                 const std::string &file_prefix, bool unique_parameters,
                 int num_qubits, int num_input_parameters,
                 int max_num_quantum_gates, int max_num_param_gates = 1);

void gen_ecc_set(const std::vector<GateType> &supported_gates,
                 const std::string &file_prefix, bool unique_parameters,
                 int num_qubits, int num_input_parameters,
                 int max_num_quantum_gates, int max_num_param_gates) {
  Context ctx(supported_gates, num_qubits, num_input_parameters);
  Generator gen(&ctx);

  EquivalenceSet equiv_set;

  Dataset dataset1;

  gen.generate(num_qubits, num_input_parameters, 1, max_num_param_gates,
               &dataset1,                           /*verify_equivalences=*/
               true, &equiv_set, unique_parameters, /*verbose=*/
               false);
  std::cout << "*** ch(" << file_prefix.substr(0, file_prefix.size() - 5)
            << ",q=" << file_prefix[file_prefix.size() - 2] << ") = "
            << dataset1.num_total_dags() - 1 /*exclude the empty circuit*/
            << std::endl;
  dataset1.clear();

  auto start = std::chrono::steady_clock::now();
  decltype(start - start) verification_time{0};
  gen.generate(num_qubits, num_input_parameters, max_num_quantum_gates,
               max_num_param_gates, &dataset1,      /*verify_equivalences=*/
               true, &equiv_set, unique_parameters, /*verbose=*/
               true, &verification_time);
  dataset1.remove_singletons(&ctx);
  dataset1.save_json(&ctx, file_prefix + "pruning_unverified.json");

  auto start2 = std::chrono::steady_clock::now();
  system(("python src/python/verifier/verify_equivalences.py " + file_prefix +
      "pruning_unverified.json " + file_prefix + "pruning.json")
             .c_str());
  auto end2 = std::chrono::steady_clock::now();
  verification_time += end2 - start2;
  equiv_set.clear(); // this is necessary
  equiv_set.load_json(&ctx, file_prefix + "pruning.json");
  std::cout << "Before ECC simplification: num_total_dags = "
            << equiv_set.num_total_dags()
            << ", num_equivalence_classes = "
            << equiv_set.num_equivalence_classes() << ", #transformations "
            << file_prefix.substr(0, file_prefix.size() - 1) << " = "
            << (equiv_set.num_total_dags() -
                equiv_set.num_equivalence_classes()) *
                2
            << std::endl;
  start2 = std::chrono::steady_clock::now();
  equiv_set.simplify(&ctx);
  end2 = std::chrono::steady_clock::now();
  equiv_set.save_json(file_prefix + "complete_ECC_set.json");
  auto end = std::chrono::steady_clock::now();

  std::cout << file_prefix.substr(0, file_prefix.size() - 1)
            << " generated. Running Time (s): "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start)
                .count() /
                1000.0
            << std::endl;
  std::cout << "Pruning Time (s): "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end2 - start2)
                .count() /
                1000.0
            << std::endl;
  std::cout << "Verification Time (s): "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                verification_time)
                .count() /
                1000.0
            << std::endl;
  std::cout << "Num_total_dags = " << equiv_set.num_total_dags()
            << ", num_equivalence_classes = "
            << equiv_set.num_equivalence_classes() << std::endl;

  std::cout << "*** Number of transformations of "
            << file_prefix.substr(0, file_prefix.size() - 1) << " = "
            << (equiv_set.num_total_dags() -
                equiv_set.num_equivalence_classes()) *
                2
            << std::endl;
}
}
