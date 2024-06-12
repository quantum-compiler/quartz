#pragma once

#include "quartz/context/context.h"
#include "quartz/generator/generator.h"

#include <filesystem>

namespace quartz {
void gen_ecc_set(const std::vector<GateType> &supported_gates,
                 const std::string &file_prefix, bool unique_parameters,
                 bool generate_representative_set, int num_qubits,
                 int num_input_parameters, int max_num_quantum_gates);

void gen_ecc_set(const std::vector<GateType> &supported_gates,
                 const std::string &file_prefix, bool unique_parameters,
                 bool generate_representative_set, int num_qubits,
                 int num_input_parameters, int max_num_quantum_gates) {
  ParamInfo param_info(/*num_input_symbolic_params=*/num_input_parameters);
  Context ctx(supported_gates, num_qubits, &param_info);
  Generator gen(&ctx);

  EquivalenceSet equiv_set;

  Dataset dataset1;

  gen.generate(num_qubits, 1, &dataset1,            /*invoke_python_verifier=*/
               true, &equiv_set, unique_parameters, /*verbose=*/
               false);
  std::cout << "*** ch(" << file_prefix.substr(0, file_prefix.size() - 5)
            << ",q=" << file_prefix[file_prefix.size() - 2] << ") = "
            << dataset1.num_total_dags() - 1 /*exclude the empty circuit*/
            << std::endl;
  dataset1.clear();

  auto start = std::chrono::steady_clock::now();
  decltype(start - start) verification_time{0};
  const bool invoke_python_verifier = (num_input_parameters > 0);
  // We are not going to invoke the Python verifier when |num_input_parameters|
  // is 0. This will be simply verifying two static matrices are equal.
  //
  // If we mistakenly treat two different matrices as equal due to
  // floating-point error, our optimizations will still preserve the
  // resulting circuit matrix up to an error of a small number times machine
  // precision. Plus, this situation is known to be extremely rare.
  gen.generate(num_qubits, max_num_quantum_gates, &dataset1,
               invoke_python_verifier, &equiv_set,
               unique_parameters, /*verbose=*/
               true, &verification_time);
  if (!generate_representative_set) {
    // For better performance
    dataset1.remove_singletons(&ctx);
  }
  if (invoke_python_verifier) {
    dataset1.save_json(&ctx, file_prefix + "pruning_unverified.json");

    auto start2 = std::chrono::steady_clock::now();
    system(("python " + kQuartzRootPath.string() +
            "/src/python/verifier/verify_equivalences.py " + file_prefix +
            "pruning_unverified.json " + file_prefix + "pruning.json")
               .c_str());
    auto end2 = std::chrono::steady_clock::now();
    verification_time += end2 - start2;
    equiv_set.clear();  // this is necessary
    equiv_set.load_json(&ctx, file_prefix + "pruning.json",
                        /*from_verifier=*/true);
  } else {
    // Create the ECC set by ourselves.
    equiv_set.clear();  // this is necessary
    for (auto &it : dataset1.dataset) {
      auto ecc = std::make_unique<EquivalenceClass>();
      ecc->set_dags(std::move(it.second));
      equiv_set.insert_class(&ctx, std::move(ecc));
    }
  }
  if (generate_representative_set) {
    auto start2 = std::chrono::steady_clock::now();
    auto rep_set = equiv_set.get_representative_set();
    rep_set->sort();
    rep_set->save_json(file_prefix + "representative_set.json");
    auto end2 = std::chrono::steady_clock::now();
    std::cout << "Representative set saved in "
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                     end2 - start2)
                         .count() /
                     1000.0
              << " seconds." << std::endl;
  }
  std::cout << "Before ECC simplification: num_total_dags = "
            << equiv_set.num_total_dags() << ", num_equivalence_classes = "
            << equiv_set.num_equivalence_classes() << ", #transformations "
            << file_prefix.substr(0, file_prefix.size() - 1) << " = "
            << (equiv_set.num_total_dags() -
                equiv_set.num_equivalence_classes()) *
                   2
            << std::endl;
  auto start2 = std::chrono::steady_clock::now();
  equiv_set.simplify(&ctx);
  auto end2 = std::chrono::steady_clock::now();
  equiv_set.save_json(&ctx, file_prefix + "complete_ECC_set.json");
  auto end = std::chrono::steady_clock::now();

  std::cout << file_prefix.substr(0, file_prefix.size() - 1)
            << " generated. Running Time (s): "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << std::endl;
  std::cout << "Pruning Time (s): "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end2 - start2)
                       .count() /
                   1000.0
            << std::endl;
  std::cout << "Verification Time (s): "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
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
}  // namespace quartz
