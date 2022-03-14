#include "quartz/context/context.h"
#include "quartz/generator/generator.h"

#include <chrono>
#include <fstream>

using namespace quartz;

void gen_ecc_set(const std::vector<GateType> &supported_gates,
                 const std::string &file_prefix, bool unique_parameters,
                 int num_qubits,
                 int num_input_parameters, int max_num_quantum_gates,
                 int max_num_param_gates = 1) {
  Context ctx(supported_gates);
  ctx.get_and_gen_input_dis(3);
  ctx.get_and_gen_hashing_dis(3);
  ctx.get_and_gen_parameters(5);
  Generator gen(&ctx);

  EquivalenceSet equiv_set;

  Dataset dataset1;

  gen.generate(num_qubits, num_input_parameters, 1, max_num_param_gates,
               &dataset1,        /*verify_equivalences=*/
               true, &equiv_set, unique_parameters, /*verbose=*/
               false);
  std::cout << "*** ch(" << file_prefix.substr(0, file_prefix.size() - 5)
            << ") = "
            << dataset1.num_total_dags() - 1 /*exclude the empty circuit*/
            << std::endl;
  dataset1.clear();

  auto start = std::chrono::steady_clock::now();
  gen.generate(num_qubits, num_input_parameters, max_num_quantum_gates,
               max_num_param_gates, &dataset1, /*verify_equivalences=*/
               true, &equiv_set,  unique_parameters,             /*verbose=*/
               false);
  dataset1.remove_singletons(&ctx);
  dataset1.save_json(&ctx, file_prefix + "pruning_unverified.json");

  system(("python src/python/verifier/verify_equivalences.py " + file_prefix +
          "pruning_unverified.json " + file_prefix + "pruning.json")
             .c_str());
  equiv_set.load_json(&ctx, file_prefix + "pruning.json");
  equiv_set.simplify(&ctx,
                     /*normalize_to_minimal_circuit_representation=*/false);
  equiv_set.save_json(file_prefix + "complete_ECC_set.json");
  auto end = std::chrono::steady_clock::now();

  std::cout << file_prefix.substr(0, file_prefix.size() - 1)
            << " generated. Running Time (s): "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << std::endl;

  std::cout << "*** Number of transformations of "
            << file_prefix.substr(0, file_prefix.size() - 1) << " = "
            << (equiv_set.num_total_dags() -
                equiv_set.num_equivalence_classes()) *
                   2
            << std::endl;
}

int main() {
  gen_ecc_set(
      {GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add},
      "Nam_5_3_", false, 3, 2, 5);
  gen_ecc_set({GateType::u1, GateType::u2, GateType::cx, GateType::add},
              "IBM_4_2_", false, 2, 2, 4);
  gen_ecc_set({GateType::rx, GateType::rz, GateType::cz, GateType::add},
              "Rigetti_5_3_", false, 3, 2, 5);
  gen_ecc_set({GateType::h, GateType::cz}, "H_CZ_2_2_", false, 2, 0, 2);
  std::cout << "Now running IBM gate set with U3 gate, which may take a long time." << std::endl;
  gen_ecc_set({GateType::u1, GateType::u2, GateType::u3, GateType::add},
              "IBM_with_U3_2_1_", false, 1, 3, 2);
  return 0;
}
