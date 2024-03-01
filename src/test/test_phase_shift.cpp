#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/dataset/dataset.h"
#include "quartz/dataset/equivalence_set.h"
#include "quartz/gate/gate.h"
#include "quartz/generator/generator.h"

#include <chrono>
#include <iostream>

using namespace quartz;

int main() {
  const int num_qubits = 1;
  const int num_input_parameters = 1;
  const int max_num_gates = 2;
  const int max_num_param_gates = 1;
  ParamInfo param_info(/*num_input_symbolic_params=*/num_input_parameters);
  Context ctx({GateType::rz, GateType::u1, GateType::add}, num_qubits,
              &param_info);

  Generator gen(&ctx);
  Dataset dataset;
  EquivalenceSet equiv_set;
  auto start = std::chrono::steady_clock::now();
  gen.generate(num_qubits, max_num_gates, &dataset, /*invoke_python_verifier=*/
               true, &equiv_set, /*unique_parameters=*/false, /*verbose=*/
               true);
  auto end = std::chrono::steady_clock::now();
  std::cout << std::dec << "Test phase shift with BFS verified: "
            << dataset.num_total_dags() << " circuits with "
            << dataset.num_hash_values()
            << " different hash values are found in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;
  dataset.save_json(&ctx, "phase_shift_before_verify.json");

  start = std::chrono::steady_clock::now();
  equiv_set.clear();
  system("python src/python/verifier/verify_equivalences.py "
         "phase_shift_before_verify.json phase_shift_verified.json True True");
  equiv_set.load_json(&ctx, "phase_shift_verified.json",
                      /*from_verifier=*/true);
  equiv_set.simplify(&ctx);
  equiv_set.save_json(&ctx, "phase_shift_verified_simplified.json");
  end = std::chrono::steady_clock::now();
  std::cout << std::dec << "Test phase shift with BFS verified: there are "
            << equiv_set.num_total_dags() << " circuits in "
            << equiv_set.num_equivalence_classes()
            << " equivalence classes after verification and simplification in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;

  return 0;
}
