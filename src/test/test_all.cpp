#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/gate/gate.h"
#include "quartz/math/vector.h"
#include "test_dataset.h"
#include "test_generator.h"

#include <iostream>

using namespace quartz;

int main() {
  std::cout << "Hello, World!" << std::endl;
  ParamInfo param_info(/*num_input_symbolic_params=*/2);
  Context ctx({GateType::x, GateType::y, GateType::add, GateType::neg,
               GateType::u2, GateType::u3, GateType::cx},
              2, &param_info);

  auto y = ctx.get_gate(GateType::y);
  y->get_matrix()->print();

  CircuitSeq dag(2);
  dag.add_gate({0}, {}, y, nullptr);
  std::cout << "Is_canonical=" << dag.is_canonical_representation()
            << std::endl;

  Vector input_dis = Vector::random_generate(2);
  Vector output_dis;
  input_dis.print();
  dag.evaluate(input_dis, {}, output_dis);
  output_dis.print();

  auto dag1 = std::make_unique<CircuitSeq>(2);
  auto dag2 = std::make_unique<CircuitSeq>(2);
  auto p2 =
      ctx.get_new_param_expression_id({0, 0}, ctx.get_gate(GateType::add));
  auto p3 =
      ctx.get_new_param_expression_id({0, 1}, ctx.get_gate(GateType::add));
  auto p4 =
      ctx.get_new_param_expression_id({1, 1}, ctx.get_gate(GateType::add));

  dag1->add_gate({0}, {p3, 0}, ctx.get_gate(GateType::u2), &ctx);
  dag1->add_gate({1}, {p3, p4}, ctx.get_gate(GateType::u2), &ctx);
  dag1->add_gate({0, 1}, {}, ctx.get_gate(GateType::cx), &ctx);
  dag1->add_gate({1}, {p2, p4}, ctx.get_gate(GateType::u2), &ctx);

  dag2->add_gate({0}, {p3, 0}, ctx.get_gate(GateType::u2), &ctx);
  dag2->add_gate({1}, {0, p4}, ctx.get_gate(GateType::u2), &ctx);
  dag2->add_gate({1, 0}, {}, ctx.get_gate(GateType::cx), &ctx);
  dag2->add_gate({0}, {0, p3}, ctx.get_gate(GateType::u2), &ctx);

  std::cout << std::hex << "dag1->hash() = " << dag1->hash(&ctx) << std::endl;
  std::cout << std::hex << "dag2->hash() = " << dag2->hash(&ctx) << std::endl;
  std::cout << dag1->to_json() << std::endl;
  std::cout << dag2->to_json() << std::endl;

  /*dag1 = std::make_unique<CircuitSeq>(1, 7);
  dag2 = std::make_unique<CircuitSeq>(1, 7);
  ctx.set_generated_parameter(0, 0);
  ctx.set_generated_parameter(3, 0);
  ctx.set_generated_parameter(6, std::acos(-1.0) / 2);
  dag1->add_gate({0}, {1, 2}, ctx.get_gate(GateType::u2), &tmp);
  dag1->add_gate({0}, {4, 5}, ctx.get_gate(GateType::u2), &tmp);
  dag2->add_gate({}, {2, 4}, ctx.get_gate(GateType::add), &tmp);  // 7
  dag2->add_gate({}, {7}, ctx.get_gate(GateType::neg), &tmp);  // 8
  dag2->add_gate({}, {6, 6}, ctx.get_gate(GateType::add), &tmp);  // 9
  dag2->add_gate({}, {8, 9}, ctx.get_gate(GateType::add), &tmp);  // 10
  dag2->add_gate({}, {10, 0}, ctx.get_gate(GateType::add), &tmp);  // 11
  dag2->add_gate({}, {11, 3}, ctx.get_gate(GateType::add), &tmp);  // 12
  dag2->add_gate({}, {1, 6}, ctx.get_gate(GateType::add), &tmp);  // 13
  dag2->add_gate({}, {5, 6}, ctx.get_gate(GateType::add), &tmp);  // 14
  dag2->add_gate({0}, {12, 13, 14}, ctx.get_gate(GateType::u3), &tmp);

  std::cout << std::hex << "dag1->hash() = " << dag1->hash(&ctx) << std::endl;
  std::cout << std::hex << "dag2->hash() = " << dag2->hash(&ctx) << std::endl;
  std::cout << dag1->to_json() << std::endl;
  std::cout << dag2->to_json() << std::endl;*/

  test_generator(/*support_gates=*/{GateType::x, GateType::rx, GateType::cx,
                                    GateType::add},
                 /*num_qubits=*/3,
                 /*max_num_input_parameters=*/2,
                 /*max_num_gates=*/3,
                 /*verbose=*/false,
                 /*save_file_name=*/"data.json",
                 /*count_minimal_representations=*/true);

  // Working directory is cmake-build-debug/ here.
  system("python ../src/test/test_verifier.py");

  test_equivalence_set(all_supported_gates(), "equivalences.json",
                       "equivalences_simplified.json");
  return 0;
}
