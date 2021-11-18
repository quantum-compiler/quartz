#include "../gate/gate.h"
#include "../dag/dag.h"
#include "../math/vector.h"
#include "../context/context.h"
#include "test_dataset.h"
#include "test_generator.h"

#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;
  Context ctx
      ({GateType::x, GateType::y, GateType::add, GateType::u2, GateType::cx});

  auto y = ctx.get_gate(GateType::y);
  y->get_matrix()->print();

  DAG dag(2, 0);
  dag.add_gate({0}, {}, y, nullptr);
  std::cout << "Is_minimal=" << dag.is_minimal_representation() << std::endl;

  Vector input_dis = Vector::random_generate(2);
  Vector output_dis;
  input_dis.print();
  dag.evaluate(input_dis, {}, output_dis);
  output_dis.print();

  DAG dag1(2, 2);
  DAG dag2(2, 2);
  int tmp;
  dag1.add_gate({}, {0, 0}, ctx.get_gate(GateType::add), &tmp);
  dag1.add_gate({}, {0, 1}, ctx.get_gate(GateType::add), &tmp);
  dag1.add_gate({}, {1, 1}, ctx.get_gate(GateType::add), &tmp);
  dag1.add_gate({0}, {3, 0}, ctx.get_gate(GateType::u2), &tmp);
  dag1.add_gate({1}, {3, 4}, ctx.get_gate(GateType::u2), &tmp);
  dag1.add_gate({0, 1}, {}, ctx.get_gate(GateType::cx), &tmp);
  dag1.add_gate({1}, {2, 4}, ctx.get_gate(GateType::u2), &tmp);

  dag2.add_gate({}, {0, 0}, ctx.get_gate(GateType::add), &tmp);
  dag2.add_gate({}, {0, 1}, ctx.get_gate(GateType::add), &tmp);
  dag2.add_gate({}, {1, 1}, ctx.get_gate(GateType::add), &tmp);
  dag2.add_gate({0}, {3, 0}, ctx.get_gate(GateType::u2), &tmp);
  dag2.add_gate({1}, {0, 4}, ctx.get_gate(GateType::u2), &tmp);
  dag2.add_gate({1, 0}, {}, ctx.get_gate(GateType::cx), &tmp);
  dag2.add_gate({0}, {0, 3}, ctx.get_gate(GateType::u2), &tmp);

  std::cout << std::hex << "dag1.hash() = " << dag1.hash(&ctx) << std::endl;
  std::cout << std::hex << "dag2.hash() = " << dag2.hash(&ctx) << std::endl;
  std::cout << dag1.to_json() << std::endl;
  std::cout << dag2.to_json() << std::endl;

  test_generator(/*support_gates=*/{GateType::x, GateType::rx,
                                    GateType::cx, GateType::add},
      /*num_qubits=*/3,
      /*max_num_input_parameters=*/2,
      /*max_num_gates=*/3,
      /*verbose=*/false,
      /*save_file_name=*/"data.json",
      /*count_minimal_representations=*/true);

  // Working directory is cmake-build-debug/ here.
  system("python ../test/test_verifier.py");

  test_equivalence_set(all_supported_gates(),
                       "equivalences.json",
                       "equivalences_simplified.json");
  return 0;
}
