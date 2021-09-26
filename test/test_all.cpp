#include "../gate/gate.h"
#include "../dag/dag.h"
#include "../math/vector.h"
#include "../context/context.h"
#include "test_generator.h"

#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;
  Context ctx({GateType::x, GateType::y});

  auto y = ctx.get_gate(GateType::y);
  y->get_matrix()->print();

  DAG dag(2, 0);
  dag.add_gate({0}, {}, y, nullptr);

  Vector input_dis = Vector::random_generate(2);
  Vector output_dis;
  input_dis.print();
  dag.evaluate(input_dis, {}, output_dis);
  output_dis.print();

  test_generator(/*support_gates=*/{GateType::x, GateType::y, GateType::rx,
                                    GateType::cx, GateType::add},
      /*num_qubits=*/3,
      /*max_num_parameters=*/2,
      /*max_num_gates=*/4,
      /*verbose=*/false,
      /*save_file_name=*/"data.json");
  return 0;
}
