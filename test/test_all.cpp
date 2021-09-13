#include "../gate/gate.h"
#include "../dag/dag.h"
#include "../math/vector.h"
#include "../context/context.h"

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
  return 0;
}
