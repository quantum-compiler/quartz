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
  return 0;
}
