#include "../gate/all_gates.h"
#include "../dag/dag.h"
#include "../math/matrix.h"

#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;
  std::unique_ptr<Gate> y = std::make_unique<YGate>();
  y->to_matrix()->print();
  return 0;
}
