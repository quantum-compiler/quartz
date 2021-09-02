#include "../gate/all_gates.h"
#include "../dagcircuit/dagcircuit.h"
#include "../math/matrix.h"

#include "iostream"

int main() {
  std::cout << "Hello, World!" << std::endl;
  XGate x;
  x.to_matrix()->print();
  return 0;
}
