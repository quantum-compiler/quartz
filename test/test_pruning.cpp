#include "test_pruning.h"

int main() {
  test_pruning({GateType::u1, GateType::u2, GateType::u3, GateType::cx,
                GateType::add}, "ibmq_", 2, 3, 3, 1, true, false);
  return 0;
}
