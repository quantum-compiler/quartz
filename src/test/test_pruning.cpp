#include "test_pruning.h"

int main() {
    test_pruning({GateType::rz, GateType::h, GateType::cx, GateType::x,
                  GateType::add}, "Nam_4_", 3, 2, 4, true, 1, true, false, false, true);
    /*
  test_pruning({GateType::rz, GateType::h, GateType::cx, GateType::x,
                GateType::add}, "Nam_3_", 3, 2, 3, true, 1, true, true, false, true);
  test_pruning({GateType::rz, GateType::h, GateType::cx, GateType::x,
                GateType::add}, "Nam_4_", 3, 2, 4, true, 1, true, true, false, true);
  test_pruning({GateType::rz, GateType::h, GateType::cx, GateType::x,
                GateType::add}, "Nam_5_", 3, 2, 5, true, 1, true, true, false, true);
  test_pruning({GateType::u1, GateType::u2, GateType::cx,
                GateType::add}, "IBM_2_", 2, 2, 2, true, 1, true, true, false, true);
  test_pruning({GateType::u1, GateType::u2, GateType::cx,
                GateType::add}, "IBM_3_", 2, 2, 3, true, 1, true, true, false, true);
  test_pruning({GateType::u1, GateType::u2, GateType::cx,
                GateType::add}, "IBM_4_", 2, 2, 4, true, 1, true, true, false, true);
  test_pruning({GateType::rx, GateType::rz, GateType::cz,
                GateType::add}, "Rigetti_3_", 3, 2, 3, true, 1, true, true, false, true);
  test_pruning({GateType::rx, GateType::rz, GateType::cz,
                GateType::add}, "Rigetti_4_", 3, 2, 4, true, 1, true, true, false, true);
  test_pruning({GateType::rx, GateType::rz, GateType::cz,
                GateType::add}, "Rigetti_5_", 3, 2, 5, true, 1, true, true, true, false);
  */
     return 0;
}
