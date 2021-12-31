#include "test_pruning.h"

int main() {
//  test_pruning({GateType::u1, GateType::u2, GateType::u3, GateType::cx,
//                GateType::add}, "ibmq222_", 2, 2, 2, 1, true, true);
//  test_pruning({GateType::u1, GateType::u2, GateType::u3, GateType::cx,
//                GateType::add}, "ibmq223_", 2, 2, 3, 1, true, true);
  test_pruning({GateType::u1, GateType::u2, GateType::cx,
                GateType::add}, "ibmq_circuit_222_", 2, 2, 2, 1, true, true);
  test_pruning({GateType::u1, GateType::u2, GateType::cx,
                GateType::add}, "ibmq_circuit_223_", 2, 2, 3, 1, true, true);
  test_pruning({GateType::u1, GateType::u2, GateType::cx,
                GateType::add}, "ibmq_circuit_224_", 2, 2, 4, 1, true, true);
//  test_pruning({GateType::u1, GateType::u2, GateType::cx,
//                GateType::add}, "ibmq_no_u3_233_", 2, 3, 3, 1, false, true);
//  test_pruning({GateType::rz, GateType::h, GateType::cx, GateType::x,
//                GateType::add}, "rzhcxx_323_speed_", 3, 2, 3, 1, true, false);
//  test_pruning({GateType::rz, GateType::h, GateType::cx, GateType::x,
//                GateType::add}, "rzhcxx_324_speed_", 3, 2, 4, 1, true, false);
//  test_pruning({GateType::rz, GateType::h, GateType::cx, GateType::x,
//                GateType::add}, "rzhcxx_325_speed_", 3, 2, 5, 1, false, true);
//  test_pruning({GateType::rx, GateType::rz, GateType::cz,
//                GateType::add}, "rigetti_323_speed_", 3, 2, 3, 1, true, false);
//  test_pruning({GateType::rx, GateType::rz, GateType::cz,
//                GateType::add}, "rigetti_324_speed_", 3, 2, 4, 1, true, false);
//  test_pruning({GateType::rx, GateType::rz, GateType::cz,
//                GateType::add}, "rigetti_325_speed_", 3, 2, 5, 1, true, false);
//  test_pruning({GateType::u1, GateType::u2, GateType::cx,
//                GateType::add}, "ibmq_no_u3_225_", 2, 2, 5, 1, false, true);
  return 0;
}
