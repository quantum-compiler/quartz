#include "test_generator.h"

int main() {
  test_generator(/*support_gates=*/{GateType::rx, GateType::ry,
                                    GateType::add}, /*num_qubits=*/
                                   1, /*max_num_parameters=*/
                                   2, /*max_num_gates=*/
                                   2, /*verbose=*/
                                   true);
  return 0;
}
