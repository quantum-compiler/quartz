#include "test_generator.h"

int main() {
  test_generator(/*support_gates=*/{GateType::rx, GateType::ry,
                                    GateType::add,
                                    GateType::cx}, /*num_qubits=*/
                                   2, /*max_num_parameters=*/
                                   2, /*max_num_gates=*/
                                   2, /*verbose=*/
                                   true, /*save_file_name=*/"data.json");
  return 0;
}
