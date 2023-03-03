#include "kernel.h"

namespace quartz {
std::string quartz::Kernel::to_string() const {
  std::string result;
  result += "qubits [";
  for (int j = 0; j < (int)qubits.size(); j++) {
    result += std::to_string(qubits[j]);
    if (j != (int)qubits.size() - 1) {
      result += ", ";
    }
  }
  result += "], gates ";
  result += gates.to_string();
  return result;
}
} // namespace quartz
