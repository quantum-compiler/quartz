#include "kernel.h"

namespace quartz {
std::string kernel_type_name(KernelType tp) {
  switch (tp) {
  case KernelType::fusion:
    return "fusion";
  case KernelType::shared_memory:
    return "shared_memory";
  }
  return "undefined";
}

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
