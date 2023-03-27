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

std::string Kernel::to_string() const {
  std::string result;
  result += kernel_type_name(type);
  result += ", qubits [";
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

size_t KernelInDP::get_hash() const {
  size_t result = 5381 + (int)tp;
  for (const auto &i : active_qubits) {
    result = result * 33 + i;
  }
  for (const auto &i : touching_qubits) {
    result = result * 33 + i;
  }
  return result;
}

bool KernelInDP::operator==(const KernelInDP &b) const {
  if (tp != b.tp) {
    return false;
  }
  if (active_qubits.size() != b.active_qubits.size()) {
    return false;
  }
  for (int j = 0; j < (int)active_qubits.size(); j++) {
    if (active_qubits[j] != b.active_qubits[j]) {
      return false;
    }
  }
  if (touching_qubits.size() != b.touching_qubits.size()) {
    return false;
  }
  for (int j = 0; j < (int)touching_qubits.size(); j++) {
    if (touching_qubits[j] != b.touching_qubits[j]) {
      return false;
    }
  }
  return true;
}

std::string KernelInDP::to_string() const {
  std::string result;
  result += kernel_type_name(tp);
  result += "{";
  for (int j = 0; j < (int)active_qubits.size(); j++) {
    result += std::to_string(active_qubits[j]);
    if (j != (int)active_qubits.size() - 1) {
      result += ", ";
    }
  }
  result += "}";
  if (!touching_qubits.empty()) {
    result += " touching {";
    for (int j = 0; j < (int)touching_qubits.size(); j++) {
      result += std::to_string(touching_qubits[j]);
      if (j != (int)touching_qubits.size() - 1) {
        result += ", ";
      }
    }
    result += "}";
  }
  return result;
}

} // namespace quartz
