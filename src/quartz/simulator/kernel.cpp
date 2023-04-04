#include "kernel.h"

#include <cassert>

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

bool KernelInDP::operator<(const KernelInDP &b) const {
  // Put all kernels with non-empty |active_qubits| at the beginning.
  if (!active_qubits.empty() && !b.active_qubits.empty()) {
    // And sort them in ascending order of the first active qubit.
    return active_qubits[0] < b.active_qubits[0];
  } else if (active_qubits.empty() != b.active_qubits.empty()) {
    // If this |active_qubits| is not empty, this is smaller.
    return b.active_qubits.empty();
  } else {
    // Assume we don't have kernels with empty |active_qubits| and empty
    // |touching_qubits|.
    assert(!touching_qubits.empty() && !b.touching_qubits.empty());
    // Sort kernels with empty |active_qubits| in ascending order of the first
    // touching qubit.
    return touching_qubits[0] < b.touching_qubits[0];
  }
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
