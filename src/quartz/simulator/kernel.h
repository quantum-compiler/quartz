#pragma once

#include "quartz/circuitseq/circuitseq.h"

#include <string>
#include <vector>

namespace quartz {

enum KernelType { fusion, shared_memory };

class Kernel {
public:
  Kernel(const CircuitSeq &gates, const std::vector<int> &qubits,
         KernelType type)
      : gates(gates), qubits(qubits), type(type) {}

  [[nodiscard]] std::string to_string() const;
  CircuitSeq gates;
  std::vector<int> qubits;
  KernelType type;
};

} // namespace quartz
