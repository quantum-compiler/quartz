#pragma once

#include "quartz/circuitseq/circuitseq.h"

#include <string>
#include <vector>

namespace quartz {

enum KernelType { fusion, shared_memory };

std::string kernel_type_name(KernelType tp);

/**
 * A kernel used in the output of schedule.
 */
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

/**
 * A kernel used in the state of dynamic programming when computing the
 * schedule.
 */
struct KernelInDP {
public:
  KernelInDP(const std::vector<int> &active_qubits,
             const std::vector<int> touching_qubits, KernelType tp)
      : active_qubits(active_qubits), touching_qubits(touching_qubits), tp(tp) {
  }
  size_t get_hash() const;
  bool operator==(const KernelInDP &b) const;
  [[nodiscard]] std::string to_string() const;
  std::vector<int> active_qubits;
  std::vector<int> touching_qubits;
  KernelType tp;
};

} // namespace quartz
