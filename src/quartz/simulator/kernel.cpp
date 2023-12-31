#include "kernel.h"

#include "quartz/utils/string_utils.h"

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

KernelType string_to_kernel_type(const std::string &s) {
  if (s == "fusion") {
    return KernelType::fusion;
  } else if (s == "shared_memory") {
    return KernelType::shared_memory;
  } else {
    std::cerr << "Unknown kernel type " << s << std::endl;
    assert(false);
    return KernelType::fusion;
  }
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
  result += gates->to_string();
  return result;
}

Kernel Kernel::from_qasm_style_string(Context *ctx, const std::string &str) {
  std::stringstream ss(str);
  std::string token;
  KernelType tp;
  std::vector<int> qubits;
  while (ss >> token) {
    if (token == "//") {
      ss >> token;
      tp = string_to_kernel_type(token);
      break;
    }
  }
  while (ss >> token) {
    if (token == "//") {
      bool ok = read_json_style_vector(ss, qubits);
      assert(ok);
      break;
    }
  }
  std::getline(ss, token, '\0');  // extract the rest
  auto gates = CircuitSeq::from_qasm_style_string(ctx, token);
  return {std::move(gates), qubits, tp};
}

std::string Kernel::to_qasm_style_string(Context *ctx,
                                         int param_precision) const {
  std::string result = "// ";
  result += kernel_type_name(type);
  result += "\n// ";
  result += to_json_style_string(qubits);
  result += "\n";
  result += gates->to_qasm_style_string(ctx, param_precision);
  return result;
}

KernelCostType
Kernel::cost(const KernelCost &cost_function,
             const std::vector<int> &qubit_layout,
             const KernelCostType *customized_shared_memory_gate_cost) const {
  if (type == KernelType::fusion) {
    auto &vec = cost_function.get_fusion_kernel_costs();
    if (qubits.size() < vec.size()) {
      return vec[qubits.size()];
    } else {
      return (KernelCostType)(INFINITY);
    }
  } else if (type == KernelType::shared_memory) {
    if (qubits.size() > cost_function.get_shared_memory_num_free_qubits()) {
      // possible to be infeasible
      if (qubits.size() >
          cost_function.get_shared_memory_num_free_qubits() +
              cost_function.get_shared_memory_num_cacheline_qubits()) {
        // infeasible
        return (KernelCostType)(INFINITY);
      }
      // we need the local qubit layout to determine the cacheline qubits.
      assert(qubit_layout.size() >=
             cost_function.get_shared_memory_num_cacheline_qubits());
      int extra_qubits = (int)qubits.size() -
                         cost_function.get_shared_memory_num_free_qubits();
      for (auto &qubit : qubits) {
        if (std::find(
                qubit_layout.begin(),
                qubit_layout.begin() +
                    cost_function.get_shared_memory_num_cacheline_qubits(),
                qubit) !=
            qubit_layout.begin() +
                cost_function.get_shared_memory_num_cacheline_qubits()) {
          extra_qubits--;
          if (extra_qubits <= 0) {
            break;
          }
        }
      }
      if (extra_qubits > 0) {
        // infeasible
        return (KernelCostType)(INFINITY);
      }
    }
    auto result = cost_function.get_shared_memory_init_cost();
    if (customized_shared_memory_gate_cost == nullptr) {
      for (auto &gate : gates->gates) {
        result += cost_function.get_shared_memory_gate_cost(gate->gate->tp);
      }
    } else {
      result += *customized_shared_memory_gate_cost;
    }
    return result;
  } else {
    assert(false);
    return (KernelCostType)(INFINITY);
  }
}

bool Kernel::add_gate(CircuitGate *gate, Context *ctx,
                      const std::function<bool(int)> &is_local_qubit,
                      const std::vector<int> &customized_non_insular_qubits) {
  if (gates->add_gate(gate, ctx)) {
    auto gate_qubits = (type == KernelType::shared_memory
                            ? (customized_non_insular_qubits.empty()
                                   ? gate->get_non_insular_qubit_indices()
                                   : customized_non_insular_qubits)
                            : gate->get_qubit_indices());
    for (auto qubit : gate_qubits) {
      if (is_local_qubit(qubit) &&
          std::find(qubits.begin(), qubits.end(), qubit) == qubits.end()) {
        qubits.push_back(qubit);
      }
    }
    return true;
  }
  return false;
}

bool Kernel::verify(const std::function<bool(int)> &is_local_qubit) const {
  for (auto &gate : gates->gates) {
    const auto &gate_qubits = (type == KernelType::shared_memory
                                   ? gate->get_non_insular_qubit_indices()
                                   : gate->get_qubit_indices());
    const auto &insular_qubits = gate->get_insular_qubit_indices();
    for (auto qubit : gate_qubits) {
      if (is_local_qubit(qubit) &&
          std::find(qubits.begin(), qubits.end(), qubit) == qubits.end()) {
        std::cerr << "Qubit " << qubit << " of gate " << gate->to_string()
                  << " not active in kernel." << std::endl;
        return false;
      }
      if (!is_local_qubit(qubit)) {
        if (type == KernelType::shared_memory) {
          std::cerr << "Non-insular non-local qubit " << qubit
                    << " detected in a shared-memory kernel." << std::endl;
          return false;
        } else if (std::find(insular_qubits.begin(), insular_qubits.end(),
                             qubit) == insular_qubits.end()) {
          std::cerr << "Non-insular non-local qubit " << qubit
                    << " detected in a fusion kernel." << std::endl;
          return false;
        }
      }
    }
  }
  return true;
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

}  // namespace quartz
