#pragma once

#include "../gate/gate.h"
#include "../utils/utils.h"

#include <istream>
#include <queue>
#include <unordered_map>
#include <vector>

namespace quartz {

class CircuitWire;
class Context;

/**
 * A gate in the circuit.
 * Stores the gate type, input and output information.
 */
class CircuitGate {
 public:
  // Get the minimum qubit index of the gate.
  [[nodiscard]] int get_min_qubit_index() const;
  // Get the qubit indices of the gate.
  [[nodiscard]] std::vector<int> get_qubit_indices() const;
  // Get the control qubit indices of the gate if it is a controlled gate.
  [[nodiscard]] std::vector<int> get_control_qubit_indices() const;
  // Get the "insular" qubit indices of the gate.
  [[nodiscard]] std::vector<int> get_insular_qubit_indices() const;
  // Get the "non-insular" qubit indices of the gate.
  [[nodiscard]] std::vector<int> get_non_insular_qubit_indices() const;
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] std::string to_json() const;
  /**
   * Extract the information to reconstruct a gate in a circuit from an input
   * Json file. The first '[' character may or may not be already read.
   * @param fin The input stream (read and written).
   * @param ctx The context to reconstruct the gate (read and written).
   * @param input_qubits The input qubit indices of the gate (written).
   * @param input_params The input parameter indices of the gate (written).
   * @param output_qubits The output qubit indices of the gate (written).
   * @param output_params The output parameter indices of the gate (written).
   * @param gate The gate type (written).
   */
  static void read_json(std::istream &fin, Context *ctx,
                        std::vector<int> &input_qubits,
                        std::vector<int> &input_params,
                        std::vector<int> &output_qubits,
                        std::vector<int> &output_params, Gate *&gate);
  [[nodiscard]] std::string to_qasm_style_string(Context *ctx,
                                                 int param_precision) const;

  /**
   * Check if two gates in two circuits are equivalent given a mapping of
   * circuit wires.
   * @param this_gate A gate in the first circuit.
   * @param other_gate A gate in the second circuit.
   * @param wires_mapping A mapping from the wires in the first circuit to the
   * wires in the second circuit.
   * @param update_mapping If true, map the output wires, and push the output
   * wires in the first circuit to the queue;
   * if false, also check if the output wires are already correctly mapped
   * (only return true if the gates are equivalent and the output wires are
   * correctly mapped).
   * @param wires_to_search When |update_mapping| is true and the two gates
   * are equivalent under the wires mapping, store the new wires to search in
   * the topological sort procedure. Otherwise, this parameter has no effect.
   * When |update_mapping| is false, this parameter can be nullptr.
   * @param backward If true, the topological sort is performed from the end
   * of the circuit to the beginning. Only the output wires instead of the
   * input wires are compared when |update_mapping| is true.
   * @return True iff the two gates are equivalent under the wires mapping.
   */
  static bool
  equivalent(const CircuitGate *this_gate, const CircuitGate *other_gate,
             std::unordered_map<CircuitWire *, CircuitWire *> &wires_mapping,
             bool update_mapping, std::queue<CircuitWire *> *wires_to_search,
             bool backward = false);

  std::vector<CircuitWire *> input_wires;  // Include parameters!
  std::vector<CircuitWire *> output_wires;

  Gate *gate;
};
}  // namespace quartz
