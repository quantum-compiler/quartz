#include "verifier.h"

#include "quartz/dataset/dataset.h"
#include "quartz/tasograph/tasograph.h"
#include "quartz/utils/utils.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <queue>
#include <unordered_set>

namespace quartz {
bool Verifier::verify_transformation_steps(Context *ctx,
                                           const std::string &steps_file_prefix,
                                           bool verbose) {
  int step_count = 0;
  std::ifstream fin(steps_file_prefix + ".txt");
  assert(fin.is_open());
  fin >> step_count;
  fin.close();
  std::vector<std::unique_ptr<CircuitSeq>> circuits;
  circuits.reserve(step_count + 1);
  for (int i = 0; i <= step_count; i++) {
    circuits.push_back(CircuitSeq::from_qasm_file(
        ctx, steps_file_prefix + std::to_string(i) + ".qasm"));
    if (verbose) {
      std::cout << "circuit " << i << std::endl;
      std::cout << circuits[i]->to_string(/*line_number=*/true) << std::endl;
    }
    if (i > 506) {
      if (verbose) {
        std::cout << "Verifying circuit " << i - 1 << " -> circuit " << i
                  << std::endl;
      }
      if (!equivalent(ctx, circuits[i - 1].get(), circuits[i].get(), verbose)) {
        return false;
      }
    }
  }
  if (verbose) {
    std::cout << step_count << " transformations verified." << std::endl;
  }
  return true;
}

bool Verifier::equivalent(Context *ctx, const CircuitSeq *circuit1,
                          const CircuitSeq *circuit2, bool verbose) {
  const int num_qubits = circuit1->get_num_qubits();
  std::unique_ptr<CircuitSeq> c1, c2;
  if (!extract_difference(ctx, circuit1, circuit2, c1, c2, verbose)) {
    // Already printed out the reason there when verbose is true
    return false;
  }
  if (c1->get_num_gates() == 0 && c2->get_num_gates() == 0) {
    // The two circuits are equivalent.
    if (verbose) {
      std::cout << "Equivalent: same circuit." << std::endl;
    }
    return true;
  }
  // Remove qubits with no gates to avoid creating 2^num_qubits variables in the
  // verifier.
  std::vector<int> qubit_permutation(num_qubits, -1);
  int remaining_qubits = 0;
  for (int i = 0; i < num_qubits; i++) {
    if (c1->outputs[i] != c1->wires[i].get() ||
        c2->outputs[i] != c2->wires[i].get()) {
      // qubit used in at least one of the two circuits
      qubit_permutation[i] = remaining_qubits++;
    }
  }
  if (remaining_qubits < num_qubits) {
    if (verbose) {
      std::cout << "Reducing the number of qubits from " << num_qubits << " to "
                << remaining_qubits << std::endl;
    }
    c1 = c1->get_permuted_seq(qubit_permutation, {}, ctx, remaining_qubits);
    c2 = c2->get_permuted_seq(qubit_permutation, {}, ctx, remaining_qubits);
  }

  if (verbose) {
    std::cout << "Checking Verifier::equivalent() on:" << std::endl;
    std::cout << c1->to_string(/*line_number=*/true, ctx) << std::endl;
    std::cout << c2->to_string(/*line_number=*/true, ctx) << std::endl;
    c1->to_qasm_file(ctx, kQuartzRootPath.string() + "/c1.qasm");
    c2->to_qasm_file(ctx, kQuartzRootPath.string() + "/c2.qasm");
  }

  Dataset dataset;
  bool ret = dataset.insert(ctx, std::move(c1));
  assert(ret);
  ret = dataset.insert_to_nearby_set_if_exists(ctx, std::move(c2));
  if (ret) {
    // no nearby set
    if (verbose) {
      std::cout << "Not equivalent: different hash values." << std::endl;
    }
    return false;  // hash value not equal or adjacent
  }
  ret = dataset.save_json(ctx,
                          kQuartzRootPath.string() + "/tmp_before_verify.json");
  assert(ret);
  std::string command_string =
      std::string("python ") + kQuartzRootPath.string() +
      "/src/python/verifier/verify_equivalences.py " +
      kQuartzRootPath.string() + "/tmp_before_verify.json " +
      kQuartzRootPath.string() + "/tmp_after_verify.json" +
      (verbose ? " True True" : "");
  system(command_string.c_str());
  EquivalenceSet equiv_set;
  ret = equiv_set.load_json(ctx,
                            kQuartzRootPath.string() + "/tmp_after_verify.json",
                            /*from_verifier=*/true, nullptr);
  assert(ret);
  if (equiv_set.num_equivalence_classes() == 1) {
    return true;  // equivalent
  } else {
    if (verbose) {
      std::cout << "Not equivalent: Z3 cannot prove they are equivalent."
                << std::endl;
    }
    return false;  // not equivalent
  }
}

bool Verifier::extract_difference(Context *ctx, const CircuitSeq *circuit1,
                                  const CircuitSeq *circuit2,
                                  std::unique_ptr<CircuitSeq> &output_circuit1,
                                  std::unique_ptr<CircuitSeq> &output_circuit2,
                                  bool verbose) {
  if (output_circuit1 || output_circuit2) {
    if (verbose) {
      std::cout << "Error: output_circuit1 and output_circuit2 should be both "
                   "set to nullptr when calling extract_difference()."
                << std::endl;
    }
    return false;
  }
  if (circuit1->get_num_qubits() != circuit2->get_num_qubits()) {
    if (verbose) {
      std::cout << "Not equivalent: different numbers of qubits." << std::endl;
    }
    return false;
  }
  const int num_qubits = circuit1->get_num_qubits();

  // Mappings for topological sort
  std::unordered_map<CircuitWire *, CircuitWire *> wires_mapping;
  std::queue<CircuitWire *> wires_to_search;
  std::unordered_map<CircuitGate *, int> gate_remaining_in_degree;
  for (int i = 0; i < num_qubits; i++) {
    wires_mapping[circuit1->wires[i].get()] = circuit2->wires[i].get();
    wires_to_search.push(circuit1->wires[i].get());
  }
  // Try to remove common first gates and record the frontier
  std::vector<bool> qubit_blocked(num_qubits, false);
  std::vector<CircuitWire *> frontier1(num_qubits);
  std::vector<CircuitWire *> frontier2(num_qubits);
  for (int i = 0; i < num_qubits; i++) {
    frontier1[i] = circuit1->wires[i].get();
    frontier2[i] = circuit2->wires[i].get();
  }
  // (Partial) topological sort on circuit1
  while (!wires_to_search.empty()) {
    CircuitWire *wire1 = wires_to_search.front();
    assert(wires_mapping.count(wire1) > 0);
    auto wire2 = wires_mapping[wire1];
    wires_to_search.pop();
    assert(wire1->output_gates.size() <= 1);
    assert(wire2->output_gates.size() <= 1);
    if (wire1->output_gates.empty() || wire2->output_gates.empty() ||
        std::any_of(wire1->output_gates[0]->output_wires.begin(),
                    wire1->output_gates[0]->output_wires.end(),
                    [&qubit_blocked](CircuitWire *output_wire) {
                      return qubit_blocked[output_wire->index];
                    })) {
      // Block qubits of potential unmatched gates.
      for (auto &gate1 : wire1->output_gates) {
        if (verbose) {
          std::cout << "gate1 blocked: " << gate1->to_string() << std::endl;
        }
        // Use a for loop because |wire1->output_gates| can be empty.
        for (auto &input_wire : gate1->input_wires) {
          qubit_blocked[input_wire->index] = true;
        }
      }
      for (auto &gate2 : wire2->output_gates) {
        if (verbose) {
          std::cout << "gate2 blocked: " << gate2->to_string() << std::endl;
        }
        // Use a for loop because |wire2->output_gates| can be empty.
        for (auto &output_wire : gate2->output_wires) {
          qubit_blocked[output_wire->index] = true;
        }
      }
      continue;
    }

    auto gate1 = wire1->output_gates[0];
    auto gate2 = wire2->output_gates[0];
    if (gate_remaining_in_degree.count(gate1) == 0) {
      // A new gate
      gate_remaining_in_degree[gate1] = gate1->gate->get_num_qubits();
    }
    if (!--gate_remaining_in_degree[gate1]) {
      // Check if this gate is the same as the other gate and continue
      // the topological sort (wires_to_search is updated in
      // CircuitGate::equivalent())
      if (!CircuitGate::equivalent(gate1, gate2, wires_mapping,
                                   /*update_mapping=*/true, &wires_to_search)) {
        if (verbose) {
          std::cout << "gate1 and gate2 not equivalent: " << gate1->to_string()
                    << " " << gate2->to_string() << std::endl;
        }
        // If not matched, block each qubit of gate1.
        // Note that we should not block qubits of gate2 because it's not
        // on the frontier.
        for (auto &input_wire : gate1->input_wires) {
          if (input_wire->is_qubit()) {
            qubit_blocked[input_wire->index] = true;
          }
        }
      } else {
        for (auto &output_wire : gate1->output_wires) {
          frontier1[output_wire->index] = output_wire;
        }
        for (auto &output_wire : gate2->output_wires) {
          frontier2[output_wire->index] = output_wire;
        }
      }
    }
  }

  output_circuit1 = circuit1->get_suffix_seq(frontier1, ctx);
  output_circuit2 = circuit2->get_suffix_seq(frontier2, ctx);
  if (verbose) {
    std::cout << "Removed "
              << circuit1->get_num_gates() - output_circuit1->get_num_gates()
              << " prefix gates from circuit1 and "
              << circuit2->get_num_gates() - output_circuit2->get_num_gates()
              << " prefix gates from circuit2." << std::endl;
    std::cout << "Frontier of circuit1:" << std::endl;
    for (auto &wire : frontier1) {
      std::cout << "Q" << wire->index << ": ";
      if (wire->output_gates.size() == 1) {
        std::cout << wire->output_gates[0]->to_string() << std::endl;
      } else {
        std::cout << "(end)" << std::endl;
      }
    }
    std::cout << "Frontier of circuit2:" << std::endl;
    for (auto &wire : frontier2) {
      std::cout << "Q" << wire->index << ": ";
      if (wire->output_gates.size() == 1) {
        std::cout << wire->output_gates[0]->to_string() << std::endl;
      } else {
        std::cout << "(end)" << std::endl;
      }
    }
  }
  // We should have removed the same number of gates
  assert(circuit1->get_num_gates() - c1->get_num_gates() ==
         circuit2->get_num_gates() - c2->get_num_gates());

  // Remove common last gates
  while (true) {
    bool removed_anything = false;
    for (int i = 0; i < num_qubits; i++) {
      if (output_circuit1->outputs[i]->input_gates.empty() ||
          output_circuit2->outputs[i]->input_gates.empty()) {
        // At least of the two circuits does not have a gate at qubit i
        continue;
      }
      assert(c1->outputs[i]->input_gates.size() == 1);
      assert(c2->outputs[i]->input_gates.size() == 1);
      auto gate1 = output_circuit1->outputs[i]->input_gates[0];
      auto gate2 = output_circuit2->outputs[i]->input_gates[0];
      if (gate1->gate != gate2->gate ||
          gate1->input_wires.size() != gate2->input_wires.size() ||
          !output_circuit1->is_one_of_last_gates(gate1) ||
          !output_circuit2->is_one_of_last_gates(gate2)) {
        continue;
      }
      bool matched = true;
      for (int j = 0; j < (int)gate1->input_wires.size(); j++) {
        if (gate1->input_wires[j]->is_qubit() !=
            gate2->input_wires[j]->is_qubit()) {
          matched = false;
          break;
        }
        if (gate1->input_wires[j]->is_qubit()) {
          if (gate1->input_wires[j]->index != gate2->input_wires[j]->index) {
            matched = false;
            break;
          }
        } else {
          // parameters should not be mapped
          if (gate1->input_wires[j] != gate2->input_wires[j]) {
            matched = false;
            break;
          }
        }
      }
      if (matched) {
        output_circuit1->remove_gate_near_end(gate1);
        output_circuit2->remove_gate_near_end(gate2);
        removed_anything = true;
      }
    }
    if (!removed_anything) {
      break;
    }
  }
  return true;
}

std::string Verifier::difference_str(Context *ctx, const CircuitSeq *circuit1,
                                     const CircuitSeq *circuit2,
                                     int columns_before_midline,
                                     int param_precision) {
  std::unique_ptr<CircuitSeq> c1, c2;
  if (!extract_difference(ctx, circuit1, circuit2, c1, c2, /*verbose=*/false)) {
    return "(Error during extract_difference().)";
  }
  std::string result;
  int num_gates = std::max(c1->get_num_gates(), c2->get_num_gates());
  if (num_gates == 0) {
    return "(same)";
  }
  for (int i = 0; i < num_gates; i++) {
    std::string line;
    if (i < c1->get_num_gates()) {
      line = c1->gates[i]->to_qasm_style_string(ctx, param_precision);
      line.pop_back();  // remove '\n'
    }
    if (line.size() < columns_before_midline) {
      line.append(columns_before_midline - line.size(), ' ');
    }
    line += "| ";
    if (i < c2->get_num_gates()) {
      line += c2->gates[i]->to_qasm_style_string(ctx, param_precision);
      line.pop_back();  // remove '\n'
    }
    result += line + '\n';
  }
  return result;
}

std::string Verifier::difference_str(const Graph *circuit1,
                                     const Graph *circuit2,
                                     int columns_before_midline,
                                     int param_precision) {
  if (circuit1->context != circuit2->context) {
    return "(Error: different context.)";
  }
  auto c1 = circuit1->to_circuit_sequence();
  auto c2 = circuit2->to_circuit_sequence();
  return difference_str(circuit1->context, c1.get(), c2.get(),
                        columns_before_midline, param_precision);
}

bool Verifier::equivalent_on_the_fly(Context *ctx, CircuitSeq *circuit1,
                                     CircuitSeq *circuit2) {
  // Disable the verifier.
  return false;
  // Aggressively assume circuits with the same hash values are
  // equivalent.
  return circuit1->hash(ctx) == circuit2->hash(ctx);
}

bool Verifier::redundant(Context *ctx, CircuitSeq *dag) {
  // RepGen.
  // Check if |circuitseq| is a canonical sequence.
  if (!dag->is_canonical_representation()) {
    return true;
  }
  if (kUseRowRepresentationToCompare) {
    // We have already known that DropLast(circuitseq) is a representative.
    // Check if canonicalize(DropFirst(circuitseq)) is a representative.
    auto first_gates = dag->first_quantum_gate_positions();
    for (auto &gate_position : first_gates) {
      auto dropfirst = std::make_unique<CircuitSeq>(*dag);
      dropfirst->remove_gate(gate_position);
      CircuitSeqHashType hash_value = dropfirst->hash(ctx);
      // XXX: here we treat any CircuitSeq with hash values differ no more than
      // 1 with any representative as equivalent.
      bool found = false;
      for (const auto &hash_value_offset : {0, 1, -1}) {
        CircuitSeq *rep = nullptr;
        if (ctx->get_possible_representative(hash_value + hash_value_offset,
                                             rep)) {
          assert(rep);
          if (!dropfirst->topologically_equivalent(*rep)) {
            // |dropfirst| already exists and is not the
            // representative. So the whole |circuitseq| is redundant.
            return true;
          } else {
            // |dropfirst| already exists and is the representative.
            found = true;
            break;
          }
        }
      }
      if (!found) {
        // |dropfirst| is not found and therefore is not a representative.
        return true;
      }
    }
    // All |dropfirst|s are representatives.
    return false;
  } else {
    // We have already known that DropLast(circuitseq) is a representative.
    // Check if canonicalize(DropFirst(circuitseq)) is a representative.
    auto dropfirst = std::make_unique<CircuitSeq>(*dag);
    dropfirst->remove_first_quantum_gate();
    CircuitSeqHashType hash_value = dropfirst->hash(ctx);
    // XXX: here we treat any CircuitSeq with hash values differ no more than 1
    // with any representative as equivalent.
    for (const auto &hash_value_offset : {0, 1, -1}) {
      CircuitSeq *rep = nullptr;
      if (ctx->get_possible_representative(hash_value + hash_value_offset,
                                           rep)) {
        assert(rep);
        if (!dropfirst->fully_equivalent(*rep)) {
          // |dropfirst| already exists and is not the
          // representative. So the whole |circuitseq| is redundant.
          return true;
        } else {
          // |dropfirst| already exists and is the representative.
          return false;
        }
      }
    }
    // |dropfirst| is not found and therefore is not a representative.
    return true;
  }
}

bool Verifier::redundant(Context *ctx, const EquivalenceSet *eqs,
                         CircuitSeq *dag) {
  // RepGen.
  // Check if |circuitseq| is a canonical sequence.
  if (!dag->is_canonical_representation()) {
    return true;
  }
  if (kUseRowRepresentationToCompare) {
    // We have already known that DropLast(circuitseq) is a representative.
    // Check if canonicalize(DropFirst(circuitseq)) is a representative.
    auto first_gates = dag->first_quantum_gate_positions();
    for (auto &gate_position : first_gates) {
      auto dropfirst = std::make_unique<CircuitSeq>(*dag);
      dropfirst->remove_gate(gate_position);
      CircuitSeqHashType hash_value = dropfirst->hash(ctx);
      auto possible_classes = eqs->get_possible_classes(hash_value);
      for (const auto &other_hash : dropfirst->other_hash_values()) {
        auto more_possible_classes = eqs->get_possible_classes(other_hash);
        possible_classes.insert(possible_classes.end(),
                                more_possible_classes.begin(),
                                more_possible_classes.end());
      }
      std::sort(possible_classes.begin(), possible_classes.end());
      auto last = std::unique(possible_classes.begin(), possible_classes.end());
      possible_classes.erase(last, possible_classes.end());
      bool found = false;
      for (const auto &equiv_class : possible_classes) {
        if (equiv_class->contains(*dropfirst)) {
          if (!dropfirst->topologically_equivalent(
                  *equiv_class->get_representative())) {
            // |dropfirst| already exists and is not the
            // representative. So the whole |circuitseq| is redundant.
            return true;
          } else {
            // |dropfirst| already exists and is the representative.
            found = true;
            break;
          }
        }
      }
      if (!found) {
        // |dropfirst| is not found and therefore is not a representative.
        return true;
      }
    }
    // All |dropfirst|s are representatives.
    return false;
  } else {
    // We have already known that DropLast(circuitseq) is a representative.
    // Check if canonicalize(DropFirst(circuitseq)) is a representative.
    auto dropfirst = std::make_unique<CircuitSeq>(*dag);
    dropfirst->remove_first_quantum_gate();
    CircuitSeqHashType hash_value = dropfirst->hash(ctx);
    auto possible_classes = eqs->get_possible_classes(hash_value);
    for (const auto &other_hash : dropfirst->other_hash_values()) {
      auto more_possible_classes = eqs->get_possible_classes(other_hash);
      possible_classes.insert(possible_classes.end(),
                              more_possible_classes.begin(),
                              more_possible_classes.end());
    }
    std::sort(possible_classes.begin(), possible_classes.end());
    auto last = std::unique(possible_classes.begin(), possible_classes.end());
    possible_classes.erase(last, possible_classes.end());
    for (const auto &equiv_class : possible_classes) {
      if (equiv_class->contains(*dropfirst)) {
        if (!dropfirst->fully_equivalent(*equiv_class->get_representative())) {
          // |dropfirst| already exists and is not the
          // representative. So the whole |circuitseq| is redundant.
          return true;
        } else {
          // |dropfirst| already exists and is the representative.
          return false;
        }
      }
    }
    // |dropfirst| is not found and therefore is not a representative.
    return true;
  }
}

}  // namespace quartz
