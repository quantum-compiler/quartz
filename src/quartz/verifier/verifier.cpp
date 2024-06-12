#include "verifier.h"

#include "quartz/dataset/dataset.h"
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
    if (i > 0) {
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
  std::unordered_set<CircuitGate *> leftover_gates_start1;
  std::unordered_set<CircuitGate *> leftover_gates_start2;
  // (Partial) topological sort on circuit1
  while (!wires_to_search.empty()) {
    auto wire1 = wires_to_search.front();
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
      // Block qubits of potential unmatched gates
      for (auto &gate : wire1->output_gates) {
        for (auto &output_wire : gate->output_wires) {
          qubit_blocked[output_wire->index] = true;
        }
        // Use std::unordered_set to deduplicate, similarly hereinafter
        leftover_gates_start1.insert(gate);
      }
      for (auto &gate : wire2->output_gates) {
        for (auto &output_wire : gate->output_wires) {
          qubit_blocked[output_wire->index] = true;
        }
        leftover_gates_start2.insert(gate);
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
        // If not matched, block each qubit of both gates
        for (auto &input_wire : gate1->input_wires) {
          if (input_wire->is_qubit()) {
            qubit_blocked[input_wire->index] = true;
            // Note that this input wire might be not in the search queue now.
            // We need to manually add the gate in circuit2 to the frontier.
            assert(wires_mapping.count(input_wire) > 0);
            for (auto &gate : wires_mapping[input_wire]->output_gates) {
              leftover_gates_start2.insert(gate);
            }
          }
        }
        leftover_gates_start1.insert(gate1);
        for (auto &output_wire : gate2->output_wires) {
          qubit_blocked[output_wire->index] = true;
        }
        leftover_gates_start2.insert(gate2);
      }
    }
  }

  if (leftover_gates_start1.empty() && leftover_gates_start2.empty()) {
    // The two circuits are equivalent.
    if (verbose) {
      std::cout << "Equivalent: same circuit." << std::endl;
    }
    return true;
  }

  auto c1 = circuit1->get_suffix_seq(leftover_gates_start1, ctx);
  auto c2 = circuit2->get_suffix_seq(leftover_gates_start2, ctx);
  // We should have removed the same number of gates
  assert(circuit1->get_num_gates() - c1->get_num_gates() ==
         circuit2->get_num_gates() - c2->get_num_gates());

  // Remove common last gates
  while (true) {
    bool removed_anything = false;
    for (int i = 0; i < num_qubits; i++) {
      if (c1->outputs[i]->input_gates.empty() ||
          c2->outputs[i]->input_gates.empty()) {
        // At least of the two circuits does not have a gate at qubit i
        continue;
      }
      assert(c1->outputs[i]->input_gates.size() == 1);
      assert(c2->outputs[i]->input_gates.size() == 1);
      auto gate1 = c1->outputs[i]->input_gates[0];
      auto gate2 = c2->outputs[i]->input_gates[0];
      if (gate1->gate != gate2->gate ||
          gate1->input_wires.size() != gate2->input_wires.size() ||
          !c1->is_one_of_last_gates(gate1) ||
          !c2->is_one_of_last_gates(gate2)) {
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
        c1->remove_gate_near_end(gate1);
        c2->remove_gate_near_end(gate2);
        removed_anything = true;
      }
    }
    if (!removed_anything) {
      break;
    }
  }

  if (verbose) {
    std::cout << "Checking Verifier::equivalent() on:" << std::endl;
    std::cout << c1->to_string(/*line_number=*/true) << std::endl;
    std::cout << c2->to_string(/*line_number=*/true) << std::endl;
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
      kQuartzRootPath.string() + "/tmp_after_verify.json";
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
