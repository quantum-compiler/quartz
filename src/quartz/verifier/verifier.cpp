#include "verifier.h"

#include "quartz/utils/utils.h"

#include <algorithm>
#include <cassert>

namespace quartz {
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
  assert(!kUseRowRepresentationToCompare);  // not implemented otherwise
  // We have already known that DropLast(circuitseq) is a representative.
  // Check if canonicalize(DropFirst(circuitseq)) is a representative.
  auto dropfirst = std::make_unique<CircuitSeq>(*dag);
  dropfirst->remove_first_quantum_gate();
  CircuitSeqHashType hash_value = dropfirst->hash(ctx);
  // XXX: here we treat any CircuitSeq with hash values differ no more than 1
  // with any representative as equivalent.
  for (const auto &hash_value_offset : {0, 1, -1}) {
    CircuitSeq *rep = nullptr;
    if (ctx->get_possible_representative(hash_value + hash_value_offset, rep)) {
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
