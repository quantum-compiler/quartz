#include "verifier.h"

#include <algorithm>

namespace quartz {
bool Verifier::equivalent_on_the_fly(Context *ctx, DAG *circuit1,
                                     DAG *circuit2) {
  // Disable the verifier.
  return false;
  // Aggressively assume circuits with the same hash values are
  // equivalent.
  return circuit1->hash(ctx) == circuit2->hash(ctx);
}

bool Verifier::redundant(Context *ctx, DAG *dag) {
  // Check if any suffix already exists.
  // This function assumes that two DAGs are equivalent iff they share the
  // same hash value.
  auto subgraph = std::make_unique<DAG>(*dag);
  while (subgraph->get_num_gates() > 0) {
    subgraph->remove_gate(subgraph->edges[0].get());
    if (subgraph->get_num_gates() == 0) {
      break;
    }
    subgraph->hash(ctx);
    const auto &rep = ctx->get_possible_representative(subgraph.get());
    if (rep && !subgraph->fully_equivalent(*rep)) {
      // |subgraph| already exists and is not the representative.
      return true;
    }
  }
  return false;
}

bool Verifier::redundant(Context *ctx, const EquivalenceSet *eqs,
                         DAG *dag) {
  // RepGen.
  // Check if |dag| is a canonical sequence.
  if (!dag->is_canonical_representation()) {
    return true;
  }
  // We have already known that DropLast(dag) is a representative.
  // Check if canonicalize(DropFirst(dag)) is a representative.
  auto dropfirst = std::make_unique<DAG>(*dag);
  dropfirst->remove_first_quantum_gate();
  dropfirst->to_canonical_representation();
  DAGHashType hash_value = dropfirst->hash(ctx);
  auto possible_classes = eqs->get_possible_classes(hash_value);
  for (const auto &other_hash : dropfirst->other_hash_values()) {
    auto more_possible_classes =
        eqs->get_possible_classes(other_hash);
    possible_classes.insert(possible_classes.end(),
                            more_possible_classes.begin(),
                            more_possible_classes.end());
  }
  std::sort(possible_classes.begin(), possible_classes.end());
  auto last =
      std::unique(possible_classes.begin(), possible_classes.end());
  possible_classes.erase(last, possible_classes.end());
  for (const auto &equiv_class : possible_classes) {
    if (equiv_class->contains(*dropfirst)) {
      if (!dropfirst->fully_equivalent(
          *equiv_class->get_representative())) {
        // |dropfirst| already exists and is not the
        // representative. So the whole |dag| is redundant.
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

} // namespace quartz
