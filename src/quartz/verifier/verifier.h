#pragma once

#include "../context/context.h"
#include "../dataset/equivalence_set.h"
#include "quartz/circuitseq/circuitseq.h"

namespace quartz {
// Verify if two circuits are equivalent and other things about DAGs.
class Verifier {
public:
  bool equivalent(Context *ctx, CircuitSeq *circuit1, CircuitSeq *circuit2);
  // On-the-fly equivalence checking while generating circuits
  bool equivalent_on_the_fly(Context *ctx, CircuitSeq *circuit1,
                             CircuitSeq *circuit2);

  // Check if the CircuitSeq is redundant (equivalence opportunities have
  // already been covered by smaller circuits). This function assumes that two
  // DAGs are equivalent iff they share the same hash value.
  bool redundant(Context *ctx, CircuitSeq *dag);

  // Check if the CircuitSeq is redundant (equivalence opportunities have
  // already been covered by smaller circuits).
  bool redundant(Context *ctx, const EquivalenceSet *eqs, CircuitSeq *dag);
};

} // namespace quartz
