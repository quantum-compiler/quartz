#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

// Verify if two circuits are equivalent.
class Verifier {
 public:
  bool equivalent(Context *ctx, DAG *circuit1, DAG *circuit2);
  // On-the-fly equivalence checking while generating circuits
  bool equivalent_on_the_fly(Context *ctx, DAG *circuit1, DAG *circuit2);
};
