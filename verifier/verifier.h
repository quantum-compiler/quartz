#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

// Verify if two circuits are equivalent and other things about DAGs.
class Verifier {
 public:
  bool equivalent(Context *ctx, DAG *circuit1, DAG *circuit2);
  // On-the-fly equivalence checking while generating circuits
  bool equivalent_on_the_fly(Context *ctx, DAG *circuit1, DAG *circuit2);

  // Check if the DAG is redundant (equivalence opportunities have already
  // been covered by smaller circuits).
  bool redundant(Context *ctx, DAG *dag);
};
