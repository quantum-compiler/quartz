#include "verifier.h"

bool Verifier::equivalent_on_the_fly(Context *ctx,
                                     DAG *circuit1,
                                     DAG *circuit2) {
  // Disable the verifier.
  return false;
  // Aggressively assume circuits with the same hash values are equivalent.
  return circuit1->hash(ctx) == circuit2->hash(ctx);
}

bool Verifier::redundant(Context *ctx, DAG *dag) {
  // Check if any suffix already exists.
  auto subgraph = std::make_unique<DAG>(*dag);
  while (subgraph->get_num_gates() > 0) {
    subgraph->remove_gate(subgraph->edges[0].get());
    subgraph->hash(ctx);
    DAG *rep = ctx->get_representative(subgraph.get());
    if (rep && !subgraph->fully_equivalent(*rep)) {
      // |subgraph| already exists and is not the representative.
      return true;
    }
  }
  return false;
}
