#include "verifier.h"

bool Verifier::equivalent_on_the_fly(Context *ctx,
                                     DAG *circuit1,
                                     DAG *circuit2) {
  // Disable the verifier.
  return false;
  // Aggressively assume circuits with the same hash values are equivalent.
  return circuit1->hash(ctx) == circuit2->hash(ctx);
}
