#pragma once

#include "../context/context.h"
#include "../dataset/equivalence_set.h"
#include "quartz/circuitseq/circuitseq.h"

namespace quartz {
// Verify if two circuits are equivalent and other things about DAGs.
class Verifier {
 public:
  /**
   * Verify all transformation steps.
   * @param ctx The context to load the circuits from files.
   * @param steps_file_prefix The parameter |store_all_steps_file_prefix| when
   * calling Graph::optimize().
   * @param verbose Print logs to the screen.
   * @return True if we can verify all transformation steps.
   */
  bool verify_transformation_steps(Context *ctx,
                                   const std::string &steps_file_prefix,
                                   bool verbose = false);
  /**
   * Verify if two circuits are functionally equivalent. They are expected
   * to differ by only one small circuit transformation.
   * @param ctx The context for both circuits.
   * @param verbose Print logs to the screen.
   * @return True if we can prove that the two circuits are functionally
   * equivalent.
   */
  bool equivalent(Context *ctx, const CircuitSeq *circuit1,
                  const CircuitSeq *circuit2, bool verbose = false);
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

}  // namespace quartz
