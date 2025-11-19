#pragma once

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/dataset/equivalence_set.h"

namespace quartz {
class Graph;
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
  static bool verify_transformation_steps(Context *ctx,
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
  static bool equivalent(Context *ctx, const CircuitSeq *circuit1,
                         const CircuitSeq *circuit2, bool verbose = false);
  /**
   * A helper method to extract the difference between two circuits, used
   * by the equivalent() function above. The two circuits are expected
   * to differ by only one small circuit transformation.
   * @param ctx The context for both circuits.
   * @param circuit1 The first input circuit.
   * @param circuit2 The second input circuit.
   * @param output_circuit1 Return the different part for circuit1.
   * Should pass in a variable initialized to nullptr to store the return value.
   * @param output_circuit2 Return the different part for circuit2.
   * Should pass in a variable initialized to nullptr to store the return value.
   * @param verbose Print logs to the screen.
   * @return True iff the function succeeded.
   */
  static bool extract_difference(Context *ctx, const CircuitSeq *circuit1,
                                 const CircuitSeq *circuit2,
                                 std::unique_ptr<CircuitSeq> &output_circuit1,
                                 std::unique_ptr<CircuitSeq> &output_circuit2,
                                 bool verbose = false);
  /**
   * Extract the difference between two circuits to a string.
   * @param ctx The context for both circuits.
   * @param circuit1 The first input circuit.
   * @param circuit2 The second input circuit.
   * @param columns_before_midline Print out a character '|' at this location
   * for each line to separate two circuits.
   * @param param_precision The parameter precision for the output.
   * @return A string side-by-side for the difference.
   */
  static std::string
  difference_str(Context *ctx, const CircuitSeq *circuit1,
                 const CircuitSeq *circuit2, int columns_before_midline = 40,
                 int param_precision = kDefaultQASMParamPrecision);
  static std::string
  difference_str(const Graph *circuit1, const Graph *circuit2,
                 int columns_before_midline = 40,
                 int param_precision = kDefaultQASMParamPrecision);
  // On-the-fly equivalence checking while generating circuits
  static bool equivalent_on_the_fly(Context *ctx, CircuitSeq *circuit1,
                                    CircuitSeq *circuit2);

  // Check if the CircuitSeq is redundant (equivalence opportunities have
  // already been covered by smaller circuits). This function assumes that two
  // DAGs are equivalent iff they share the same hash value.
  static bool redundant(Context *ctx, CircuitSeq *dag);

  // Check if the CircuitSeq is redundant (equivalence opportunities have
  // already been covered by smaller circuits).
  static bool redundant(Context *ctx, const EquivalenceSet *eqs,
                        CircuitSeq *dag);
};

}  // namespace quartz
