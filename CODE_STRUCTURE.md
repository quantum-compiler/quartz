# Quartz Code Structure

This file documents the organization of our repo.

In `src/python/verifier/gates.py`
* `add` and `neg`: methods for computing the trigonometric functions for parameter expressions
* Others: methods for constructing a matrix representation for different quantum gates

In `src/quartz/gate/all_gates.h`
* Similar to `src/python/verifier/gates.py` but in C++

In `src/python/verifier/verifier.py`
* `search_phase_factor_to_check_equivalence`: search the phase factor between two quantum circuits to be verified
* `equivalent`: use the Z3 SMT solver to examine the equivalence of two quantum circuits, invoking `search_phase_factor_to_check_equivalence`
* `find_equivalences`: group circuits into ECCs (invoking `equivalent`) and return an ECC set

In `src/quartz/context/context.h`
* `class Context`: define the execution context for a quantum device, including the set of gates supported by the processor

In `src/quartz/context/rule_parser.h`
* `class RuleParser`: define the rules to write 3-qubit gates in each of the gate sets

In `src/quartz/parser/qasm_parser.h`
* `class QASMParser`: parse an input QASM file to Quartz's DAG representation

In `src/dag/dag.h`
* `class DAG`: a circuit represented in a DAG with all gates stored in an `std::vector` (which is a sequence representation)

In `src/dag/dagnode.h`
* `class DAGNode`: a node in DAG corresponds to a wire in the circuit

In `src/dag/daghyperedge.h`
* `class DAGHyperEdge`: a hyperedge in DAG corresponds to a gate (or parameter expression) in the circuit

In `src/dataset/dataset.h`
* `class Dataset`: a collection of circuits grouped by fingerprints

In `src/dataset/equivalence_set.h`
* `class EquivalenceClass`: an ECC
* `class EquivalenceSet`: an ECC set

In `src/generator/generator.h`
* `class Generator`: the circuit generator
* `Generator::generate`: generate circuits for an unverified ECC set (then use `src/python/verify_equivalences.py` to get the ECC set)

In `src/math/matrix.h`
* `class Matrix`: a complex square matrix

In `src/math/vector.h`
* `class Vector`: a complex vector

In `src/verifier/verifier.h`
* `Verifier::redundant`: check if the circuit generated is redundant, i.e., having some slices not in the representative set

In `src/tasograph/tasograph.h`
* `class Graph`: the circuit to be optimized
* `Graph::optimize`: use the search algorithm to optimize the circuit
* `Graph::context_shift`: shift the context of the circuit, e.g., changing the gate set

In `src/tasograph/substitution.h`
* `class GraphXfer`: a circuit transformation

In `src/test/gen_ecc_set.cpp`
* `gen_ecc_set`: a function to generate ECC sets with given gate set and hyperparameters
