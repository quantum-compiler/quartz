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
		auto subgraph = std::make_unique< DAG >(*dag);
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
		// Representative pruning.
		// Check if any suffix already exists.
		auto subgraph = std::make_unique< DAG >(*dag);
		while (subgraph->get_num_gates() > 0) {
			if (subgraph->remove_first_quantum_gate() == 0) {
				// Already no quantum gates
				break;
			}
			if (subgraph->get_num_gates() == 0) {
				break;
			}
			DAGHashType hash_value = subgraph->hash(ctx);
			auto possible_classes = eqs->get_possible_classes(hash_value);
			for (const auto &other_hash : subgraph->other_hash_values()) {
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
				if (equiv_class->contains(*subgraph)) {
					if (!subgraph->fully_equivalent(
					        *equiv_class->get_representative())) {
						// |subgraph| already exists and is not the
						// representative. So the whole |dag| is redundant.
						return true;
					}
					else {
						// |subgraph| already exists and is the representative.
						// So we need to check other subgraphs.
						break;
					}
				}
			}
		}
		return false;
	}

} // namespace quartz
