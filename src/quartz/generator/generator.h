#pragma once

#include "../context/context.h"
#include "../dag/dag.h"
#include "../dataset/dataset.h"
#include "../dataset/equivalence_set.h"
#include "../verifier/verifier.h"

#include <unordered_set>

namespace quartz {
	class Generator {
	public:
		explicit Generator(Context *ctx) : context(ctx) {}

		// Use DFS to generate all equivalent DAGs with |num_qubits| qubits,
		// <= |max_num_input_parameters| input parameters,
		// and <= |max_num_quantum_gates| gates.
		// If |restrict_search_space| is false, we search for all possible DAGs
		// with no unused internal parameters.
		// If |restrict_search_space| is true, we only search for DAGs which:
		//   - Use qubits in an increasing order;
		//   - Use input parameters in an increasing order;
		//   - When a gate uses more than one fresh new qubits or fresh new
		//   input
		//     parameters, restrict the order (for example, if a CX gate uses
		//     two fresh new qubits, the control qubit must have the smaller
		//     index).
        // If |unique_parameters| is true, we only search for DAGs that use
        // each input parameters only once (note: use a doubled parameter, i.e.,
        // Rx(2theta) is considered using the parameter theta once).
		void generate_dfs(int num_qubits, int max_num_input_parameters,
		                  int max_num_quantum_gates, int max_num_param_gates,
		                  Dataset &dataset, bool restrict_search_space,
                          bool unique_parameters);

		// Use BFS to generate all equivalent DAGs with |num_qubits| qubits,
		// |num_input_parameters| input parameters (probably with some unused),
		// and <= |max_num_quantum_gates| gates.
		// If |unique_parameters| is true, we only search for DAGs that use
		// each input parameters only once (note: use a doubled parameter, i.e.,
		// Rx(2theta) is considered using the parameter theta once).
		void generate(int num_qubits, int num_input_parameters,
		              int max_num_quantum_gates, int max_num_param_gates,
		              Dataset *dataset, bool verify_equivalences,
		              EquivalenceSet *equiv_set, bool unique_parameters,
		              bool verbose = false);

	private:
		void dfs(int gate_idx, int max_num_gates, int max_remaining_param_gates,
		         DAG *dag, std::vector< int > &used_parameters,
		         Dataset &dataset, bool restrict_search_space,
                 bool unique_parameters);

		// |dags[i]| is the DAGs with |i| gates.
		void bfs(const std::vector< std::vector< DAG * > > &dags,
		         int max_num_param_gates, Dataset &dataset,
		         std::vector< DAG * > *new_representatives,
		         bool verify_equivalences, const EquivalenceSet *equiv_set,
                 bool unique_parameters);

		void dfs_parameter_gates(std::unique_ptr< DAG > dag,
		                         int remaining_gates, int max_unused_params,
		                         int current_unused_params,
		                         std::vector< int > &params_used_times,
		                         std::vector< std::unique_ptr< DAG > > &result);

		Context *context;
		Verifier verifier_;
	};

} // namespace quartz