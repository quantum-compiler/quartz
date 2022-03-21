#pragma once

#include "../gate/gate_utils.h"
#include "../math/vector.h"
#include "../utils/utils.h"

#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <set>

namespace quartz {
	class DAG;

	class Context {
	public:
		explicit Context(const std::vector< GateType > &supported_gates);
        Context(const std::vector< GateType > &supported_gates, const int num_qubits, const int num_params);
		Gate *get_gate(GateType tp);
		[[nodiscard]] const std::vector< GateType > &
		get_supported_gates() const;
		[[nodiscard]] const std::vector< GateType > &
		get_supported_parameter_gates() const;
		[[nodiscard]] const std::vector< GateType > &
		get_supported_quantum_gates() const;
		// Two deterministic (random) distributions for each number of qubits.
        const Vector& get_and_gen_input_dis(int num_qubits);
        const Vector& get_and_gen_hashing_dis(int num_qubits);
        std::vector< ParamType > get_and_gen_parameters(int num_params);
		const Vector &get_generated_input_dis(int num_qubits) const;
		const Vector &get_generated_hashing_dis(int num_qubits) const;
		std::vector< ParamType > get_generated_parameters(int num_params) const;
		std::vector< ParamType > get_all_generated_parameters() const;
        // generate at once
        void generate_input_dis(const int max_num_qubits);
        void generate_hashing_dis(const int max_num_qubits);
        void generate_parameters(const int max_num_params);
		size_t next_global_unique_id();

		// A hacky function: set a generated parameter.
		void set_generated_parameter(int id, ParamType param);

		// This function assumes that two DAGs are equivalent iff they share the
		// same hash value.
		DAG *get_possible_representative(DAG *dag);

		// This function assumes that two DAGs are equivalent iff they share the
		// same hash value.
		void set_representative(std::unique_ptr< DAG > dag);
		void clear_representatives();

		// This function generates a deterministic series of random numbers
		// ranging [0, 1].
		double random_number();

	private:
		bool insert_gate(GateType tp);
		size_t global_unique_id;
		std::unordered_map< GateType, std::unique_ptr< Gate > > gates_;
		std::vector< GateType > supported_gates_;
		std::vector< GateType > supported_parameter_gates_;
		std::vector< GateType > supported_quantum_gates_;
		std::vector< Vector > random_input_distribution_;
		std::vector< Vector > random_hashing_distribution_;
		std::vector< ParamType > random_parameters_;

		// A vector to store the representative DAGs.
		std::vector< std::unique_ptr< DAG > > representative_dags_;
		std::unordered_map< DAGHashType, DAG * > representatives_;
        // Standard mersenne_twister_engine seeded with 0
        std::mt19937 gen{0};
	};

	Context union_contexts(Context *ctx_0, Context *ctx_1);

} // namespace quartz
