#include "dag.h"
#include "../gate/gate.h"
#include "../context/context.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <unordered_set>
#include <utility>

namespace quartz {
	DAG::DAG(int num_qubits, int num_input_parameters)
	    : num_qubits(num_qubits), num_input_parameters(num_input_parameters),
	      hash_value_(0), hash_value_valid_(false) {
		nodes.reserve(num_qubits + num_input_parameters);
		outputs.reserve(num_qubits);
		parameters.reserve(num_input_parameters);
		// Initialize num_qubits qubits
		for (int i = 0; i < num_qubits; i++) {
			auto node = std::make_unique< DAGNode >();
			node->type = DAGNode::input_qubit;
			node->index = i;
			outputs.push_back(node.get());
			nodes.push_back(std::move(node));
		}
		// Initialize num_input_parameters parameters
		for (int i = 0; i < num_input_parameters; i++) {
			auto node = std::make_unique< DAGNode >();
			node->type = DAGNode::input_param;
			node->index = i;
			parameters.push_back(node.get());
			nodes.push_back(std::move(node));
		}
	}

	DAG::DAG(const DAG &other) { clone_from(other, {}, {}); }

	std::unique_ptr< DAG > DAG::clone() const {
		return std::make_unique< DAG >(*this);
	}

	bool DAG::fully_equivalent(const DAG &other) const {
		// Do not check the hash value because of floating point errors
		// and it is possible that one of the two DAGs may have not calculated
		// the hash value.
		if (num_qubits != other.num_qubits ||
		    num_input_parameters != other.num_input_parameters) {
			return false;
		}
		if (nodes.size() != other.nodes.size() ||
		    edges.size() != other.edges.size()) {
			return false;
		}
		std::unordered_map< DAGNode *, DAGNode * > nodes_mapping;
		for (int i = 0; i < (int)nodes.size(); i++) {
			nodes_mapping[other.nodes[i].get()] = nodes[i].get();
		}
		for (int i = 0; i < (int)edges.size(); i++) {
			if (edges[i]->gate->tp != other.edges[i]->gate->tp) {
				return false;
			}
			if (edges[i]->input_nodes.size() !=
			        other.edges[i]->input_nodes.size() ||
			    edges[i]->output_nodes.size() !=
			        other.edges[i]->output_nodes.size()) {
				return false;
			}
			for (int j = 0; j < (int)edges[i]->input_nodes.size(); j++) {
				if (nodes_mapping[other.edges[i]->input_nodes[j]] !=
				    edges[i]->input_nodes[j]) {
					return false;
				}
			}
			for (int j = 0; j < (int)edges[i]->output_nodes.size(); j++) {
				if (nodes_mapping[other.edges[i]->output_nodes[j]] !=
				    edges[i]->output_nodes[j]) {
					return false;
				}
			}
		}
		return true;
	}

	bool DAG::fully_equivalent(Context *ctx, DAG &other) {
		if (hash(ctx) != other.hash(ctx)) {
			return false;
		}
		return fully_equivalent(other);
	}

	bool DAG::less_than(const DAG &other) const {
		if (num_qubits != other.num_qubits) {
			return num_qubits < other.num_qubits;
		}
		if (get_num_gates() != other.get_num_gates()) {
			return get_num_gates() < other.get_num_gates();
		}
		if (get_num_input_parameters() != other.get_num_input_parameters()) {
			return get_num_input_parameters() <
			       other.get_num_input_parameters();
		}
		if (get_num_total_parameters() != other.get_num_total_parameters()) {
			// We want fewer quantum gates, i.e., more traditional parameters.
			return get_num_total_parameters() >
			       other.get_num_total_parameters();
		}
		for (int i = 0; i < (int)edges.size(); i++) {
			if (edges[i]->gate->tp != other.edges[i]->gate->tp) {
				return edges[i]->gate->tp < other.edges[i]->gate->tp;
			}
			assert(edges[i]->input_nodes.size() ==
			       other.edges[i]->input_nodes.size());
			assert(edges[i]->output_nodes.size() ==
			       other.edges[i]->output_nodes.size());
			for (int j = 0; j < (int)edges[i]->input_nodes.size(); j++) {
				if (edges[i]->input_nodes[j]->is_qubit() !=
				    other.edges[i]->input_nodes[j]->is_qubit()) {
					return edges[i]->input_nodes[j]->is_qubit();
				}
				if (edges[i]->input_nodes[j]->index !=
				    other.edges[i]->input_nodes[j]->index) {
					return edges[i]->input_nodes[j]->index <
					       other.edges[i]->input_nodes[j]->index;
				}
			}
			for (int j = 0; j < (int)edges[i]->output_nodes.size(); j++) {
				if (edges[i]->output_nodes[j]->is_qubit() !=
				    other.edges[i]->output_nodes[j]->is_qubit()) {
					return edges[i]->output_nodes[j]->is_qubit();
				}
				if (edges[i]->output_nodes[j]->index !=
				    other.edges[i]->output_nodes[j]->index) {
					return edges[i]->output_nodes[j]->index <
					       other.edges[i]->output_nodes[j]->index;
				}
			}
		}
		return false; // fully equivalent
	}

	bool DAG::add_gate(const std::vector< int > &qubit_indices,
	                   const std::vector< int > &parameter_indices, Gate *gate,
	                   int *output_para_index) {
		if (gate->get_num_qubits() != qubit_indices.size())
			return false;
		if (gate->get_num_parameters() != parameter_indices.size())
			return false;
		if (gate->is_parameter_gate() && output_para_index == nullptr)
			return false;
		// qubit indices must stay in range
		for (auto qubit_idx : qubit_indices)
			if ((qubit_idx < 0) || (qubit_idx >= get_num_qubits()))
				return false;
		// parameter indices must stay in range
		for (auto para_idx : parameter_indices)
			if ((para_idx < 0) || (para_idx >= parameters.size()))
				return false;
		auto edge = std::make_unique< DAGHyperEdge >();
		edge->gate = gate;
		for (auto qubit_idx : qubit_indices) {
			edge->input_nodes.push_back(outputs[qubit_idx]);
			outputs[qubit_idx]->output_edges.push_back(edge.get());
		}
		for (auto para_idx : parameter_indices) {
			edge->input_nodes.push_back(parameters[para_idx]);
			parameters[para_idx]->output_edges.push_back(edge.get());
		}
		if (gate->is_parameter_gate()) {
			auto node = std::make_unique< DAGNode >();
			node->type = DAGNode::internal_param;
			node->index = *output_para_index = (int)parameters.size();
			node->input_edges.push_back(edge.get());
			edge->output_nodes.push_back(node.get());
			parameters.push_back(node.get());
			nodes.push_back(std::move(node));
		}
		else {
			assert(gate->is_quantum_gate());
			for (auto qubit_idx : qubit_indices) {
				auto node = std::make_unique< DAGNode >();
				node->type = DAGNode::internal_qubit;
				node->index = qubit_idx;
				node->input_edges.push_back(edge.get());
				edge->output_nodes.push_back(node.get());
				outputs[qubit_idx] = node.get(); // Update outputs
				nodes.push_back(std::move(node));
			}
		}
		edges.push_back(std::move(edge));
		hash_value_valid_ = false;
		return true;
	}

	void DAG::add_input_parameter() {
		auto node = std::make_unique< DAGNode >();
		node->type = DAGNode::input_param;
		node->index = num_input_parameters;
		parameters.insert(parameters.begin() + num_input_parameters,
		                  node.get());
		nodes.insert(nodes.begin() + num_input_parameters, std::move(node));

		num_input_parameters++;

		// Update internal parameters' indices
		for (auto &it : nodes) {
			if (it->type == DAGNode::internal_param) {
				it->index++;
			}
		}

		// This function should not modify the hash value.
	}

	bool DAG::remove_last_gate() {
		if (edges.empty()) {
			return false;
		}

		auto *edge = edges.back().get();
		auto *gate = edge->gate;
		// Remove edges from input nodes.
		for (auto *input_node : edge->input_nodes) {
			assert(!input_node->output_edges.empty());
			assert(input_node->output_edges.back() == edge);
			input_node->output_edges.pop_back();
		}

		if (gate->is_parameter_gate()) {
			// Remove the parameter.
			assert(!nodes.empty());
			assert(nodes.back()->type == DAGNode::internal_param);
			assert(nodes.back()->index == (int)parameters.size() - 1);
			parameters.pop_back();
			nodes.pop_back();
		}
		else {
			assert(gate->is_quantum_gate());
			// Restore the outputs.
			for (auto *input_node : edge->input_nodes) {
				if (input_node->is_qubit()) {
					outputs[input_node->index] = input_node;
				}
			}
			// Remove the qubit wires.
			while (!nodes.empty() && !nodes.back()->input_edges.empty() &&
			       nodes.back()->input_edges.back() == edge) {
				nodes.pop_back();
			}
		}

		// Remove the edge.
		edges.pop_back();

		hash_value_valid_ = false;
		return true;
	}

	void DAG::generate_parameter_gates(Context *ctx, int max_recursion_depth) {
		assert(max_recursion_depth == 1);
		for (const auto &idx : ctx->get_supported_parameter_gates()) {
			Gate *gate = ctx->get_gate(idx);
			if (gate->get_num_parameters() == 1) {
				std::vector< int > param_indices(1);
				for (param_indices[0] = 0;
				     param_indices[0] < get_num_input_parameters();
				     param_indices[0]++) {
					int output_param_index;
					bool ret =
					    add_gate({}, param_indices, gate, &output_param_index);
					assert(ret);
				}
			}
			else if (gate->get_num_parameters() == 2) {
				// Case: 0-qubit operators with 2 parameters
				std::vector< int > param_indices(2);
				for (param_indices[0] = 0;
				     param_indices[0] < get_num_input_parameters();
				     param_indices[0]++) {
					for (param_indices[1] = 0;
					     param_indices[1] < get_num_input_parameters();
					     param_indices[1]++) {
						if (gate->is_commutative() &&
						    param_indices[0] > param_indices[1]) {
							// For commutative gates, enforce param_indices[0]
							// <= param_indices[1]
							continue;
						}
						int output_param_index;
						bool ret = add_gate({}, param_indices, gate,
						                    &output_param_index);
						assert(ret);
					}
				}
			}
			else {
				assert(false && "Unsupported gate type");
			}
		}
	}

	int DAG::remove_gate(DAGHyperEdge *edge) {
		auto edge_pos = std::find_if(edges.begin(), edges.end(),
		                             [&](std::unique_ptr< DAGHyperEdge > &p) {
			                             return p.get() == edge;
		                             });
		if (edge_pos == edges.end()) {
			return 0;
		}

		auto *gate = edge->gate;
		// Remove edges from input nodes.
		for (auto *input_node : edge->input_nodes) {
			assert(!input_node->output_edges.empty());
			auto it = std::find(input_node->output_edges.begin(),
			                    input_node->output_edges.end(), edge);
			assert(it != input_node->output_edges.end());
			input_node->output_edges.erase(it);
		}

		int ret = 1;

		if (gate->is_parameter_gate()) {
			// Remove the parameter.
			assert(edge->output_nodes.size() == 1);
			auto node = edge->output_nodes[0];
			assert(node->type == DAGNode::internal_param);
			while (!node->output_edges.empty()) {
				// Remove edges using the parameter at first.
				// Note: we can't use a for loop with iterators because they
				// will be invalidated.
				ret += remove_gate(node->output_edges[0]);
			}
			auto it = std::find_if(
			    nodes.begin(), nodes.end(),
			    [&](std::unique_ptr< DAGNode > &p) { return p.get() == node; });
			assert(it != nodes.end());
			auto idx = node->index;
			assert(idx >= get_num_input_parameters());
			nodes.erase(it);
			parameters.erase(parameters.begin() + idx);
			// Update the parameter indices.
			for (auto &j : nodes) {
				if (j->is_parameter() && j->index > idx) {
					j->index--;
				}
			}
		}
		else {
			assert(gate->is_quantum_gate());
			int num_outputs = (int)edge->output_nodes.size();
			int j = 0;
			for (int i = 0; i < num_outputs; i++, j++) {
				// Match the input qubits and the output qubits.
				while (j < (int)edge->input_nodes.size() &&
				       !edge->input_nodes[j]->is_qubit()) {
					j++;
				}
				assert(j < (int)edge->input_nodes.size());
				assert(edge->input_nodes[j]->index ==
				       edge->output_nodes[i]->index);
				if (outputs[edge->output_nodes[i]->index] ==
				    edge->output_nodes[i]) {
					// Restore the outputs.
					outputs[edge->output_nodes[i]->index] =
					    edge->input_nodes[j];
				}
				if (edge->output_nodes[i]->output_edges.empty()) {
					// Remove the qubit wires.
					auto it = std::find_if(nodes.begin(), nodes.end(),
					                       [&](std::unique_ptr< DAGNode > &p) {
						                       return p.get() ==
						                              edge->output_nodes[i];
					                       });
					assert(it != nodes.end());
					nodes.erase(it);
				}
				else {
					// Merge the adjacent qubit wires.
					for (auto &e : edge->output_nodes[i]->output_edges) {
						auto it = std::find(e->input_nodes.begin(),
						                    e->input_nodes.end(),
						                    edge->output_nodes[i]);
						assert(it != e->input_nodes.end());
						*it = edge->input_nodes[j];
						edge->input_nodes[j]->output_edges.push_back(e);
					}
					// And then remove the disconnected qubit wire.
					auto it = std::find_if(nodes.begin(), nodes.end(),
					                       [&](std::unique_ptr< DAGNode > &p) {
						                       return p.get() ==
						                              edge->output_nodes[i];
					                       });
					assert(it != nodes.end());
					nodes.erase(it);
				}
			}
		}

		// Remove the edge.
		edge_pos = std::find_if(edges.begin(), edges.end(),
		                        [&](std::unique_ptr< DAGHyperEdge > &p) {
			                        return p.get() == edge;
		                        });
		assert(edge_pos != edges.end());
		edges.erase(edge_pos);

		hash_value_valid_ = false;
		return ret;
	}

	int DAG::remove_first_quantum_gate() {
		for (auto &edge : edges) {
			if (edge->gate->is_quantum_gate()) {
				return remove_gate(edge.get());
			}
		}
		return 0; // nothing removed
	}

	bool DAG::evaluate(const Vector &input_dis,
	                   const std::vector< ParamType > &input_parameters,
	                   Vector &output_dis,
	                   std::vector< ParamType > *parameter_values) const {
		// We should have 2**n entries for the distribution
		if (input_dis.size() != (1 << get_num_qubits()))
			return false;
		if (input_parameters.size() != get_num_input_parameters())
			return false;
		assert(get_num_input_parameters() <= get_num_total_parameters());
		bool output_parameter_values = true;
		if (!parameter_values) {
			parameter_values = new std::vector< ParamType >();
			output_parameter_values = false;
		}
		*parameter_values = input_parameters;
		parameter_values->resize(get_num_total_parameters());

		output_dis = input_dis;

		// Assume the edges are already sorted in the topological order.
		const int num_edges = (int)edges.size();
		for (int i = 0; i < num_edges; i++) {
			std::vector< int > qubit_indices;
			std::vector< ParamType > params;
			for (const auto &input_node : edges[i]->input_nodes) {
				if (input_node->is_qubit()) {
					qubit_indices.push_back(input_node->index);
				}
				else {
					params.push_back((*parameter_values)[input_node->index]);
				}
			}
			if (edges[i]->gate->is_parameter_gate()) {
				// A parameter gate. Compute the new parameter.
				assert(edges[i]->output_nodes.size() == 1);
				const auto &output_node = edges[i]->output_nodes[0];
				(*parameter_values)[output_node->index] =
				    edges[i]->gate->compute(params);
			}
			else {
				// A quantum gate. Update the distribution.
				assert(edges[i]->gate->is_quantum_gate());
				auto *mat = edges[i]->gate->get_matrix(params);
				output_dis.apply_matrix(mat, qubit_indices);
			}
		}
		if (!output_parameter_values) {
			// Delete the temporary variable newed in this function.
			delete parameter_values;
		}
		return true;
	}

	int DAG::get_num_qubits() const { return num_qubits; }

	int DAG::get_num_input_parameters() const { return num_input_parameters; }

	int DAG::get_num_total_parameters() const { return (int)parameters.size(); }

	int DAG::get_num_internal_parameters() const {
		return (int)parameters.size() - num_input_parameters;
	}

	int DAG::get_num_gates() const { return (int)edges.size(); }

	bool DAG::qubit_used(int qubit_index) const {
		return outputs[qubit_index] != nodes[qubit_index].get();
	}

	bool DAG::input_param_used(int param_index) const {
		assert(nodes[get_num_qubits() + param_index]->type ==
		       DAGNode::input_param);
		assert(nodes[get_num_qubits() + param_index]->index == param_index);
		return !nodes[get_num_qubits() + param_index]->output_edges.empty();
	}

    std::pair<InputParamMaskType, std::vector<InputParamMaskType>>
    DAG::get_input_param_mask() const {
	  std::vector<InputParamMaskType> param_mask(get_num_total_parameters());
	  for (int i = 0; i < get_num_input_parameters(); i++) {
	    param_mask[i] = 1 << i;
	  }
	  for (int i = get_num_input_parameters(); i < get_num_total_parameters(); i++) {
	    param_mask[i] = 0;
	    assert(parameters[i]->input_edges.size() == 1);
	    for (auto &input_node : parameters[i]->input_edges[0]->input_nodes) {
          param_mask[i] |= param_mask[input_node->index];
	    }
	  }
      InputParamMaskType usage_mask{0};
	  for (auto &edge : edges) {
	    // Only consider quantum gate usages of parameters
	    if (edge->gate->is_parametrized_gate()) {
	      for (auto &input_node : edge->input_nodes) {
	        if (input_node->is_parameter()) {
	          usage_mask |= param_mask[input_node->index];
	        }
	      }
	    }
	  }
	  return std::make_pair(usage_mask, param_mask);
	}

	void DAG::generate_hash_values(
	    Context *ctx, const ComplexType &hash_value,
	    const PhaseShiftIdType &phase_shift_id,
	    const std::vector< ParamType > &param_values, DAGHashType *main_hash,
	    std::vector< std::pair< DAGHashType, PhaseShiftIdType > > *other_hash) {
		if (kFingerprintInvariantUnderPhaseShift) {
#ifdef USE_ARBLIB
			auto val = hash_value.abs();
			auto max_error = hash_value.get_abs_max_error();
			assert(max_error < kDAGHashMaxError);
#else
			auto val = std::abs(hash_value);
#endif
			*main_hash =
			    (DAGHashType)std::floor((long double)val / (2 * kDAGHashMaxError));
			// Besides rounding the hash value down, we might want to round it
			// up to account for floating point errors.
			other_hash->emplace_back(*main_hash + 1, phase_shift_id);
			return;
		}

		auto val = hash_value.real() * kDAGHashAlpha +
		           hash_value.imag() * (1 - kDAGHashAlpha);
		*main_hash =
		    (DAGHashType)std::floor((long double)val / (2 * kDAGHashMaxError));
		// Besides rounding the hash value down, we might want to round it up to
		// account for floating point errors.
		other_hash->emplace_back(*main_hash + 1, phase_shift_id);
	}

	DAGHashType DAG::hash(Context *ctx) {
		if (hash_value_valid_) {
			return hash_value_;
		}
		const Vector &input_dis =
		    ctx->get_generated_input_dis(get_num_qubits());
		Vector output_dis;
		auto input_parameters =
		    ctx->get_generated_parameters(get_num_input_parameters());
		std::vector< ParamType > all_parameters;
		evaluate(input_dis, input_parameters, output_dis, &all_parameters);
		ComplexType dot_product =
		    output_dis.dot(ctx->get_generated_hashing_dis(get_num_qubits()));

		original_fingerprint_ = dot_product;

		other_hash_values_.clear();
		generate_hash_values(ctx, dot_product, kNoPhaseShift, all_parameters,
		                     &hash_value_, &other_hash_values_);
		hash_value_valid_ = true;

		// Account for phase shifts.
		// If |kFingerprintInvariantUnderPhaseShift| is true,
		// this was already handled above in |generate_hash_values|.
		if (!kFingerprintInvariantUnderPhaseShift &&
		    kCheckPhaseShiftInGenerator) {
			// We try the simplest version first:
			// Apply phase shift for e^(ip) or e^(-ip) for p being a parameter
			// (either input or internal).
			DAGHashType tmp;
			assert(all_parameters.size() == get_num_total_parameters());
			const int num_total_params = get_num_total_parameters();
			for (int i = 0; i < num_total_params; i++) {
				const auto &param = all_parameters[i];
				ComplexType shifted =
				    dot_product * ComplexType{std::cos(param), std::sin(param)};
				generate_hash_values(ctx, shifted, i, all_parameters, &tmp,
				                     &other_hash_values_);
				other_hash_values_.emplace_back(tmp, i);
				shifted = dot_product *
				          ComplexType{std::cos(param), -std::sin(param)};
				generate_hash_values(ctx, shifted, i + num_total_params,
				                     all_parameters, &tmp, &other_hash_values_);
				other_hash_values_.emplace_back(tmp, i + num_total_params);
			}
			if (kCheckPhaseShiftOfPiOver4) {
				// Check phase shift of pi/4, 2pi/4, ..., 7pi/4.
				for (int i = 1; i < 8; i++) {
					const double pi = std::acos(-1.0);
					ComplexType shifted =
					    dot_product *
					    ComplexType{std::cos(pi / 4 * i), std::sin(pi / 4 * i)};
					generate_hash_values(ctx, shifted, i, all_parameters, &tmp,
					                     &other_hash_values_);
					other_hash_values_.emplace_back(
					    tmp, kCheckPhaseShiftOfPiOver4Index + i);
				}
			}
		}
		return hash_value_;
	}

	std::vector< Vector > DAG::get_matrix(Context *ctx) const {
		const auto sz = 1 << get_num_qubits();
		Vector input_dis(sz);
		auto input_parameters =
		    ctx->get_generated_parameters(get_num_input_parameters());
		std::vector< ParamType > all_parameters;
		std::vector< Vector > result(sz);
		for (int S = 0; S < sz; S++) {
			input_dis[S] = ComplexType(1);
			if (S > 0) {
				input_dis[S - 1] = ComplexType(0);
			}
			evaluate(input_dis, input_parameters, result[S], &all_parameters);
		}
		return result;
	}

	bool DAG::hash_value_valid() const { return hash_value_valid_; }

	DAGHashType DAG::cached_hash_value() const {
		assert(hash_value_valid_);
		return hash_value_;
	}

	std::vector< DAGHashType > DAG::other_hash_values() const {
		assert(hash_value_valid_);
		std::vector< DAGHashType > result(other_hash_values_.size());
		for (int i = 0; i < (int)other_hash_values_.size(); i++) {
			result[i] = other_hash_values_[i].first;
		}
		return result;
	}

	std::vector< std::pair< DAGHashType, PhaseShiftIdType > >
	DAG::other_hash_values_with_phase_shift_id() const {
		assert(hash_value_valid_);
		return other_hash_values_;
	}

	bool DAG::remove_unused_qubits(std::vector< int > unused_qubits) {
		if (unused_qubits.empty()) {
			return true;
		}
		std::sort(unused_qubits.begin(), unused_qubits.end(), std::greater<>());
		for (auto &id : unused_qubits) {
			if (id >= get_num_qubits()) {
				return false;
			}
			if (nodes[id]->type != DAGNode::input_qubit) {
				return false;
			}
			if (nodes[id]->index != id) {
				return false;
			}
			if (!nodes[id]->output_edges.empty()) {
				// used
				return false;
			}
			nodes.erase(nodes.begin() + id);
			outputs.erase(outputs.begin() + id);
			num_qubits--;
			for (auto &node : nodes) {
				if (node->is_qubit() && node->index > id) {
					node->index--;
				}
			}
		}
		hash_value_valid_ = false;
		return true;
	}

	bool
	DAG::remove_unused_input_params(std::vector< int > unused_input_params) {
		if (unused_input_params.empty()) {
			return true;
		}
		std::sort(unused_input_params.begin(), unused_input_params.end(),
		          std::greater<>());
		for (auto &id : unused_input_params) {
			if (id >= get_num_input_parameters()) {
				return false;
			}
			if (nodes[get_num_qubits() + id]->type != DAGNode::input_param) {
				return false;
			}
			if (nodes[get_num_qubits() + id]->index != id) {
				return false;
			}
			if (!nodes[get_num_qubits() + id]->output_edges.empty()) {
				// used
				return false;
			}
			nodes.erase(nodes.begin() + get_num_qubits() + id);
			parameters.erase(parameters.begin() + id);
			num_input_parameters--;
			for (auto &node : nodes) {
				if (node->is_parameter() && node->index > id) {
					node->index--;
				}
			}
		}
		hash_value_valid_ = false;
		return true;
	}

	DAG &DAG::shrink_unused_input_parameters() {
		// Warning: the hash function should be designed such that this function
		// doesn't change the hash value.
		if (get_num_input_parameters() == 0) {
			return *this;
		}
		int last_unused_input_param_index = get_num_input_parameters();
		while (last_unused_input_param_index > 0 &&
		       nodes[get_num_qubits() + last_unused_input_param_index - 1]
		           ->output_edges.empty()) {
			last_unused_input_param_index--;
		}
		if (last_unused_input_param_index == get_num_input_parameters()) {
			// no need to shrink
			return *this;
		}
		int num_parameters_shrinked =
		    get_num_input_parameters() - last_unused_input_param_index;

		// Erase the parameters and the nodes
		parameters.erase(parameters.begin() + last_unused_input_param_index,
		                 parameters.begin() + get_num_input_parameters());
		nodes.erase(
		    nodes.begin() + get_num_qubits() + last_unused_input_param_index,
		    nodes.begin() + get_num_qubits() + get_num_input_parameters());

		// Update the parameter indices
		for (auto &node : nodes) {
			if (node->is_parameter() &&
			    node->index >= get_num_input_parameters()) {
				// An internal parameter
				node->index -= num_parameters_shrinked;
			}
		}

		// Update num_input_parameters
		num_input_parameters -= num_parameters_shrinked;
		return *this;
	}

	std::unique_ptr< DAG >
	DAG::clone_and_shrink_unused_input_parameters() const {
		auto cloned_dag = std::make_unique< DAG >(*this);
		cloned_dag->shrink_unused_input_parameters();
		return cloned_dag;
	}

	bool DAG::has_unused_parameter() const {
		for (auto &node : nodes) {
			if (node->is_parameter() && node->output_edges.empty()) {
				return true;
			}
		}
		return false;
	}

	int DAG::remove_unused_internal_parameters() {
		int num_removed = 0;
		int edge_id = (int)edges.size() - 1;
		while (edge_id >= 0) {
			if (edges[edge_id]->gate->is_parameter_gate()) {
				assert(edges[edge_id]->output_nodes.size() == 1);
				if (edges[edge_id]->output_nodes[0]->output_edges.empty()) {
					num_removed += remove_gate(edges[edge_id].get());
				}
			}
			edge_id--;
		}
		return num_removed;
	}

	void DAG::print(Context *ctx) const {
		for (size_t i = 0; i < edges.size(); i++) {
			DAGHyperEdge *edge = edges[i].get();
			printf("gate[%zu] type(%d)\n", i, edge->gate->tp);
			for (size_t j = 0; j < edge->input_nodes.size(); j++) {
				DAGNode *node = edge->input_nodes[j];
				if (node->is_qubit()) {
					printf("    inputs[%zu]: qubit(%d)\n", j, node->index);
				}
				else {
					printf("    inputs[%zu]: param(%d)\n", j, node->index);
				}
			}
		}
	}

	std::string DAG::to_string() const {
		std::string result;
		result += "DAG {\n";
		const int num_edges = (int)edges.size();
		for (int i = 0; i < num_edges; i++) {
			result += "  ";
			if (edges[i]->output_nodes.size() == 1) {
				result += edges[i]->output_nodes[0]->to_string();
			}
			else if (edges[i]->output_nodes.size() == 2) {
				result += "[" + edges[i]->output_nodes[0]->to_string();
				result += ", " + edges[i]->output_nodes[1]->to_string();
				result += "]";
			}
			else {
				assert(false && "A hyperedge should have 1 or 2 outputs.");
			}
			result += " = ";
			result += gate_type_name(edges[i]->gate->tp);
			result += "(";
			for (int j = 0; j < (int)edges[i]->input_nodes.size(); j++) {
				result += edges[i]->input_nodes[j]->to_string();
				if (j != (int)edges[i]->input_nodes.size() - 1) {
					result += ", ";
				}
			}
			result += ")";
			result += "\n";
		}
		result += "}\n";
		return result;
	}

	std::string DAG::to_json() const {
		std::string result;
		result += "[";

		// basic info
		result += "[";
		result += std::to_string(get_num_qubits());
		result += ",";
		result += std::to_string(get_num_input_parameters());
		result += ",";
		result += std::to_string(get_num_total_parameters());
		result += ",";
		result += std::to_string(get_num_gates());
		result += ",";

		result += "[";
		if (hash_value_valid_) {
			bool first_other_hash_value = true;
			for (const auto &val : other_hash_values_with_phase_shift_id()) {
				if (first_other_hash_value) {
					first_other_hash_value = false;
				}
				else {
					result += ",";
				}
				static char buffer[64];
				auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer),
				                               val.first, /*base=*/
				                               16);
				assert(ec == std::errc());
				auto hash_value = std::string(buffer, ptr);
				if (kCheckPhaseShiftInGenerator &&
				    val.second != kNoPhaseShift) {
					// hash value and phase shift id
					result += "[\"" + hash_value + "\"," +
					          std::to_string(val.second) + "]";
				}
				else {
					// hash value only
					result += "\"" + hash_value + "\"";
				}
			}
		}
		result += "]";

		result += ",";
		result += "[";
		// std::to_chars for floating-point numbers is not supported by some
		// compilers, including GCC with version < 11.
		result += to_string_with_precision(original_fingerprint_.real(),
		                                   /*precision=*/17);
		result += ",";
		result += to_string_with_precision(original_fingerprint_.imag(),
		                                   /*precision=*/17);
		result += "]";

		result += "],";

		// gates
		const int num_edges = (int)edges.size();
		result += "[";
		for (int i = 0; i < num_edges; i++) {
			result += "[";
			result += "\"" + gate_type_name(edges[i]->gate->tp) + "\", ";
			if (edges[i]->output_nodes.size() == 1) {
				result +=
				    "[\"" + edges[i]->output_nodes[0]->to_string() + "\"],";
			}
			else if (edges[i]->output_nodes.size() == 2) {
				result += "[\"" + edges[i]->output_nodes[0]->to_string();
				result += "\", \"" + edges[i]->output_nodes[1]->to_string();
				result += "\"],";
			}
			else {
				assert(false && "A hyperedge should have 1 or 2 outputs.");
			}
			result += "[";
			for (int j = 0; j < (int)edges[i]->input_nodes.size(); j++) {
				result += "\"" + edges[i]->input_nodes[j]->to_string() + "\"";
				if (j != (int)edges[i]->input_nodes.size() - 1) {
					result += ", ";
				}
			}
			result += "]]";
			if (i + 1 != num_edges)
				result += ",";
		}
		result += "]";

		result += "]\n";
		return result;
	}

	std::unique_ptr< DAG > DAG::read_json(Context *ctx, std::istream &fin) {
		fin.ignore(std::numeric_limits< std::streamsize >::max(), '[');

		// basic info
		int num_dag_qubits, num_input_params, num_total_params, num_gates;
		fin.ignore(std::numeric_limits< std::streamsize >::max(), '[');
		fin >> num_dag_qubits;
		fin.ignore(std::numeric_limits< std::streamsize >::max(), ',');
		fin >> num_input_params;
		fin.ignore(std::numeric_limits< std::streamsize >::max(), ',');
		fin >> num_total_params;
		fin.ignore(std::numeric_limits< std::streamsize >::max(), ',');
		fin >> num_gates;

		// ignore other hash values
		fin.ignore(std::numeric_limits< std::streamsize >::max(), '[');
		while (true) {
			char ch;
			fin.get(ch);
			while (ch != '[' && ch != ']') {
				fin.get(ch);
			}
			if (ch == '[') {
				// A hash value with a phase shift id.
				fin.ignore(std::numeric_limits< std::streamsize >::max(), ']');
			}
			else {
				// ch == ']'
				break;
			}
		}

		fin.ignore(std::numeric_limits< std::streamsize >::max(), ']');
		fin.ignore(std::numeric_limits< std::streamsize >::max(), ',');

		auto result = std::make_unique< DAG >(num_dag_qubits, num_input_params);

		// gates
		fin.ignore(std::numeric_limits< std::streamsize >::max(), '[');
		while (true) {
			char ch;
			fin.get(ch);
			while (ch != '[' && ch != ']') {
				fin.get(ch);
			}
			if (ch == ']') {
				break;
			}

			// New gate
			fin.ignore(std::numeric_limits< std::streamsize >::max(), '\"');
			std::string name;
			std::getline(fin, name, '\"');
			auto gate_type = to_gate_type(name);
			Gate *gate = ctx->get_gate(gate_type);

			std::vector< int > input_qubits, input_params, output_qubits,
			    output_params;
			auto read_indices = [&](std::vector< int > &qubit_indices,
			                        std::vector< int > &param_indices) {
				fin.ignore(std::numeric_limits< std::streamsize >::max(), '[');
				while (true) {
					fin.get(ch);
					while (ch != '\"' && ch != ']') {
						fin.get(ch);
					}
					if (ch == ']') {
						break;
					}

					// New index
					fin.get(ch);
					assert(ch == 'P' || ch == 'Q');
					int index;
					fin >> index;
					fin.ignore(); // '\"'
					if (ch == 'Q') {
						qubit_indices.push_back(index);
					}
					else {
						param_indices.push_back(index);
					}
				}
			};
			read_indices(output_qubits, output_params);
			read_indices(input_qubits, input_params);
			fin.ignore(std::numeric_limits< std::streamsize >::max(), ']');

			int output_param_index;
			result->add_gate(input_qubits, input_params, gate,
			                 &output_param_index);
			if (gate->is_parameter_gate()) {
				assert(output_param_index == output_params[0]);
			}
		}

		fin.ignore(std::numeric_limits< std::streamsize >::max(), ']');

		return result;
	}

	bool DAG::minimal_circuit_representation(std::unique_ptr< DAG > *output_dag,
	                                         bool output) const {
		if (output) {
			// |output_dag| cannot be nullptr but its content can (and should)
			// be nullptr.
			assert(output_dag);
			// This deletes the content |output_dag| previously stored.
			*output_dag = std::make_unique< DAG >(get_num_qubits(),
			                                      get_num_input_parameters());
		}

		std::vector< int > qubit_depth(get_num_qubits(), 0);

		bool this_is_minimal_circuit_representation = true;
		// map this DAG to the minimal circuit representation
		int num_mapped_edges = 0;

		// Check if all parameter gates are at the beginning.
		bool have_quantum_gates = false;
		for (auto &edge : edges) {
			if (edge->gate->is_parameter_gate()) {
				if (have_quantum_gates) {
					this_is_minimal_circuit_representation = false;
					if (!output) {
						// no side-effects, early return
						return false;
					}
				}
				num_mapped_edges++;
				if (output) {
					int output_param_index;
					std::vector< int > param_indices;
					for (auto &input_node : edge->input_nodes) {
						assert(input_node->is_parameter());
						param_indices.push_back(input_node->index);
					}
					(*output_dag)
					    ->add_gate({}, param_indices, edge->gate,
					               &output_param_index);
				}
			}
			else {
				have_quantum_gates = true;
			}
		}

		auto get_gate_depth = [&](DAGHyperEdge *edge) {
			int result = 0;
			for (auto &input_node : edge->input_nodes) {
				if (input_node->is_qubit()) {
					result = std::max(result, qubit_depth[input_node->index]);
				}
			}
			return result;
		};

		auto get_min_qubit_index = [&](DAGHyperEdge *edge) {
			int result = get_num_qubits();
			for (auto &input_node : edge->input_nodes) {
				if (input_node->is_qubit()) {
					result = std::min(result, input_node->index);
				}
			}
			return result;
		};

		std::unordered_map< DAGNode *, int > node_id;
		std::unordered_map< DAGHyperEdge *, int > edge_id;
		for (int i = 0; i < (int)nodes.size(); i++) {
			node_id[nodes[i].get()] = i;
		}
		for (int i = 0; i < (int)edges.size(); i++) {
			edge_id[edges[i].get()] = i;
		}

		// We are not using std::priority_queue because the depth of each edge
		// may change.
		std::vector< DAGHyperEdge * > free_edges;

		std::vector< DAGNode * > free_nodes;
		free_nodes.reserve(get_num_qubits() + get_num_input_parameters());

		// Construct the |free_nodes| vector with the input qubit nodes.
		std::vector< int > node_in_degree(nodes.size(), 0);
		std::vector< int > edge_in_degree(edges.size(), 0);
		for (auto &node : nodes) {
			if (node->is_parameter()) {
				node_in_degree[node_id[node.get()]] = -1;
				continue;
			}
			node_in_degree[node_id[node.get()]] = (int)node->input_edges.size();
			if (!node_in_degree[node_id[node.get()]]) {
				free_nodes.push_back(node.get());
			}
		}
		for (auto &edge : edges) {
			edge_in_degree[edge_id[edge.get()]] = 0;
			for (auto &input_node : edge->input_nodes) {
				if (input_node->is_qubit()) {
					edge_in_degree[edge_id[edge.get()]]++;
				}
			}
		}

		while (!free_nodes.empty() || !free_edges.empty()) {
			// Remove the nodes in |free_nodes|.
			for (auto &node : free_nodes) {
				for (auto &output_edge : node->output_edges) {
					if (!--edge_in_degree[edge_id[output_edge]]) {
						free_edges.push_back(output_edge);
					}
				}
			}
			free_nodes.clear();

			if (!free_edges.empty()) {
				// Find the smallest free edge (gate).
				int min_depth = -1;
				int min_qubit_index = -1;
				DAGHyperEdge *smallest_free_edge = nullptr;
				int smallest_free_edge_pos = -1;
				for (int i = 0; i < (int)free_edges.size(); i++) {
					auto &edge = free_edges[i];
					int depth = get_gate_depth(edge);
					if (smallest_free_edge && depth > min_depth) {
						continue;
					}
					int qubit_index = get_min_qubit_index(edge);
					if (!smallest_free_edge || depth < min_depth ||
					    (depth == min_depth && qubit_index < min_qubit_index)) {
						min_depth = depth;
						min_qubit_index = qubit_index;
						smallest_free_edge = edge;
						smallest_free_edge_pos = i;
					}
				}
				free_edges.erase(free_edges.begin() + smallest_free_edge_pos);

				// Map |smallest_free_edge| (a quantum gate).
				if (edges[num_mapped_edges].get() != smallest_free_edge) {
					this_is_minimal_circuit_representation = false;
					if (!output) {
						// no side-effects, early return
						return false;
					}
				}
				num_mapped_edges++;
				if (output) {
					std::vector< int > qubit_indices, param_indices;
					for (auto &input_node : smallest_free_edge->input_nodes) {
						if (input_node->is_qubit()) {
							qubit_indices.push_back(input_node->index);
						}
						else {
							param_indices.push_back(input_node->index);
						}
					}
					int output_param_index;
					(*output_dag)
					    ->add_gate(qubit_indices, param_indices,
					               smallest_free_edge->gate,
					               &output_param_index);
					if (smallest_free_edge->gate->is_parameter_gate()) {
						assert(smallest_free_edge->output_nodes[0]->index ==
						       output_param_index);
					}
				}

				// Update the free nodes.
				for (auto &output_node : smallest_free_edge->output_nodes) {
					if (!--node_in_degree[node_id[output_node]]) {
						free_nodes.push_back(output_node);
					}
				}
			}
		}

		// The DAG should have all gates mapped.
		assert(num_mapped_edges == get_num_gates());

		return this_is_minimal_circuit_representation;
	}

	bool DAG::is_minimal_circuit_representation() const {
		return minimal_circuit_representation(nullptr, false);
	}

	std::unique_ptr< DAG >
	DAG::get_permuted_dag(const std::vector< int > &qubit_permutation,
	                      const std::vector< int > &param_permutation) const {
		auto result = std::make_unique< DAG >(0, 0);
		result->clone_from(*this, qubit_permutation, param_permutation);
		return result;
	}

	void DAG::clone_from(const DAG &other,
	                     const std::vector< int > &qubit_permutation,
	                     const std::vector< int > &param_permutation) {
		num_qubits = other.num_qubits;
		num_input_parameters = other.num_input_parameters;
		hash_value_ = other.hash_value_;
		other_hash_values_ = other.other_hash_values_;
		hash_value_valid_ = other.hash_value_valid_;
		original_fingerprint_ = other.original_fingerprint_;
		std::unordered_map< DAGNode *, DAGNode * > nodes_mapping;
		std::unordered_map< DAGHyperEdge *, DAGHyperEdge * > edges_mapping;
		nodes.reserve(other.nodes.size());
		edges.reserve(other.edges.size());
		outputs.reserve(other.outputs.size());
		parameters.reserve(other.parameters.size());
		if (qubit_permutation.empty() && param_permutation.empty()) {
			for (int i = 0; i < (int)other.nodes.size(); i++) {
				nodes.emplace_back(
				    std::make_unique< DAGNode >(*(other.nodes[i])));
				assert(nodes[i].get() !=
				       other.nodes[i].get()); // make sure we make a copy
				nodes_mapping[other.nodes[i].get()] = nodes[i].get();
			}
		}
		else {
			assert(qubit_permutation.size() == num_qubits);
			nodes.resize(other.nodes.size());
			for (int i = 0; i < num_qubits; i++) {
				assert(qubit_permutation[i] >= 0 &&
				       qubit_permutation[i] < num_qubits);
				nodes[qubit_permutation[i]] =
				    std::make_unique< DAGNode >(*(other.nodes[i]));
				nodes[qubit_permutation[i]]->index =
				    qubit_permutation[i]; // update index
				assert(nodes[qubit_permutation[i]].get() !=
				       other.nodes[i].get());
				nodes_mapping[other.nodes[i].get()] =
				    nodes[qubit_permutation[i]].get();
			}
			const int num_permuted_parameters =
			    std::min(num_input_parameters, (int)param_permutation.size());
			for (int i_inc = 0; i_inc < num_permuted_parameters; i_inc++) {
				assert(param_permutation[i_inc] >= 0 &&
				       param_permutation[i_inc] < num_input_parameters);
				const int i = num_qubits + i_inc;
				nodes[num_qubits + param_permutation[i_inc]] =
				    std::make_unique< DAGNode >(*(other.nodes[i]));
				nodes[num_qubits + param_permutation[i_inc]]->index =
				    param_permutation[i_inc]; // update index
				assert(nodes[num_qubits + param_permutation[i_inc]].get() !=
				       other.nodes[i].get());
				nodes_mapping[other.nodes[i].get()] =
				    nodes[num_qubits + param_permutation[i_inc]].get();
			}
			for (int i = num_qubits + num_permuted_parameters;
			     i < (int)other.nodes.size(); i++) {
				nodes[i] = std::make_unique< DAGNode >(*(other.nodes[i]));
				if (nodes[i]->is_qubit()) {
					nodes[i]->index =
					    qubit_permutation[nodes[i]->index]; // update index
				}
				assert(nodes[i].get() != other.nodes[i].get());
				nodes_mapping[other.nodes[i].get()] = nodes[i].get();
			}
		}
		for (int i = 0; i < (int)other.edges.size(); i++) {
			edges.emplace_back(
			    std::make_unique< DAGHyperEdge >(*(other.edges[i])));
			assert(edges[i].get() != other.edges[i].get());
			edges_mapping[other.edges[i].get()] = edges[i].get();
		}
		for (auto &node : nodes) {
			for (auto &edge : node->input_edges) {
				edge = edges_mapping[edge];
			}
			for (auto &edge : node->output_edges) {
				edge = edges_mapping[edge];
			}
		}
		for (auto &edge : edges) {
			for (auto &node : edge->input_nodes) {
				node = nodes_mapping[node];
			}
			for (auto &node : edge->output_nodes) {
				node = nodes_mapping[node];
			}
		}
		for (auto &node : other.outputs) {
			outputs.emplace_back(nodes_mapping[node]);
		}
		for (auto &node : other.parameters) {
			parameters.emplace_back(nodes_mapping[node]);
		}
	}

	std::vector< DAGHyperEdge * > DAG::first_quantum_gates() const {
		std::vector< DAGHyperEdge * > result;
		std::unordered_set< DAGHyperEdge * > depend_on_other_gates;
		depend_on_other_gates.reserve(edges.size());
		for (const auto &edge : edges) {
			if (edge->gate->is_parameter_gate()) {
				continue;
			}
			if (depend_on_other_gates.find(edge.get()) ==
			    depend_on_other_gates.end()) {
				result.push_back(edge.get());
			}
			for (const auto &output_node : edge->output_nodes) {
				for (const auto &output_edge : output_node->output_edges) {
					depend_on_other_gates.insert(output_edge);
				}
			}
		}
		return result;
	}

	std::vector< DAGHyperEdge * > DAG::last_quantum_gates() const {
		std::vector< DAGHyperEdge * > result;
		for (const auto &edge : edges) {
			if (edge->gate->is_parameter_gate()) {
				continue;
			}
			bool all_output = true;
			for (const auto &output_node : edge->output_nodes) {
				if (outputs[output_node->index] != output_node) {
					all_output = false;
					break;
				}
			}
			if (all_output) {
				result.push_back(edge.get());
			}
		}
		return result;
	}

	bool DAG::same_gate(const DAG &dag1, int index1, const DAG &dag2,
	                    int index2) {
		assert(dag1.get_num_gates() > index1);
		assert(dag2.get_num_gates() > index2);
		return same_gate(dag1.edges[index1].get(), dag2.edges[index2].get());
	}

	bool DAG::same_gate(DAGHyperEdge *edge1, DAGHyperEdge *edge2) {
		if (edge1->gate != edge2->gate) {
			return false;
		}
		if (edge1->input_nodes.size() != edge2->input_nodes.size()) {
			return false;
		}
		if (edge1->output_nodes.size() != edge2->output_nodes.size()) {
			return false;
		}
		for (int i = 0; i < (int)edge1->output_nodes.size(); i++) {
			if (edge1->output_nodes[i]->type != edge2->output_nodes[i]->type) {
				return false;
			}
			if (edge1->output_nodes[i]->index !=
			        edge2->output_nodes[i]->index &&
			    edge1->output_nodes[i]->type != DAGNode::internal_param) {
				return false;
			}
		}
		for (int i = 0; i < (int)edge1->input_nodes.size(); i++) {
			if (edge1->input_nodes[i]->type != edge2->input_nodes[i]->type) {
				return false;
			}
			if (edge1->input_nodes[i]->index != edge2->input_nodes[i]->index &&
			    edge1->input_nodes[i]->type != DAGNode::internal_param) {
				return false;
			}
			if (edge1->input_nodes[i]->type == DAGNode::internal_param) {
				// Internal parameters are checked recursively.
				assert(edge1->input_nodes[i]->input_edges.size() == 1);
				assert(edge2->input_nodes[i]->input_edges.size() == 1);
				if (!same_gate(edge1->input_nodes[i]->input_edges[0],
				               edge2->input_nodes[i]->input_edges[0])) {
					return false;
				}
			}
		}
		return true;
	}

} // namespace quartz