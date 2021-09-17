#include "dag.h"
#include "../gate/gate.h"
#include "../context/context.h"

#include <cassert>

DAG::DAG(int _num_qubits, int _num_parameters)
    : num_qubits(_num_qubits),
      num_input_parameters(_num_parameters),
      hash_value_valid_(false) {
  // Initialize num_qubits qubits
  for (int i = 0; i < num_qubits; i++) {
    auto node = std::make_unique<DAGNode>();
    node->type = DAGNode::input_qubit;
    node->index = i;
    outputs.push_back(node.get());
    nodes.push_back(std::move(node));
  }
  // Initialize num_input_parameters parameters
  for (int i = 0; i < num_input_parameters; i++) {
    auto node = std::make_unique<DAGNode>();
    node->type = DAGNode::input_param;
    node->index = i;
    parameters.push_back(node.get());
    nodes.push_back(std::move(node));
  }
}

DAG::DAG(const DAG &other) {
  // TODO: implement
}

bool DAG::add_gate(const std::vector<int> &qubit_indices,
                   const std::vector<int> &parameter_indices,
                   Gate *gate,
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
  auto edge = std::make_unique<DAGHyperEdge>();
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
    auto node = std::make_unique<DAGNode>();
    node->type = DAGNode::internal_param;
    node->index = *output_para_index = (int) parameters.size();
    node->input_edges.push_back(edge.get());
    edge->output_nodes.push_back(node.get());
    parameters.push_back(node.get());
    nodes.push_back(std::move(node));
  } else {
    assert(gate->is_quantum_gate());
    for (auto qubit_idx : qubit_indices) {
      auto node = std::make_unique<DAGNode>();
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

bool DAG::evaluate(const Vector &input_dis,
                   const std::vector<ParamType> &input_parameters,
                   Vector &output_dis) const {
  // We should have 2**n entries for the distribution
  if (input_dis.size() != (1 << get_num_qubits()))
    return false;
  if (input_parameters.size() != get_num_input_parameters())
    return false;
  assert(get_num_input_parameters() <= get_num_total_parameters());
  std::vector<ParamType> parameter_values = input_parameters;
  parameter_values.resize(get_num_total_parameters());

  output_dis = input_dis;

  // Assume the edges are already sorted in the topological order.
  const int num_edges = (int) edges.size();
  for (int i = 0; i < num_edges; i++) {
    std::vector<int> qubit_indices;
    std::vector<ParamType> params;
    for (const auto &input_node : edges[i]->input_nodes) {
      if (input_node->is_qubit()) {
        qubit_indices.push_back(input_node->index);
      } else {
        params.push_back(parameter_values[input_node->index]);
      }
    }
    if (edges[i]->gate->is_parameter_gate()) {
      // A parameter gate. Compute the new parameter.
      assert(edges[i]->output_nodes.size() == 1);
      const auto &output_node = edges[i]->output_nodes[0];
      parameter_values[output_node->index] = edges[i]->gate->compute(params);
    } else {
      // A quantum gate. Update the distribution.
      assert(edges[i]->gate->is_quantum_gate());
      auto *mat = edges[i]->gate->get_matrix(params);
      output_dis.apply_matrix(mat, qubit_indices);
    }
  }
  return true;
}

int DAG::get_num_qubits() const {
  return num_qubits;
}

int DAG::get_num_input_parameters() const {
  return num_input_parameters;
}

int DAG::get_num_total_parameters() const {
  return (int) parameters.size();
}

size_t DAG::hash(Context *ctx) {
  if (hash_value_valid_) {
    return hash_value_;
  }
  const Vector &input_dis = ctx->get_generated_input_dis(get_num_qubits());
  Vector output_dis;
  evaluate(input_dis,
           ctx->get_generated_parameters(get_num_input_parameters()),
           output_dis);
  ComplexType dot_product =
      output_dis.dot(ctx->get_generated_hashing_dis(get_num_qubits()));
  const int discard_bits = 10;
  assert(typeid(ComplexType::value_type) == typeid(double));
  assert(sizeof(size_t) == sizeof(double));
  auto val1 = dot_product.real(), val2 = dot_product.imag();
  size_t result = *((size_t *) (&val1)) >> discard_bits << discard_bits;
  result ^= *((size_t *) (&val2)) >> discard_bits;
  hash_value_ = result;
  hash_value_valid_ = true;
  return result;
}
