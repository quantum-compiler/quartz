#include "dag.h"
#include "../gate/gate.h"
#include "../context/context.h"

#include <cassert>

DAG::DAG(int _num_qubits, int _num_parameters)
    : num_qubits(_num_qubits),
      num_input_parameters(_num_parameters),
      hash_value_(0),
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

DAG::DAG(const DAG &other)
    : num_qubits(other.num_qubits),
      num_input_parameters(other.num_input_parameters),
      hash_value_(other.hash_value_),
      hash_value_valid_(other.hash_value_valid_) {
  std::unordered_map<DAGNode *, DAGNode *> nodes_mapping;
  std::unordered_map<DAGHyperEdge *, DAGHyperEdge *> edges_mapping;
  nodes.reserve(other.nodes.size());
  edges.reserve(other.edges.size());
  outputs.reserve(other.outputs.size());
  parameters.reserve(other.parameters.size());
  for (int i = 0; i < (int) other.nodes.size(); i++) {
    nodes.emplace_back(std::make_unique<DAGNode>(*(other.nodes[i])));
    assert(nodes[i].get() != other.nodes[i].get()); // make sure we make a copy
    nodes_mapping[other.nodes[i].get()] = nodes[i].get();
  }
  for (int i = 0; i < (int) other.edges.size(); i++) {
    edges.emplace_back(std::make_unique<DAGHyperEdge>(*(other.edges[i])));
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

bool DAG::remove_last_gate() {
  if (edges.empty()) {
    return false;
  }

  auto *edge = edges.back().get();
  auto *gate = edge->gate;
  // Remove edges from input nodes.
  for (auto *input_node : edge->input_nodes) {
    assert(input_node->output_edges.back() == edge);
    input_node->output_edges.pop_back();
  }

  if (gate->is_parameter_gate()) {
    // Remove the parameter.
    assert(nodes.back()->type == DAGNode::internal_param);
    assert(nodes.back()->index == (int) parameters.size() - 1);
    parameters.pop_back();
  } else {
    assert(gate->is_quantum_gate());
    // Restore the outputs.
    for (auto *input_node : edge->input_nodes) {
      if (input_node->is_qubit()) {
        outputs[input_node->index] = input_node;
      }
    }
    // Remove the qubit wires.
    while (!nodes.empty() && nodes.back()->input_edges.back() == edge) {
      nodes.pop_back();
    }
  }

  // Remove the edge.
  edges.pop_back();
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
