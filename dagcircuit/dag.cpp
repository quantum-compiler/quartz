#include "dag.h"
#include <cassert>

DAG::DAG(int _num_qubits, int _num_parameters)
: num_qubits(_num_qubits), num_parameters(_num_parameters)
{
  // Initialize num_qubits qubits
  for (int i = 0; i < num_qubits; i++) {
    DAGNode* node = new DAGNode();
    node->type = DAGNode::input_qubit;
    nodes.push_back(node);
    outputs.push_back(node);
  }
  // Initialize num_parameters parameters
  for (int i = 0; i < num_parameters; i++) {
    DAGNode* node = new DAGNode();
    node->type = DAGNode::input_param;
    nodes.push_back(node);
    parameters.push_back(node);
  }
}

bool DAG::add_gate(const std::vector<int>& qubit_indices,
                   const std::vector<int>& parameter_indices,
                   const Gate* gate,
                   int* output_para_index)
{
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
  DAGHyperEdge* edge = new DAGHyperEdge();
  edges.push_back(edge);
  edge->gate = gate;
  for (auto qubit_idx : qubit_indices) {
    edge->input_nodes.push_back(outputs[qubit_idx]);
    outputs[qubit_idx]->output_edges.push_back(edge);
  }
  for (auto para_idx : parameter_indices) {
    edge->input_nodes.push_back(parameters[para_idx]);
    parameters[para_idx]->output_edges.push_back(edge);
  }
  if (gate->is_parameter_gate()) {
    DAGNode* node = new DAGNode();
    nodes.push_back(node);
    node->type = DAGNode::internal_param;
    node->input_edges.push_back(edge);
    edge->output_nodes.push_back(node);
    parameters.push_back(node);
    *output_para_index = parameters.size();
  } else {
    assert(gate->is_quantum_gate());
    for (auto qubit_idx : qubit_indices) {
      DAGNode* node = new DAGNode();
      nodes.push_back(node);
      node->type = DAGNode::internal_param;
      node->input_edges.push_back(edge);
      edge->output_nodes.push_back(node);
      outputs[qubit_idx] = node; // Update outputs
    }
  }
}

int DAG::get_num_qubits() const
{
  return num_qubits;
}

DAG::~DAG()
{
  for (auto node : nodes)
    delete node;
  for (auto edge : edges)
    delete edge;
}
