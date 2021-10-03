#include "dag.h"
#include "../gate/gate.h"
#include "../context/context.h"

#include <algorithm>
#include <cassert>

DAG::DAG(int num_qubits, int num_input_parameters)
    : num_qubits(num_qubits),
      num_input_parameters(num_input_parameters),
      hash_value_(0),
      hash_value_valid_(false) {
  nodes.reserve(num_qubits + num_input_parameters);
  outputs.reserve(num_qubits);
  parameters.reserve(num_input_parameters);
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

std::unique_ptr<DAG> DAG::clone() const {
  return std::make_unique<DAG>(*this);
}

bool DAG::fully_equivalent(const DAG &other) const {
  if (num_qubits != other.num_qubits
      || num_input_parameters != other.num_input_parameters) {
    return false;
  }
  if (nodes.size() != other.nodes.size()
      || edges.size() != other.edges.size()) {
    return false;
  }
  std::unordered_map<DAGNode *, DAGNode *> nodes_mapping;
  for (int i = 0; i < (int) nodes.size(); i++) {
    nodes_mapping[other.nodes[i].get()] = nodes[i].get();
  }
  for (int i = 0; i < (int) edges.size(); i++) {
    if (edges[i]->gate->tp != other.edges[i]->gate->tp) {
      return false;
    }
    if (edges[i]->input_nodes.size() != other.edges[i]->input_nodes.size()
        || edges[i]->output_nodes.size()
            != other.edges[i]->output_nodes.size()) {
      return false;
    }
    for (int j = 0; j < (int) edges[i]->input_nodes.size(); j++) {
      if (nodes_mapping[other.edges[i]->input_nodes[j]]
          != edges[i]->input_nodes[j]) {
        return false;
      }
    }
    for (int j = 0; j < (int) edges[i]->output_nodes.size(); j++) {
      if (nodes_mapping[other.edges[i]->output_nodes[j]]
          != edges[i]->output_nodes[j]) {
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
    return get_num_input_parameters() < other.get_num_input_parameters();
  }
  if (get_num_total_parameters() != other.get_num_total_parameters()) {
    // We want fewer quantum gates, i.e., more traditional parameters.
    return get_num_total_parameters() > other.get_num_total_parameters();
  }
  for (int i = 0; i < (int) edges.size(); i++) {
    if (edges[i]->gate->tp != other.edges[i]->gate->tp) {
      return edges[i]->gate->tp < other.edges[i]->gate->tp;
    }
    for (int j = 0; j < (int) edges[i]->input_nodes.size(); j++) {
      if (edges[i]->input_nodes[j]->index
          != other.edges[i]->input_nodes[j]->index) {
        return edges[i]->input_nodes[j]->index
            < other.edges[i]->input_nodes[j]->index;
      }
    }
    for (int j = 0; j < (int) edges[i]->output_nodes.size(); j++) {
      if (edges[i]->output_nodes[j]->index
          != other.edges[i]->output_nodes[j]->index) {
        return edges[i]->output_nodes[j]->index
            < other.edges[i]->output_nodes[j]->index;
      }
    }
  }
  return false;  // fully equivalent
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
    assert(!input_node->output_edges.empty());
    assert(input_node->output_edges.back() == edge);
    input_node->output_edges.pop_back();
  }

  if (gate->is_parameter_gate()) {
    // Remove the parameter.
    assert(!nodes.empty());
    assert(nodes.back()->type == DAGNode::internal_param);
    assert(nodes.back()->index == (int) parameters.size() - 1);
    parameters.pop_back();
    nodes.pop_back();
  } else {
    assert(gate->is_quantum_gate());
    // Restore the outputs.
    for (auto *input_node : edge->input_nodes) {
      if (input_node->is_qubit()) {
        outputs[input_node->index] = input_node;
      }
    }
    // Remove the qubit wires.
    while (!nodes.empty() && !nodes.back()->input_edges.empty()
        && nodes.back()->input_edges.back() == edge) {
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

int DAG::get_num_gates() const {
  return (int) edges.size();
}

bool DAG::qubit_used(int qubit_index) const {
  return outputs[qubit_index] != nodes[qubit_index].get();
}

DAGHashType DAG::hash(Context *ctx) {
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
  assert(sizeof(DAGHashType) == sizeof(double));
  auto val1 = dot_product.real(), val2 = dot_product.imag();
  DAGHashType
      result = *((DAGHashType *) (&val1)) >> discard_bits << discard_bits;
  result ^= *((DAGHashType *) (&val2)) >> discard_bits;
  hash_value_ = result;
  hash_value_valid_ = true;
  return result;
}

DAG &DAG::shrink_unused_input_parameters() {
  // Warning: the hash function should be designed such that this function
  // doesn't change the hash value.
  if (get_num_input_parameters() == 0) {
    return *this;
  }
  int last_unused_input_param_index = get_num_input_parameters();
  while (last_unused_input_param_index > 0
      && nodes[get_num_qubits() + last_unused_input_param_index
          - 1]->output_edges.empty()) {
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
  nodes.erase(nodes.begin() + get_num_qubits() + last_unused_input_param_index,
              nodes.begin() + get_num_qubits() + get_num_input_parameters());

  // Update the parameter indices
  for (auto &node : nodes) {
    if (node->is_parameter() && node->index >= get_num_input_parameters()) {
      // An internal parameter
      node->index -= num_parameters_shrinked;
    }
  }

  // Update num_input_parameters
  num_input_parameters -= num_parameters_shrinked;
  return *this;
}

std::unique_ptr<DAG> DAG::clone_and_shrink_unused_input_parameters() const {
  auto cloned_dag = std::make_unique<DAG>(*this);
  cloned_dag->shrink_unused_input_parameters();
  return cloned_dag;
}

void DAG::print(Context *ctx) const {
  for (size_t i = 0; i < edges.size(); i++) {
    DAGHyperEdge *edge = edges[i].get();
    printf("gate[%zu] type(%d)\n", i, edge->gate->tp);
    for (size_t j = 0; j < edge->input_nodes.size(); j++) {
      DAGNode *node = edge->input_nodes[j];
      if (node->is_qubit()) {
        printf("    inputs[%zu]: qubit(%d)\n", j, node->index);
      } else {
        printf("    inputs[%zu]: param(%d)\n", j, node->index);
      }
    }
  }
}

std::string DAG::to_string() const {
  std::string result;
  result += "DAG {\n";
  const int num_edges = (int) edges.size();
  for (int i = 0; i < num_edges; i++) {
    result += "  ";
    if (edges[i]->output_nodes.size() == 1) {
      result += edges[i]->output_nodes[0]->to_string();
    } else if (edges[i]->output_nodes.size() == 2) {
      result += "[" + edges[i]->output_nodes[0]->to_string();
      result += ", " + edges[i]->output_nodes[1]->to_string();
      result += "]";
    } else {
      assert(false && "A hyperedge should have 1 or 2 outputs.");
    }
    result += " = ";
    result += gate_type_name(edges[i]->gate->tp);
    result += "(";
    for (int j = 0; j < (int) edges[i]->input_nodes.size(); j++) {
      result += edges[i]->input_nodes[j]->to_string();
      if (j != (int) edges[i]->input_nodes.size() - 1) {
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
  result += "],";

  // gates
  const int num_edges = (int) edges.size();
  result += "[";
  for (int i = 0; i < num_edges; i++) {
    result += "[";
    result += "\"" + gate_type_name(edges[i]->gate->tp) + "\", ";
    if (edges[i]->output_nodes.size() == 1) {
      result += "[\"" + edges[i]->output_nodes[0]->to_string() + "\"],";
    } else if (edges[i]->output_nodes.size() == 2) {
      result += "[\"" + edges[i]->output_nodes[0]->to_string();
      result += "\", \"" + edges[i]->output_nodes[1]->to_string();
      result += "\"],";
    } else {
      assert(false && "A hyperedge should have 1 or 2 outputs.");
    }
    result += "[";
    for (int j = 0; j < (int) edges[i]->input_nodes.size(); j++) {
      result += "\"" + edges[i]->input_nodes[j]->to_string() + "\"";
      if (j != (int) edges[i]->input_nodes.size() - 1) {
        result += ", ";
      }
    }
    result += "]]";
    if (i + 1 != num_edges) result += ",";
  }
  result += "]";

  result += "]\n";
  return result;
}

std::unique_ptr<DAG> DAG::read_json(Context *ctx, std::istream &fin) {
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');

  // basic info
  int num_dag_qubits, num_input_params, num_total_params;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  fin >> num_dag_qubits;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');
  fin >> num_input_params;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');
  fin >> num_total_params;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');

  auto result = std::make_unique<DAG>(num_dag_qubits, num_input_params);

  // gates
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
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
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
    std::string name;
    std::getline(fin, name, '\"');
    auto gate_type = to_gate_type(name);
    Gate *gate = ctx->get_gate(gate_type);

    std::vector<int> input_qubits, input_params, output_qubits, output_params;
    auto read_indices =
        [&](std::vector<int> &qubit_indices, std::vector<int> &param_indices) {
          fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
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
            assert (ch == 'P' || ch == 'Q');
            int index;
            fin >> index;
            fin.ignore();  // '\"'
            if (ch == 'Q') {
              qubit_indices.push_back(index);
            } else {
              param_indices.push_back(index);
            }
          }
        };
    read_indices(output_qubits, output_params);
    read_indices(input_qubits, input_params);
    fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');

    int output_param_index;
    result->add_gate(input_qubits,
                     input_params,
                     gate,
                     &output_param_index);
    if (gate->is_parameter_gate()) {
      assert (output_param_index == output_params[0]);
    }
  }

  fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');

  return result;
}

bool DAG::minimal_representation(std::unique_ptr<DAG> *output_dag,
                                 std::vector<int> *qubit_permutation,
                                 std::vector<int> *param_permutation,
                                 bool output) const {
  if (output) {
    // |output_dag| cannot be nullptr but its content can be nullptr.
    assert (output_dag);
    // This deletes the content |output_dag| previously stored.
    *output_dag =
        std::make_unique<DAG>(get_num_qubits(), get_num_input_parameters());
  }

  std::unordered_map<DAGNode *, int> node_id;
  std::unordered_map<DAGHyperEdge *, int> edge_id;
  for (int i = 0; i < (int) nodes.size(); i++) {
    node_id[nodes[i].get()] = i;
  }
  for (int i = 0; i < (int) edges.size(); i++) {
    edge_id[edges[i].get()] = i;
  }

  std::vector<int> temp_vec[2];

  // Qubit index map from the current DAG to the minimal representation.
  std::vector<int>
      &qubit_index_map = (qubit_permutation ? *qubit_permutation : temp_vec[0]);
  qubit_index_map.assign(get_num_qubits(), -1);
  int num_qubits_mapped = 0;

  // Parameter index map from the current DAG to the minimal representation.
  std::vector<int>
      &param_index_map = (param_permutation ? *param_permutation : temp_vec[1]);
  param_index_map.assign(get_num_total_parameters(), -1);
  int num_input_params_mapped = 0;
  int num_internal_params_mapped = 0;

  auto compare_node = [&](DAGNode *a, DAGNode *b) {
    // Returns false iff we want to prioritize a over b.
    if (a->is_qubit() != b->is_qubit()) {
      return b->is_qubit(); // prioritize qubits over parameters
    }
    if (a->is_qubit()) {
      if (qubit_index_map[a->index] == -1
          && qubit_index_map[b->index] == -1) {
        // Both qubits are not mapped yet, compare by index in the current DAG
        return a->index > b->index;
      } else if (qubit_index_map[a->index] == -1
          || qubit_index_map[b->index] == -1) {
        // One of them is mapped, prioritize it
        return qubit_index_map[b->index] == -1;
      } else {
        return qubit_index_map[a->index] > qubit_index_map[b->index];
      }
    } else {
      if (param_index_map[a->index] == -1
          && param_index_map[b->index] == -1) {
        return a->index > b->index;
      } else if (param_index_map[a->index] == -1
          || param_index_map[b->index] == -1) {
        return param_index_map[b->index] == -1;
      } else {
        return param_index_map[a->index] > param_index_map[b->index];
      }
    }
  };

  auto compare_edge = [&](DAGHyperEdge *a, DAGHyperEdge *b) {
    // Returns false iff we want to prioritize a over b.
    if (a->gate->tp != b->gate->tp) {
      // Prioritize the "smallest" type of gate.
      return a->gate->tp > b->gate->tp;
    }
    // For the same type of gate, compare by their inputs.
    const int num_inputs = (int) a->input_nodes.size();
    assert(b->input_nodes.size() == num_inputs);
    for (int i = 0; i < num_inputs; i++) {
      assert(a->input_nodes[i]->is_qubit() == b->input_nodes[i]->is_qubit());
      if (a->input_nodes[i]->index != b->input_nodes[i]->index) {
        // Having different indices, we can compare the inputs
        return compare_node(a->input_nodes[i], b->input_nodes[i]);
      }
    }
    // Two edges are identical.
    return false;
  };

  // We are not using std::priority_queue because we need to rebuild the heap
  // once the mapping is changed.
  std::vector<DAGHyperEdge *> free_edges;

  std::vector<DAGNode *> free_nodes;
  free_nodes.reserve(get_num_qubits() + get_num_input_parameters());

  // Construct the |free_nodes| vector with the input nodes.
  std::vector<int> node_in_degree(nodes.size(), 0);
  std::vector<int> edge_in_degree(edges.size(), 0);
  for (auto &node : nodes) {
    node_in_degree[node_id[node.get()]] = (int) node->input_edges.size();
    if (!node_in_degree[node_id[node.get()]]) {
      free_nodes.push_back(node.get());
    }
  }
  for (auto &edge : edges) {
    edge_in_degree[edge_id[edge.get()]] = (int) edge->input_nodes.size();
  }

  int num_gates_mapped = 0;
  bool this_is_already_minimal = true;

  while (!free_nodes.empty() || !free_edges.empty()) {
    // Remove the nodes in |free_nodes|.
    for (auto &node : free_nodes) {
      for (auto &output_edge : node->output_edges) {
        if (!--edge_in_degree[edge_id[output_edge]]) {
          free_edges.push_back(output_edge);
          std::push_heap(free_edges.begin(), free_edges.end(), compare_edge);
        }
      }
    }
    free_nodes.clear();

    if (!free_edges.empty()) {
      std::pop_heap(free_edges.begin(), free_edges.end(), compare_edge);
      DAGHyperEdge *current_edge = free_edges.back();
      free_edges.pop_back();

      // Map the current edge (gate).
      bool has_nodes_mapped = false;
      for (auto &input_node : current_edge->input_nodes) {
        if (input_node->is_qubit()) {
          if (qubit_index_map[input_node->index] == -1) {
            qubit_index_map[input_node->index] = num_qubits_mapped++;
            has_nodes_mapped = true;
          }
        } else {
          if (param_index_map[input_node->index] == -1) {
            has_nodes_mapped = true;
            if (input_node->type == DAGNode::Type::input_param) {
              param_index_map[input_node->index] = num_input_params_mapped++;
            } else {
              assert (false && "Internal parameter used before creation");
              param_index_map[input_node->index] =
                  get_num_input_parameters() + num_internal_params_mapped++;
            }
          }
        }
      }

      if (has_nodes_mapped) {
        // The mapping has changed. We need to rebuild the heap.
        std::make_heap(free_edges.begin(), free_edges.end());
      }

      if (current_edge->gate->is_parameter_gate()) {
        for (auto &output_node : current_edge->output_nodes) {
          if (output_node->is_parameter()) {
            assert (param_index_map[output_node->index] == -1);
            param_index_map[output_node->index] =
                get_num_input_parameters() + num_internal_params_mapped++;
          }
        }
      }

      if (output) {
        std::vector<int> qubit_indices, param_indices;
        for (auto &input_node : current_edge->input_nodes) {
          if (input_node->is_qubit()) {
            qubit_indices.push_back(qubit_index_map[input_node->index]);
          } else {
            param_indices.push_back(param_index_map[input_node->index]);
          }
        }
        int output_param_index;
        (*output_dag)->add_gate(qubit_indices,
                                param_indices,
                                current_edge->gate,
                                &output_param_index);
        if (current_edge->gate->is_parameter_gate()) {
          assert (param_index_map[current_edge->output_nodes[0]->index]
                      == output_param_index);
        }
      }

      if (edge_id[current_edge] != num_gates_mapped) {
        this_is_already_minimal = false;
      }
      num_gates_mapped++;

      // Update the free nodes.
      for (auto &output_node : current_edge->output_nodes) {
        if (!--node_in_degree[node_id[output_node]]) {
          free_nodes.push_back(output_node);
        }
      }
    }
  }

  // The DAG should have all gates mapped.
  assert(num_gates_mapped == get_num_gates());

  for (int i = 0; i < get_num_qubits(); i++) {
    if (qubit_index_map[i] != -1 && qubit_index_map[i] != i) {
      // There is a used qubit that is not mapped to itself
      this_is_already_minimal = false;
      break;
    }
  }

  for (int i = 0; i < get_num_total_parameters(); i++) {
    if (param_index_map[i] != -1 && param_index_map[i] != i) {
      // There is a used parameter that is not mapped to itself
      this_is_already_minimal = false;
      break;
    }
  }

  return this_is_already_minimal;
}

bool DAG::is_minimal_representation() const {
  return minimal_representation(nullptr, nullptr, nullptr, false);
}
