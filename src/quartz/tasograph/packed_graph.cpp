#include "packed_graph.h"

#include <limits>
#include <stdexcept>

namespace quartz {

PackedGraph::PackedGraph(const Graph &graph)
    : context_(graph.context), special_op_guid_(graph.special_op_guid) {
  std::map<Op, Node, OpCompare> nodes;
  for (const auto &entry : graph.inEdges) {
    nodes.try_emplace(entry.first).first->second.has_in_edges = true;
  }
  size_t num_connections = 0;
  for (const auto &entry : graph.outEdges) {
    nodes.try_emplace(entry.first).first->second.has_out_edges = true;
    num_connections += entry.second.size();
    for (const auto &edge : entry.second) {
      nodes.try_emplace(edge.srcOp);
      nodes.try_emplace(edge.dstOp);
    }
  }
  for (const auto &entry : graph.input_qubit_op_2_qubit_idx) {
    nodes.try_emplace(entry.first).first->second.qubit_index = entry.second;
  }
  for (const auto &entry : graph.param_idx) {
    nodes.try_emplace(entry.first).first->second.parameter_index = entry.second;
  }
  if (nodes.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("Too many operations to pack a search graph");
  }
  nodes_.reserve(nodes.size());
  std::map<Op, uint32_t, OpCompare> indices;
  for (auto &entry : nodes) {
    entry.second.op = entry.first;
    indices.emplace(entry.first, static_cast<uint32_t>(nodes_.size()));
    nodes_.push_back(entry.second);
  }
  connections_.reserve(num_connections);
  for (const auto &entry : graph.outEdges) {
    for (const auto &edge : entry.second) {
      connections_.push_back({indices.at(edge.srcOp), indices.at(edge.dstOp),
                              edge.srcIdx, edge.dstIdx});
    }
  }
}

std::shared_ptr<Graph> PackedGraph::unpack() const {
  auto graph = std::make_shared<Graph>(context_);
  graph->special_op_guid = special_op_guid_;
  for (const auto &node : nodes_) {
    // Preserve empty map entries as well as isolated input qubits.
    if (node.has_in_edges) {
      graph->inEdges.try_emplace(node.op);
    }
    if (node.has_out_edges) {
      graph->outEdges.try_emplace(node.op);
    }
    if (node.qubit_index >= 0) {
      graph->input_qubit_op_2_qubit_idx.emplace(node.op, node.qubit_index);
    }
    if (node.parameter_index >= 0) {
      graph->param_idx.emplace(node.op, node.parameter_index);
    }
  }
  for (const auto &edge : connections_) {
    graph->add_edge(nodes_[edge.source].op, nodes_[edge.destination].op,
                    edge.source_port, edge.destination_port);
  }
  graph->_construct_pos_2_logical_qubit();
  return graph;
}

size_t PackedGraph::storage_bytes() const {
  return sizeof(*this) + nodes_.capacity() * sizeof(Node) +
         connections_.capacity() * sizeof(Connection);
}

}  // namespace quartz
