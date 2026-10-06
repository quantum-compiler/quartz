#pragma once

#include "tasograph.h"

#include <cstdint>

namespace quartz {

// An in-process, immutable search snapshot. Context and its Gate objects must
// outlive the snapshot. Gate pointers preserve controlled-gate variants without
// copying context-owned gate definitions. Parameter indices reference the same
// context parameter table, including constants and symbolic expressions. This
// is not portable serialization. No Graph or CircuitSeq is retained.
// Operation IDs and the special-operation counter are preserved so unpacking
// neither changes traversal order nor advances the global ID allocator.
class PackedGraph {
 public:
  explicit PackedGraph(const Graph &graph);
  [[nodiscard]] std::shared_ptr<Graph> unpack() const;
  // Owned storage, including vector capacity; excludes shared context data.
  [[nodiscard]] size_t storage_bytes() const;

 private:
  struct Node {
    Op op;
    int qubit_index = -1;
    int parameter_index = -1;
    // These record map membership, not whether a node has connections. An
    // empty adjacency entry cannot be recovered from connections_ alone; keep
    // it so unpacking restores exactly the original inEdges/outEdges maps.
    bool has_in_edges = false;
    bool has_out_edges = false;
  };
  struct Connection {
    uint32_t source, destination;
    int source_port, destination_port;
  };
  Context *context_;
  size_t special_op_guid_;
  std::vector<Node> nodes_;
  std::vector<Connection> connections_;
};

}  // namespace quartz
