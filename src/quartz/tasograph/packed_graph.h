#pragma once

#include "tasograph.h"

#include <cstdint>

namespace quartz {

// An in-process, immutable search snapshot. Context and its Gate objects must
// outlive the snapshot. Gate pointers preserve controlled-gate variants without
// copying context-owned gate definitions. No Graph or CircuitSeq is retained.
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
