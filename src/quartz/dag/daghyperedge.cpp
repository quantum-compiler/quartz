#include "daghyperedge.h"
#include "dagnode.h"

namespace quartz {
int DAGHyperEdge::get_min_qubit_index() const {
  int result = -1;
  for (auto &input_node : input_nodes) {
    if (result == -1
        || (input_node->is_qubit() && input_node->index < result)) {
      result = input_node->index;
    }
  }
  return result;
}

}
