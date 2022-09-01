#include "dagnode.h"

namespace quartz {
bool DAGNode::is_qubit() const {
  return type == internal_qubit || type == input_qubit || type == output_qubit;
}

bool DAGNode::is_parameter() const {
  return type == input_param || type == internal_param;
}

std::string DAGNode::to_string() const {
  if (is_qubit()) {
    return std::string("Q") + std::to_string(index);
  } else {
    return std::string("P") + std::to_string(index);
  }
}

} // namespace quartz
