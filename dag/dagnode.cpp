#include "dagnode.h"

bool DAGNode::is_qubit() const {
  return type == internal_qubit || type == input_qubit || type == output_qubit;
}

bool DAGNode::is_parameter() const {
  return type == input_param || type == internal_param;
}
