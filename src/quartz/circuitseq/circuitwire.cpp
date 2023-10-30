#include "circuitwire.h"

namespace quartz {
bool CircuitWire::is_qubit() const {
  return type == internal_qubit || type == input_qubit || type == output_qubit;
}

bool CircuitWire::is_parameter() const {
  return type == input_param || type == internal_param;
}

std::string CircuitWire::to_string() const {
  if (is_qubit()) {
    return std::string("Q") + std::to_string(index);
  } else {
    return std::string("P") + std::to_string(index);
  }
}

}  // namespace quartz
