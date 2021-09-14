#pragma once

#include "../gate/gate_utils.h"

#include <unordered_map>
#include <vector>
#include <memory>

class Context {
 public:
  explicit Context(const std::vector<GateType> &supported_gates);
  Gate *get_gate(GateType tp);
  [[nodiscard]] const std::vector<GateType> &get_supported_gates() const;

 private:
  bool insert_gate(GateType tp);

  std::unordered_map<GateType, std::unique_ptr<Gate>> gates_;
  std::vector<GateType> supported_gates_;
};
