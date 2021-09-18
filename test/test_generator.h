#pragma once

#include "../generator/generator.h"

void test_generator() {
  std::vector<GateType> support_gates =
      {GateType::x, GateType::y, GateType::rx, GateType::ry, GateType::rz};
  Context ctx(support_gates);
  Generator generator(&ctx);
  std::unordered_map<DAGHashType, std::unordered_set<DAG*> > dataset;
  generator.generate(2, 2, 3, dataset);
  for (auto &it : dataset) {
    std::cout << std::hex << it.first << ":" << std::endl;
    for (auto &dag : it.second) {
      std::cout << dag->to_string();
    }
  }
}
