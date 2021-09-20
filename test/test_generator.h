#pragma once

#include "../generator/generator.h"

#include <chrono>

void test_generator() {
  std::vector<GateType> support_gates =
      {GateType::x, GateType::y, GateType::rx, GateType::ry, GateType::rz,
       GateType::cx};
  Context ctx(support_gates);
  Generator generator(&ctx);
  std::unordered_map<DAGHashType, std::unordered_set<DAG *> > dataset;
  auto start = std::chrono::system_clock::now();
  generator.generate(3, 2, 4, dataset);
  auto end = std::chrono::system_clock::now();
  for (auto &it : dataset) {
    if (it.second.size() <= 1) {
      continue;
    }
    /*std::cout << std::hex << it.first << ":" << std::endl;
    for (auto &dag : it.second) {
      std::cout << dag->to_string();
    }*/
  }
  std::cout << std::dec << "Circuits with " << dataset.size()
            << " different hash values are found in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count() / 1000.0 << " seconds."
            << std::endl;
}
