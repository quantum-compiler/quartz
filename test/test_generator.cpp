#include "../gate/gate.h"
#include "../dag/dag.h"
#include "../math/vector.h"
#include "../context/context.h"
#include "../generator/generator.h"

#include <iostream>

int main() {
  Context ctx({GateType::rx, GateType::ry, GateType::add});

  std::unordered_map<DAGHashType, std::unordered_set<DAG*> > dataset;
  Generator gen(&ctx);
  gen.generate(1, 2, 2, dataset);

  return 0;
}
