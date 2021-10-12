#pragma once

#include "../tasograph/tasograph.h"
#include "../context/context.h"
#include "../parser/qasm_parser.h"
#include "../dag/dag.h"

#include <fstream>
#include <iostream>

void test_optimization(Context *ctx, const std::string file_name,
                       const std::string equivalent_file_name) {
  QASMParser qasm_parser(ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(file_name, dag)) {
	std::cout << "Parser failed" << std::endl;
	return;
  }

  TASOGraph::Graph graph(ctx, *dag);
  std::cout << graph.total_cost() << std::endl;
  graph.optimize(1.1, 0, false, ctx, equivalent_file_name);
  std::cout << graph.total_cost() << std::endl;
}