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
  std::cout << graph.total_cost() << " gates in circuit before optimizing."
            << std::endl;
  auto start = std::chrono::steady_clock::now();
  graph.optimize(1.1, 0, false, ctx, equivalent_file_name);
  auto end = std::chrono::steady_clock::now();
  std::cout << "After optimizing graph in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds, "
            << "total gate count becomes " << graph.total_cost() << "."
            << std::endl;
}