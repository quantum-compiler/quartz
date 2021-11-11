#pragma once

#include "../tasograph/tasograph.h"
#include "../context/context.h"
#include "../parser/qasm_parser.h"
#include "../dag/dag.h"
#include "../tasograph/substitution.h"

#include <fstream>
#include <iostream>

void test_optimization(Context *ctx, const std::string &file_name,
                       const std::string &equivalent_file_name,
                       bool use_simulated_annealing) {
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
  auto new_graph = graph.optimize(1.1, 0, false, ctx, equivalent_file_name,
                                  use_simulated_annealing, false/*early_stop*/,
				  false/*rotation_merging*/,
                                  GateType::rz /*Just a placeholder*/);
  auto end = std::chrono::steady_clock::now();
  std::cout << "After optimizing graph in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds, "
            << "total gate count becomes " << new_graph->total_cost() << "."
            << std::endl;
}
