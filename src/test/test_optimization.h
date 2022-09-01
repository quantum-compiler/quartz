#pragma once

#include "quartz/context/context.h"
#include "quartz/dag/dag.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <fstream>
#include <iostream>

using namespace quartz;

void test_optimization(Context *ctx, const std::string &file_name,
                       const std::string &equivalent_file_name,
                       bool use_simulated_annealing) {
  QASMParser qasm_parser(ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(file_name, dag)) {
    std::cerr << "Parser failed" << std::endl;
    return;
  }
  ctx->get_and_gen_input_dis(dag->get_num_qubits());
  ctx->get_and_gen_hashing_dis(dag->get_num_qubits());
  ctx->get_and_gen_parameters(dag->get_num_input_parameters());

  quartz::Graph graph(ctx, dag);
  std::cout << graph.total_cost() << " gates in circuit before optimizing."
            << std::endl;
  auto start = std::chrono::steady_clock::now();
  auto new_graph =
      graph.optimize(ctx, equivalent_file_name, file_name, /*print_message=*/
                     true);
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
