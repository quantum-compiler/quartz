#pragma once

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <fstream>
#include <iostream>

using namespace quartz;

void test_optimization(
    Context *ctx, const std::string &file_name,
    const std::vector<GraphXfer *> &xfers, double timeout,
    const std::string &store_all_steps_file_prefix = std::string()) {
  QASMParser qasm_parser(ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(file_name, dag)) {
    std::cerr << "Parser failed" << std::endl;
    return;
  }
  // Do not generate (this is only for CircuitSeq::hash())
  // because the number of qubits can be large.
  // ctx->gen_input_and_hashing_dis(dag->get_num_qubits());

  Graph graph(ctx, dag);
  std::cout << graph.total_cost() << " gates in circuit before optimizing."
            << std::endl;
  auto start = std::chrono::steady_clock::now();
  auto new_graph = graph.optimize(xfers, graph.total_cost() * 1.05, file_name,
                                  /*log_file_name=*/"", /*print_message=*/
                                  true, /*cost_function=*/nullptr, timeout,
                                  store_all_steps_file_prefix);
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
