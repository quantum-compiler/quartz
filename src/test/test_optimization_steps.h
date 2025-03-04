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
    Context *ctx, Context *dst_ctx, const std::string &file_name,
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

  auto graph = std::make_shared<Graph>(ctx, dag);
  std::cout << graph->total_cost() << " gates in circuit before optimizing."
            << std::endl;

  // Context shift, these constructs cannot be inside the if-block
  std::shared_ptr<Graph> graph_before_search;
  Context union_ctx = union_contexts(ctx, dst_ctx);
  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(ctx, dst_ctx, &union_ctx);
  if (dst_ctx != nullptr) {
    auto start = std::chrono::steady_clock::now();
    // Greedy toffoli flip
    graph_before_search = graph->toffoli_flip_greedy(
        GateType::rz, xfer_pair.first, xfer_pair.second,
        store_all_steps_file_prefix);
    //   graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);
    auto end = std::chrono::steady_clock::now();
    graph.swap(graph_before_search);
  }

  bool continue_storing_all_steps = !store_all_steps_file_prefix.empty();
  auto start = std::chrono::steady_clock::now();
  auto new_graph =
      graph->optimize(xfers, graph->total_cost() * 1.05, file_name,
                      /*log_file_name=*/"", /*print_message=*/
                      true, /*cost_function=*/nullptr, timeout,
                      store_all_steps_file_prefix, continue_storing_all_steps);
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
