#pragma once

#include "quartz/context/rule_parser.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

void test_context_shift(const std::string &filename, Context *src_ctx,
                        Context *dst_ctx, Context *union_ctx,
                        RuleParser *rule_parser) {
  QASMParser qasm_parser(src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(filename, dag)) {
    std::cout << "Parser failed" << std::endl;
    return;
  }

  Graph graph(src_ctx, dag);
  auto graph_new_ctx =
      graph.context_shift(src_ctx, dst_ctx, union_ctx, rule_parser);
  std::cout << graph_new_ctx->to_qasm() << std::endl;
  // graph_new_ctx->constant_eliminate();
  // for (auto it = graph_new_ctx->inEdges.begin();
  //     it != graph_new_ctx->inEdges.end(); ++it) {
  //	std::cout << gate_type_name(it->first.ptr->tp) << std::endl;
  //}
  // std::cout << graph_new_ctx->total_cost() << std::endl;
}
