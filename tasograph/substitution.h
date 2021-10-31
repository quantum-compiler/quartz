#pragma once

#include "tasograph.h"
#include "../gate/gate_utils.h"
#include "../dag/dag.h"
#include "../context/context.h"
#include "assert.h"
#include "../gate/gate_utils.h"
#include "../context/rule_parser.h"
#include <queue>

namespace TASOGraph {

class OpX;
class GraphXfer;

struct TensorX {
  TensorX(void) : op(NULL), idx(0) {}
  TensorX(OpX *_op, int _idx) : op(_op), idx(_idx) {}
  Tensor to_edge(const GraphXfer *xfer) const;
  OpX *op;
  int idx;
};

struct TensorXCompare {
  bool operator()(const TensorX &a, const TensorX &b) const {
	if (a.op != b.op)
	  return a.op < b.op;
	return a.idx < b.idx;
  };
};

class OpX {
public:
  OpX(const OpX &_op);
  OpX(::GateType _type);
  void add_input(const TensorX &input);
  void add_output(const TensorX &output);

public:
  ::GateType type;
  Op mapOp;
  std::vector<TensorX> inputs, outputs;
};

class GraphCompare {
public:
  bool operator()(Graph *lhs, Graph *rhs) {
	return lhs->total_cost() > rhs->total_cost();
  }
};

class GraphXfer {
public:
  GraphXfer(::Context *_context);
  GraphXfer(::Context *_context, const ::DAG *src_graph,
            const ::DAG *dst_graph);
  TensorX new_tensor(void);
  bool map_output(const TensorX &src, const TensorX &dst);
  bool can_match(OpX *srcOp, Op op, Graph *graph);
  void match(OpX *srcOp, Op op, Graph *graph);
  void unmatch(OpX *srcOp, Op op, Graph *graph);
  void run(int depth, Graph *graph,
           std::vector<Graph *> &new_candidates,
           std::set<size_t> &, float threshold, int maxNumOps);
  Graph *run_1_time(int depth, Graph *graph);
  Graph *create_new_graph(Graph *graph);
  bool create_new_operator(const OpX *opx, Op &op);

public:
  static GraphXfer *create_GraphXfer(::Context *_context,
                                     const ::DAG *src_graph,
                                     const ::DAG *dst_graph);
  static GraphXfer *create_single_gate_GraphXfer(Command src_cmd,
                                                 Context *dst_ctx,
                                                 std::vector<Command> dst_cmds);

public:
  ::Context *context;
  int tensorId;
  std::map<Op, OpX *, OpCompare> mappedOps;
  std::multimap<int, std::pair<Op, int>> mappedInputs;
  std::map<TensorX, TensorX, TensorXCompare> mappedOutputs;
  std::vector<OpX *> srcOps;
  std::vector<OpX *> dstOps;
  std::unordered_map<int, ParamType> paramValues;
};

} // namespace TASOGraph
