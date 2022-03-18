#pragma once

#include "../context/context.h"
#include "../context/rule_parser.h"
#include "../dag/dag.h"
#include "../gate/gate_utils.h"
#include "assert.h"
#include "tasograph.h"
#include <queue>

namespace quartz {

class OpX;
class GraphXfer;

struct TensorX {
  TensorX(void) : op(NULL), idx(0) {}
  TensorX(OpX *_op, int _idx) : op(_op), idx(_idx) {}
  Tensor to_edge(const GraphXfer *xfer) const;
  OpX *op;
  int idx;
  inline bool operator==(const TensorX &b) const {
    if (op != b.op)
      return false;
    if (idx != b.idx)
      return false;
    return true;
  }
};

struct TensorXCompare {
  bool operator()(const TensorX &a, const TensorX &b) const {
    if (a.op != b.op)
      return a.op < b.op;
    return a.idx < b.idx;
  };
};

class TensorXHash {
public:
  size_t operator()(const TensorX &a) const { return a.idx; }
};

class OpX {
public:
  OpX(const OpX &_op);
  OpX(GateType _type);
  void add_input(const TensorX &input);
  void add_output(const TensorX &output);

public:
  GateType type;
  Op mapOp;
  std::vector<TensorX> inputs, outputs;
};

class GraphCompare {
public:
  bool operator()(std::shared_ptr<Graph> lhs, std::shared_ptr<Graph> rhs) {
    return lhs->total_cost() > rhs->total_cost();
  }
};

class GraphXfer {
public:
  GraphXfer(Context *_context);
  GraphXfer(Context *_context, const DAG *src_graph, const DAG *dst_graph);
  TensorX new_tensor(void);
  bool map_output(const TensorX &src, const TensorX &dst);
  bool can_match(OpX *srcOp, Op op, const Graph *graph) const;
  void match(OpX *srcOp, Op op, const Graph *graph);
  void unmatch(OpX *srcOp, Op op, const Graph *graph);
  void run(int depth, Graph *graph,
           std::vector<std::shared_ptr<Graph>> &new_candidates,
           std::set<size_t> &, float threshold, int maxNumOps,
           bool enable_early_stop, bool &stop_search);
  std::shared_ptr<Graph> run_1_time(int depth, Graph *graph);
  std::shared_ptr<Graph> create_new_graph(const Graph *graph) const;
  bool create_new_operator(const OpX *opx, Op &op);
  int num_src_op();
  int num_dst_op();

public:
  static GraphXfer *create_GraphXfer(Context *_context, const DAG *src_graph,
                                     const DAG *dst_graph);
  static GraphXfer *create_single_gate_GraphXfer(Context *union_ctx,
                                                 Command src_cmd,
                                                 std::vector<Command> dst_cmds);
  static std::pair<GraphXfer *, GraphXfer *> ccz_cx_rz_xfer(Context *ctx);
  static std::pair<GraphXfer *, GraphXfer *> ccz_cx_u1_xfer(Context *ctx);
  static std::pair<GraphXfer *, GraphXfer *> ccz_cx_t_xfer(Context *ctx);

public:
  Context *context;
  int tensorId;
  std::unordered_map<Op, OpX *, OpHash> mappedOps;
  std::unordered_multimap<int, std::pair<Op, int>> mappedInputs;
  std::unordered_map<TensorX, TensorX, TensorXHash> mappedOutputs;
  std::vector<OpX *> srcOps;
  std::vector<OpX *> dstOps;
  std::unordered_map<int, ParamType> paramValues;
};

} // namespace quartz
