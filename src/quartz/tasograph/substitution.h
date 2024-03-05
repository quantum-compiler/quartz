#pragma once

#include "../context/context.h"
#include "../context/rule_parser.h"
#include "../gate/gate_utils.h"
#include "../parser/qasm_parser.h"
#include "assert.h"
#include "quartz/circuitseq/circuitseq.h"
#include "tasograph.h"

#include <ostream>
#include <queue>

namespace quartz {

class OpX;
class GraphXfer;

struct TensorX {
  // A TensorX represnet an output edge
  TensorX(void) : op(NULL), idx(0) {}
  TensorX(OpX *_op, int _idx) : op(_op), idx(_idx) {}
  Tensor to_edge(const GraphXfer *xfer) const;
  OpX *op;  // The op that outputs this tensor
  int idx;  // The output index of the op
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
  size_t operator()(const TensorX &a) const {
    std::hash<size_t> hash_fn;
    return hash_fn(a.idx) * 17 + hash_fn((size_t)(a.op));
  }
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
  GraphCompare() {
    cost_function_ = [](Graph *graph) { return graph->total_cost(); };
  }
  GraphCompare(const std::function<float(Graph *)> &cost_function)
      : cost_function_(cost_function) {}
  bool operator()(const std::shared_ptr<Graph> &lhs,
                  const std::shared_ptr<Graph> &rhs) {
    return cost_function_(lhs.get()) > cost_function_(rhs.get());
  }

 private:
  std::function<float(Graph *)> cost_function_;
};

class GraphXfer {
 public:
  GraphXfer(Context *src_ctx, Context *dst_ctx, Context *union_ctx);
  GraphXfer(Context *src_ctx, Context *dst_ctx, Context *union_ctx,
            const CircuitSeq *src_graph, const CircuitSeq *dst_graph);
  bool src_graph_connected(CircuitSeq *src_graph);
  TensorX new_tensor(void);
  bool is_input_qubit(const OpX *opx, int idx) const;
  bool is_input_parameter(const OpX *opx, int idx) const;
  bool is_symbolic_input_parameter(const OpX *opx, int idx) const;
  bool is_constant_input_parameter(const OpX *opx, int idx) const;
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
  std::string to_str(std::vector<OpX *> const &v) const;
  std::string src_str() const;
  std::string dst_str() const;
  // TODO: not implemented
  //   std::string to_qasm(std::vector<OpX *> const &v) const;

 public:
  static GraphXfer *create_GraphXfer(Context *_context,
                                     const CircuitSeq *src_graph,
                                     const CircuitSeq *dst_graph,
                                     bool equal_num_input_params = true);
  static GraphXfer *create_GraphXfer_from_qasm_str(Context *_context,
                                                   const std::string &src_str,
                                                   const std::string &dst_str);
  static GraphXfer *
  create_single_gate_GraphXfer(Context *src_ctx, Context *dst_ctx,
                               Context *union_ctx, Command src_cmd,
                               const std::vector<Command> &dst_cmds);
  static std::pair<GraphXfer *, GraphXfer *>
  ccz_cx_rz_xfer(Context *src_ctx, Context *dst_ctx, Context *union_ctx);
  static std::pair<GraphXfer *, GraphXfer *>
  ccz_cx_u1_xfer(Context *src_ctx, Context *dst_ctx, Context *union_ctx);
  static std::pair<GraphXfer *, GraphXfer *>
  ccz_cx_t_xfer(Context *src_ctx, Context *dst_ctx, Context *union_ctx);

 public:
  Context *src_ctx_;
  Context *dst_ctx_;
  Context *union_ctx_;
  int tensorId;
  std::unordered_map<Op, OpX *, OpHash> mappedOps;
  std::unordered_map<int, std::pair<Op, int>> mappedInputs;
  std::unordered_map<TensorX, TensorX, TensorXHash> mappedOutputs;
  std::vector<OpX *> srcOps;
  std::vector<OpX *> dstOps;
  std::unordered_map<int, ParamType> paramValues;
};

}  // namespace quartz
