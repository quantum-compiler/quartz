#pragma once

#include "../circuitseq/circuitseq.h"
#include "../context/context.h"
#include "../context/rule_parser.h"
#include "../dataset/equivalence_set.h"
#include "../gate/gate.h"
#include "../parser/qasm_parser.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>

namespace quartz {

#define eps 1e-6

bool equal_to_2k_pi(double d);

class Op {
 public:
  Op(void);
  Op(size_t _guid, Gate *_ptr) : guid(_guid), ptr(_ptr) {}
  inline bool operator==(const Op &b) const {
    if (guid != b.guid)
      return false;
    if (ptr != b.ptr)
      return false;
    return true;
  }
  inline bool operator!=(const Op &b) const {
    if (guid != b.guid)
      return true;
    if (ptr != b.ptr)
      return true;
    return false;
  }
  inline bool operator<(const Op &b) const {
    if (guid != b.guid)
      return guid < b.guid;
    if (ptr != b.ptr)
      return ptr < b.ptr;
    return false;
  }
  inline bool operator>(const Op &b) const {
    if (guid != b.guid)
      return guid > b.guid;
    if (ptr != b.ptr)
      return ptr > b.ptr;
    return false;
  }
  Op &operator=(const Op &op) {
    guid = op.guid;
    ptr = op.ptr;
    return *this;
  }
  static const Op INVALID_OP;

 public:
  size_t guid;
  Gate *ptr;
};

class OpCompare {
 public:
  bool operator()(const Op &a, const Op &b) const {
    if (a.guid != b.guid)
      return a.guid < b.guid;
    return a.ptr < b.ptr;
  };
};

class OpHash {
 public:
  size_t operator()(const Op &a) const {
    std::hash<size_t> hash_fn;
    return hash_fn(a.guid) * 17 + hash_fn((size_t)(a.ptr));
  }
};

class Pos {
 public:
  Pos() {
    op = Op();
    idx = 0;
  }
  Pos(const Pos &b) {
    op = b.op;
    idx = b.idx;
  }
  inline bool operator<(const Pos &b) const {
    if (op != b.op)
      return op < b.op;
    if (idx != b.idx)
      return idx < b.idx;
    return false;
  }
  Pos &operator=(const Pos &pos) {
    op = pos.op;
    idx = pos.idx;
    return *this;
  }
  Pos(Op op_, int idx_) : op(op_), idx(idx_) {}
  Op op;
  int idx;
};

inline bool operator==(const Pos &a, const Pos &b) {
  if (a.op != b.op)
    return false;
  if (a.idx != b.idx)
    return false;
  return true;
}
inline bool operator!=(const Pos &a, const Pos &b) {
  if (a.op != b.op)
    return true;
  if (a.idx != b.idx)
    return true;
  return false;
}

class PosHash {
 public:
  size_t operator()(const Pos &a) const {
    std::hash<size_t> hash_fn;
    OpHash op_hash;
    return op_hash(a.op) * 17 + hash_fn(a.idx);
  }
};

class PosCompare {
 public:
  bool operator()(const Pos &a, const Pos &b) const {
    if (a.op != b.op)
      return a.op < b.op;
    return a.idx < b.idx;
  }
};

class Tensor {
 public:
  Tensor(void);
  int idx;
  Op op;
};

struct Edge {
  Edge(void);
  Edge(const Op &_srcOp, const Op &_dstOp, int _srcIdx, int _dstIdx);
  Op srcOp, dstOp;
  int srcIdx, dstIdx;
};

struct EdgeCompare {
  bool operator()(const Edge &a, const Edge &b) const {
    if (!(a.srcOp == b.srcOp))
      return a.srcOp < b.srcOp;
    if (!(a.dstOp == b.dstOp))
      return a.dstOp < b.dstOp;
    if (a.srcIdx != b.srcIdx)
      return a.srcIdx < b.srcIdx;
    if (a.dstIdx != b.dstIdx)
      return a.dstIdx < b.dstIdx;
    return false;
  };
};

class GraphXfer;
class OpX;

class Graph {
 public:
  Graph(Context *ctx);
  Graph(Context *ctx, const CircuitSeq *seq);
  Graph(const Graph &graph);
  [[nodiscard]] std::unique_ptr<CircuitSeq> to_circuit_sequence() const;
  void _construct_pos_2_logical_qubit();
  void add_edge(const Op &srcOp, const Op &dstOp, int srcIdx, int dstIdx);
  bool has_edge(const Op &srcOp, const Op &dstOp, int srcIdx, int dstIdx) const;
  Op add_qubit(int qubit_idx);
  Op add_parameter(const ParamType p);
  Op new_gate(GateType gt);
  bool has_loop() const;
  size_t hash();
  bool equal(const Graph &other) const;
  bool check_correctness();
  int specific_gate_count(GateType gate_type) const;
  [[nodiscard]] float total_cost() const;
  [[nodiscard]] int gate_count() const;
  [[nodiscard]] int circuit_depth() const;
  size_t get_next_special_op_guid();
  size_t get_special_op_guid();
  void set_special_op_guid(size_t _special_op_guid);
  std::shared_ptr<Graph> context_shift(Context *src_ctx, Context *dst_ctx,
                                       Context *union_ctx,
                                       RuleParser *rule_parser,
                                       bool ignore_toffoli = false);
  /**
   * Greedily apply transformations towards lower cost.
   * @param ctx The context variable.
   * @param equiv_file_name The ECC set file name.
   * @param print_message To output the log to the screen or not.
   * @param cost_function The cost function.
   * @param store_all_steps_file_prefix If not empty, store each circuit
   * transformation step in a file with the corresponding file prefix.
   * @return The optimized circuit.
   */
  std::shared_ptr<Graph> greedy_optimize(
      Context *ctx, const std::string &equiv_file_name, bool print_message,
      std::function<float(Graph *)> cost_function = nullptr,
      const std::string &store_all_steps_file_prefix = std::string());
  std::shared_ptr<Graph>
  greedy_optimize_with_xfer(const std::vector<GraphXfer *> &xfers,
                            bool print_message,
                            std::function<float(Graph *)> cost_function);
  std::shared_ptr<Graph>
  optimize_legacy(float alpha, int budget, bool print_subst, Context *ctx,
                  const std::string &equiv_file_name,
                  bool use_simulated_annealing, bool enable_early_stop,
                  bool use_rotation_merging_in_searching,
                  GateType target_rotation, std::string circuit_name = "",
                  int timeout = 86400 /*1 day*/);

  /**
   * Optimize this circuit with a greedy phase at the beginning.
   * @param ctx The context variable.
   * @param equiv_file_name The ECC set file name.
   * @param circuit_name The circuit name shown in the log.
   * @param print_message To output the log or not. The log will be outputted
   * to a file with name combined by the ECC set file name and the circuit
   * file name.
   * @param cost_function The cost function for the search.
   * @param cost_upper_bound The maximum cost of the circuits to be searched.
   * @param timeout Timeout in seconds, for the search phase.
   * @param store_all_steps_file_prefix If not empty, store each circuit
   * transformation step in a file with the corresponding file prefix.
   * @return The optimized circuit.
   */
  std::shared_ptr<Graph>
  optimize(Context *ctx, const std::string &equiv_file_name,
           const std::string &circuit_name, bool print_message,
           std::function<float(Graph *)> cost_function = nullptr,
           double cost_upper_bound = -1 /*default = current cost * 1.05*/,
           double timeout = 3600 /*1 hour*/,
           const std::string &store_all_steps_file_prefix = std::string());
  /**
   * Optimize this circuit without a greedy phase.
   * @param xfers The circuit transformations.
   * @param cost_upper_bound The maximum cost of the circuits to be searched.
   * @param circuit_name The circuit name shown in the log.
   * @param log_file_name The file name to output the log. If empty, the log
   * will be outputted to the screen.
   * @param print_message To output the log or not.
   * @param cost_function The cost function for the search.
   * @param timeout Timeout in seconds.
   * @param store_all_steps_file_prefix If not empty, store each circuit
   * transformation step in a file with the corresponding file prefix.
   * @param continue_storing_all_steps If true, there was a greedy phase
   * before calling this function with the same |store_all_steps_file_prefix|.
   * We should continue the numbering in this case.
   * @return The optimized circuit.
   */
  std::shared_ptr<Graph>
  optimize(const std::vector<GraphXfer *> &xfers, double cost_upper_bound,
           const std::string &circuit_name, const std::string &log_file_name,
           bool print_message,
           std::function<float(Graph *)> cost_function = nullptr,
           double timeout = 3600 /*1 hour*/,
           const std::string &store_all_steps_file_prefix = std::string(),
           bool continue_storing_all_steps = false);
  void constant_and_rotation_elimination();
  void rotation_merging(GateType target_rotation);
  std::string to_qasm(bool print_result = false, bool print_guid = false) const;
  void to_qasm(const std::string &save_filename, bool print_result,
               bool print_guid) const;
  template <class _CharT, class _Traits>
  static std::shared_ptr<Graph>
  _from_qasm_stream(Context *ctx,
                    std::basic_istream<_CharT, _Traits> &qasm_stream);
  static std::shared_ptr<Graph> from_qasm_file(Context *ctx,
                                               const std::string &filename);
  static std::shared_ptr<Graph> from_qasm_str(Context *ctx,
                                              const std::string qasm_str);
  void draw_circuit(const std::string &qasm_str,
                    const std::string &save_filename);
  size_t get_num_qubits() const;
  void print_qubit_ops();
  std::shared_ptr<Graph> toffoli_flip_greedy(GateType target_rotation,
                                             GraphXfer *xfer,
                                             GraphXfer *inverse_xfer);
  void toffoli_flip_greedy_with_trace(GateType target_rotation, GraphXfer *xfer,
                                      GraphXfer *inverse_xfer,
                                      std::vector<int> &trace);
  std::shared_ptr<Graph>
  toffoli_flip_by_instruction(GateType target_rotation, GraphXfer *xfer,
                              GraphXfer *inverse_xfer,
                              std::vector<int> instruction);
  std::vector<size_t> appliable_xfers(Op op,
                                      const std::vector<GraphXfer *> &) const;
  std::vector<size_t>
  appliable_xfers_parallel(Op op, const std::vector<GraphXfer *> &) const;
  bool xfer_appliable(GraphXfer *xfer, Op op) const;
  std::shared_ptr<Graph> apply_xfer(GraphXfer *xfer, Op op,
                                    bool eliminate_rotation = false) const;
  std::pair<std::shared_ptr<Graph>, std::vector<int>>
  apply_xfer_and_track_node(GraphXfer *xfer, Op op,
                            bool eliminate_rotation = false,
                            int predecessor_layers = 1) const;
  void all_ops(std::vector<Op> &ops);
  void all_edges(std::vector<Edge> &edges);
  void topology_order_ops(std::vector<Op> &ops) const;
  void remove_node(Op oldOp);
  void remove_node_wo_input_output_connect(Op oldOp);
  std::shared_ptr<Graph> ccz_flip_t(Context *ctx);
  std::shared_ptr<Graph> ccz_flip_greedy_rz();
  std::shared_ptr<Graph> ccz_flip_greedy_u1();
  bool _loop_check_after_matching(GraphXfer *xfer) const;
  std::shared_ptr<Graph>
  subgraph(const std::unordered_set<Op, OpHash> &ops) const;
  std::vector<std::shared_ptr<Graph>>
  topology_partition(const int partition_gate_count) const;
  /**
   * Return the parameter value if the Op is a constant parameter,
   * or return 0 otherwise.
   */
  ParamType get_param_value(const Op &op) const;
  bool param_has_value(const Op &op) const;

 private:
  void replace_node(Op oldOp, Op newOp);
  void remove_edge(Op srcOp, Op dstOp);
  uint64_t xor_bitmap(uint64_t src_bitmap, int src_idx, uint64_t dst_bitmap,
                      int dst_idx);
  void explore(Pos pos, bool left, std::unordered_set<Pos, PosHash> &covered);
  void expand(Pos pos, bool left, GateType target_rotation,
              std::unordered_set<Pos, PosHash> &covered,
              std::unordered_map<int, Pos> &anchor_point,
              std::unordered_map<Pos, int, PosHash> pos_to_qubits,
              std::queue<int> &todo_qubits);
  void remove(Pos pos, bool left, std::unordered_set<Pos, PosHash> &covered);
  bool moveable(GateType tp);
  bool move_forward(Pos &pos, bool left);
  bool merge_2_rotation_op(Op op_0, Op op_1);
  // The common core part of the API xfer_appliable, apply_xfer, and
  // apply_xfer_and_track_node. Matches the src dag of xfer to the local dag
  // in the circuit whose topological-order root is op. If failed, it
  // automatically unmaps the matched nodes. Otherwise, the caller should
  // unmap the matched nodes after their work is done.
  bool _pattern_matching(
      GraphXfer *xfer, Op op,
      std::deque<std::pair<OpX *, Op>> &matched_opx_op_pairs_dq) const;

 public:
  size_t special_op_guid;
  Context *context;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges;
  std::unordered_map<Op, int, OpHash> input_qubit_op_2_qubit_idx;
  std::unordered_map<Pos, int, PosHash> pos_2_logical_qubit;
  std::unordered_map<Op, int, OpHash> param_idx;
};

}  // namespace quartz
