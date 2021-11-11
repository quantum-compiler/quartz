#pragma once

#include "../gate/gate.h"
#include "../context/context.h"
#include "../dag/dag.h"
#include "../dataset/equivalence_set.h"
#include "../context/rule_parser.h"

#include <unordered_map>
#include <set>
#include <map>
#include <vector>
#include <chrono>
#include <iostream>
#include <queue>
#include <fstream>

namespace TASOGraph {

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

class Graph {
public:
  Graph(Context *ctx);
  Graph(Context *ctx, const DAG &dag);
  Graph(const Graph &graph);
  void add_edge(const Op &srcOp, const Op &dstOp, int srcIdx, int dstIdx);
  bool has_edge(const Op &srcOp, const Op &dstOp, int srcIdx, int dstIdx);
  bool has_loop();
  size_t hash();
  bool check_correctness();
  float total_cost() const;
  size_t get_next_special_op_guid();
  size_t get_special_op_guid();
  void set_special_op_guid(size_t _special_op_guid);
  Graph *context_shift(Context *src_ctx, Context *dst_ctx, Context *union_ctx,
                       RuleParser *rule_parser, bool ignore_toffoli = false);
  Graph *optimize(float alpha, int budget, bool print_subst, Context *ctx,
                  const std::string &equiv_file_name,
                  bool use_simulated_annealing,
		  bool enable_early_stop,
                  bool use_rotation_merging_in_searching,
                  GateType target_rotation);
  void constant_and_rotation_elimination();
  void rotation_merging(GateType target_rotation);
  void to_qasm(const std::string &save_filename, bool print_result,
               bool print_id);
  void draw_circuit(const std::string &qasm_str,
                    const std::string &save_filename);
  size_t get_num_qubits();
  void print_qubit_ops();
  Graph *toffoli_flip_greedy(GateType target_rotation, GraphXfer *xfer,
                             GraphXfer *inverse_xfer);

private:
  void replace_node(Op oldOp, Op newOp);
  void remove_node(Op oldOp);
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

private:
  size_t special_op_guid;

public:
  Context *context;
  float totalCost;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges;
  std::map<Op, ParamType> constant_param_values;
  std::unordered_map<Op, int, OpHash> qubit_2_idx;
};

}; // namespace TASOGraph
