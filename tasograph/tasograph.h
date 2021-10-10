#pragma once

#include "../gate/gate.h"

#include <unordered_map>
#include <set>
#include <map>
#include <chrono>

namespace TASOGraph {

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

struct OpCompare {
  bool operator()(const Op &a, const Op &b) const {
	if (a.guid != b.guid)
	  return a.guid < b.guid;
	return a.ptr < b.ptr;
  };
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

class Graph {
public:
  Graph();
  void remove_edge(Edge e);
  void add_edge(const Op &srcOp, const Op &dstOp, int srcIdx, int dstIdx);
  bool has_edge(const Op &srcOp, const Op &dstOp, int srcIdx, int dstIdx);
  bool has_loop();
  size_t hash();
  bool check_correctness();
  void replace_node(Op oldOp, Op newOp);
  void remove_node(Op oldOp);
  float total_cost() const;
  Graph *optimize(float alpha, int budget, bool print_subst, Context *ctx,
                  const std::string &file_name);

public:
  float totalCost;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges;
};

}; // namespace TASOGraph
