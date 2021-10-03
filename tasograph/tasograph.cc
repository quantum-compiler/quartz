#include "tasograph.h"
#include "assert.h"

namespace TASOGraph {

enum {
  GUID_INVALID = 0,
  GUID_INPUT = 10,
  GUID_WEIGHT = 11,
  GUID_PRESERVED = 19,
};

Op::Op(void) : guid(GUID_INVALID), ptr(NULL) {}

const Op Op::INVALID_OP = Op();

Graph::Graph() : totalCost(0.0f) {}

void Graph::add_edge(const Op& srcOp,
                     const Op& dstOp,
                     int srcIdx,
                     int dstIdx)
{
  if (inEdges.find(dstOp) == inEdges.end()) {
    inEdges[dstOp];
  }
  if (outEdges.find(srcOp) == outEdges.end()) {
    outEdges[srcOp];
  }
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  inEdges[dstOp].insert(e);
  outEdges[srcOp].insert(e);
}

bool Graph::has_edge(const Op& srcOp,
                     const Op& dstOp,
                     int srcIdx,
                     int dstIdx)
{
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return (inEdges[dstOp].find(e) != inEdges[dstOp].end());
}

Edge::Edge(void)
: srcOp(Op::INVALID_OP), dstOp(Op::INVALID_OP),
  srcIdx(-1), dstIdx(-1)
{}

Edge::Edge(const Op& _srcOp,
           const Op& _dstOp,
           int _srcIdx,
           int _dstIdx)
: srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx)
{}

bool Graph::has_loop(void)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
  return (opList.size() < inEdges.size());
}

bool Graph::check_correctness(void)
{
  bool okay = true;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = outEdges.begin(); it != outEdges.end(); it++) {
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      if (!has_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx)) assert(false);
    }
  }
  return okay;
}

size_t Graph::hash(void)
{
  size_t total = 0;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::unordered_map<size_t, size_t> hash_values;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    size_t my_hash = 17 * 13 + (size_t)it->first.ptr;
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      assert(hash_values.find(e.srcOp.guid) != hash_values.end());
      size_t edge_hash = hash_values[e.srcOp.guid];
      edge_hash = edge_hash * 31 + std::hash<int>()(e.srcIdx);
      edge_hash = edge_hash * 31 + std::hash<int>()(e.dstIdx);
      my_hash = my_hash + edge_hash;
    }
    hash_values[it->first.guid] = my_hash;
    total += my_hash;
  }
  return total;
}

float Graph::total_cost(void) const
{
  size_t cnt = 0;
  for (const auto& it : inEdges) {
    if (it.first.ptr->is_quantum_gate())
      cnt ++;
  }
  return (float)cnt;
}

};
