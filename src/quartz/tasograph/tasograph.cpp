#include "tasograph.h"
#include "substitution.h"
#include "../sabre/sabre.h"
#include "../game/game_utils.h"

#include <cassert>
#include <iomanip>
#include <random>
#include <algorithm>
#include <iterator>
#include <vector>

namespace quartz {

// TODO: GUID_PRESERVED depends on the global guid in Context class, need to
// modify
enum {
  GUID_INVALID = 0,
  GUID_INPUT = 10,
  GUID_WEIGHT = 11,
  GUID_PRESERVED = 16383
};

bool equal_to_2k_pi(double d) {
  d = std::abs(d);
  int m = d / (2 * PI);
  if (std::abs(d - m * 2 * PI) > eps && std::abs(d - (m + 1) * 2 * PI) > eps)
    return false;
  return true;
}

Op::Op(void) : guid(GUID_INVALID), ptr(NULL) {}

const Op Op::INVALID_OP = Op();

void Graph::_construct_pos_2_logical_qubit() {
  pos_2_logical_qubit.clear();
  // Construct pos_2_logical_qubit
  std::unordered_map<Op, int, OpHash> op_in_degree;
  std::queue<Op> op_q;
  for (auto it = outEdges.cbegin(); it != outEdges.cend(); ++it) {
    if (it->first.ptr->tp == GateType::input_qubit ||
        it->first.ptr->tp == GateType::input_param) {
      op_q.push(it->first);
    }
  }

  for (auto it = inEdges.cbegin(); it != inEdges.cend(); ++it) {
    op_in_degree[it->first] = it->second.size();
  }

  while (!op_q.empty()) {
    auto op = op_q.front();
    op_q.pop();

    // An input qubit
    if (input_qubit_op_2_qubit_idx.find(op) !=
        input_qubit_op_2_qubit_idx.end()) {
      pos_2_logical_qubit[Pos(op, 0)] = input_qubit_op_2_qubit_idx[op];
    }
    if (outEdges.find(op) != outEdges.end()) {
      auto op_out_edges = outEdges.find(op)->second;
      for (auto e_it = op_out_edges.cbegin(); e_it != op_out_edges.cend();
           ++e_it) {
        if (pos_2_logical_qubit.find(Pos(e_it->srcOp, e_it->srcIdx)) !=
            pos_2_logical_qubit.end()) {
          pos_2_logical_qubit[Pos(e_it->dstOp, e_it->dstIdx)] =
              pos_2_logical_qubit[Pos(e_it->srcOp, e_it->srcIdx)];
        }
        assert(op_in_degree[e_it->dstOp] > 0);
        op_in_degree[e_it->dstOp]--;
        if (op_in_degree[e_it->dstOp] == 0) {
          op_q.push(e_it->dstOp);
        }
      }
    }
  }
}

Graph::Graph(Context *ctx) : context(ctx), special_op_guid(0) {}

Graph::Graph(Context *ctx, const DAG *dag) : context(ctx), special_op_guid(0) {
  // Guid for input qubit and input parameter nodes
  int num_input_qubits = dag->get_num_qubits();
  int num_input_params = dag->get_num_input_parameters();
  // Currently only 16383 vacant guid
  assert(num_input_qubits + num_input_params <= GUID_PRESERVED);
  std::vector<Op> input_qubits_op;
  std::vector<Op> input_params_op;
  input_qubits_op.reserve(num_input_qubits);
  input_params_op.reserve(num_input_params);
  for (auto &node : dag->nodes) {
    if (node->type == DAGNode::input_qubit) {
      auto input_qubit_op =
          Op(get_next_special_op_guid(), ctx->get_gate(GateType::input_qubit));
      input_qubits_op.push_back(input_qubit_op);
      input_qubit_op_2_qubit_idx[input_qubit_op] = node->index;
    } else if (node->type == DAGNode::input_param) {
      input_params_op.push_back(Op(context->next_global_unique_id(),
                                   ctx->get_gate(GateType::input_param)));
    }
  }

  // Map all edges in dag to Op
  std::map<DAGHyperEdge *, Op> edge_2_op;
  for (auto &edge : dag->edges) {
    auto e = edge.get();
    if (edge_2_op.find(e) == edge_2_op.end()) {
      Op op(ctx->next_global_unique_id(), edge->gate);
      edge_2_op[e] = op;
    }
  }

  for (auto &node : dag->nodes) {
    size_t srcIdx = -1; // Assumption: a node can have at most 1 input
    Op srcOp;
    if (node->type == DAGNode::input_qubit) {
      srcOp = input_qubits_op[node->index];
      srcIdx = 0;
    } else if (node->type == DAGNode::input_param) {
      srcOp = input_params_op[node->index];
      srcIdx = 0;
    } else {
      assert(node->input_edges.size() == 1); // A node can have at most 1 input
      auto input_edge = node->input_edges[0];
      bool found = false;
      for (srcIdx = 0; srcIdx < input_edge->output_nodes.size(); ++srcIdx) {
        if (node.get() == input_edge->output_nodes[srcIdx]) {
          found = true;
          break;
        }
      }
      assert(found);
      assert(edge_2_op.find(input_edge) != edge_2_op.end());
      srcOp = edge_2_op[input_edge];
    }

    assert(srcIdx >= 0);
    assert(srcOp != Op::INVALID_OP);

    for (auto output_edge : node->output_edges) {
      size_t dstIdx;
      bool found = false;
      for (dstIdx = 0; dstIdx < output_edge->input_nodes.size(); ++dstIdx) {
        if (node.get() == output_edge->input_nodes[dstIdx]) {
          found = true;
          break;
        }
      }
      assert(found);
      assert(edge_2_op.find(output_edge) != edge_2_op.end());
      auto dstOp = edge_2_op[output_edge];

      add_edge(srcOp, dstOp, srcIdx, dstIdx);
    }
  }

  _construct_pos_2_logical_qubit();
}

Graph::Graph(const Graph &graph) {
  context = graph.context;
  constant_param_values = graph.constant_param_values;
  special_op_guid = graph.special_op_guid;
  input_qubit_op_2_qubit_idx = graph.input_qubit_op_2_qubit_idx;
  pos_2_logical_qubit = graph.pos_2_logical_qubit;
  inEdges = graph.inEdges;
  outEdges = graph.outEdges;
  qubit_mapping_table = graph.qubit_mapping_table;
}

size_t Graph::get_next_special_op_guid() {
  if (special_op_guid >= GUID_PRESERVED) {
    std::cerr << "Run out of special guid." << std::endl;
    assert(false);
  }
  return special_op_guid++;
}

size_t Graph::get_special_op_guid() { return special_op_guid; }

void Graph::set_special_op_guid(size_t _special_op_guid) {
  special_op_guid = _special_op_guid;
}

void Graph::add_edge(const Op &srcOp, const Op &dstOp, int srcIdx, int dstIdx) {
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

bool Graph::has_edge(const Op &srcOp, const Op &dstOp, int srcIdx,
                     int dstIdx) const {
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return (inEdges.find(dstOp)->second.find(e) !=
          inEdges.find(dstOp)->second.end());
}

Op Graph::add_qubit(int qubit_idx) {
  Gate *gate = context->get_gate(GateType::input_qubit);
  auto guid = get_next_special_op_guid();
  Op op(guid, gate);
  input_qubit_op_2_qubit_idx[op] = qubit_idx;
  return op;
}

Op Graph::add_parameter(const ParamType p) {
  Gate *gate = context->get_gate(GateType::input_param);
  auto guid = context->next_global_unique_id();
  Op op(guid, gate);
  constant_param_values[op] = p;
  return op;
}

Op Graph::new_gate(GateType gt) {
  Gate *gate = context->get_gate(gt);
  auto guid = context->next_global_unique_id();
  Op op(guid, gate);
  return op;
}

Edge::Edge(void)
    : srcOp(Op::INVALID_OP), dstOp(Op::INVALID_OP), srcIdx(-1), dstIdx(-1),
      logical_qubit_idx(-1), physical_qubit_idx(-1){}

Edge::Edge(const Op &_srcOp, const Op &_dstOp, int _srcIdx, int _dstIdx)
    : srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx),
      logical_qubit_idx(-1), physical_qubit_idx(-1){}

Edge::Edge(const Op &_srcOp, const Op &_dstOp, int _srcIdx, int _dstIdx,
           int _logical_qubit_idx, int _physical_qubit_idx)
    : srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx),
      logical_qubit_idx(_logical_qubit_idx), physical_qubit_idx(_physical_qubit_idx) {}

bool Graph::has_loop(void) const {
  int done_ops_cnt = 0;
  std::unordered_map<Op, int, OpHash> op_in_degree;
  std::queue<Op> op_q;
  for (auto it = outEdges.cbegin(); it != outEdges.cend(); ++it) {
    if (it->first.ptr->tp == GateType::input_qubit ||
        it->first.ptr->tp == GateType::input_param) {
      op_q.push(it->first);
    }
  }

  for (auto it = inEdges.cbegin(); it != inEdges.cend(); ++it) {
    op_in_degree[it->first] = it->second.size();
  }

  while (!op_q.empty()) {
    auto op = op_q.front();
    op_q.pop();
    if (outEdges.find(op) != outEdges.end()) {
      auto op_out_edges = outEdges.find(op)->second;
      for (auto e_it = op_out_edges.cbegin(); e_it != op_out_edges.cend();
           ++e_it) {
        assert(op_in_degree[e_it->dstOp] > 0);
        op_in_degree[e_it->dstOp]--;
        if (op_in_degree[e_it->dstOp] == 0) {
          done_ops_cnt++;
          op_q.push(e_it->dstOp);
        }
      }
    }
  }
  // Return directly for better performance
  return done_ops_cnt != gate_count();
  // Debug information
  //   if (done_ops_cnt == gate_count())
  //     return false;

  //   int cnt = 0;
  //   for (const auto &it : op_in_degree) {
  //     if (it.second != 0) {
  //       std::cout << gate_type_name(it.first.ptr->tp) << "(" << it.first.guid
  //                 << ")" << it.second << std::endl;
  //       cnt++;
  //     }
  //   }
  //   std::cout << cnt << std::endl;
  //   return true;
}

bool Graph::check_correctness(void) {
  bool okay = true;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = outEdges.begin(); it != outEdges.end(); it++) {
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      if (!has_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx))
        assert(false);
    }
  }
  return okay;
}

// TODO: add constant parameters
size_t Graph::hash(void) {
  size_t total = 0;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::unordered_map<size_t, size_t> hash_values;
  std::queue<Op> op_queue;
  // Compute the hash value for input ops
  for (it = outEdges.begin(); it != outEdges.end(); it++) {
    if (it->first.ptr->tp == GateType::input_qubit ||
        it->first.ptr->tp == GateType::input_param) {
      size_t my_hash = 17 * 13 + (size_t)it->first.ptr;
      hash_values[it->first.guid] = my_hash;
      total += my_hash;
      op_queue.push(it->first);
    }
  }

  // Construct in-degree map
  std::map<Op, size_t> op_in_edges_cnt;
  for (it = inEdges.begin(); it != inEdges.end(); ++it) {
    op_in_edges_cnt[it->first] = it->second.size();
  }

  while (!op_queue.empty()) {
    auto op = op_queue.front();
    op_queue.pop();
    if (hash_values.find(op.guid) == hash_values.end()) {
      std::set<Edge, EdgeCompare> list = inEdges[op];
      std::set<Edge, EdgeCompare>::const_iterator it2;
      size_t my_hash = 17 * 13 + (size_t)op.ptr;
      for (it2 = list.begin(); it2 != list.end(); it2++) {
        Edge e = *it2;
        assert(hash_values.find(e.srcOp.guid) != hash_values.end());
        auto edge_hash = hash_values[e.srcOp.guid];
        edge_hash = edge_hash * 31 + std::hash<int>()(e.srcIdx);
        edge_hash = edge_hash * 31 + std::hash<int>()(e.dstIdx);
        my_hash = my_hash + edge_hash;
      }
      hash_values[op.guid] = my_hash;
      total += my_hash;
    }
    if (outEdges.find(op) != outEdges.end()) {
      std::set<Edge, EdgeCompare> list = outEdges[op];
      std::set<Edge, EdgeCompare>::const_iterator it2;
      for (it2 = list.begin(); it2 != list.end(); it2++) {
        auto e = *it2;
        op_in_edges_cnt[e.dstOp]--;
        if (op_in_edges_cnt[e.dstOp] == 0) {
          op_queue.push(e.dstOp);
        }
      }
    }
  }
  return total;
}

std::shared_ptr<Graph> Graph::context_shift(Context *src_ctx, Context *dst_ctx,
                                            Context *union_ctx,
                                            RuleParser *rule_parser,
                                            bool ignore_toffoli) {
  auto src_gates = src_ctx->get_supported_gates();
  auto dst_gate_set = std::set<GateType>(dst_ctx->get_supported_gates().begin(),
                                         dst_ctx->get_supported_gates().end());
  std::map<GateType, GraphXfer *> tp_2_xfer;
  for (auto gate_tp : src_gates) {
    if (ignore_toffoli && src_ctx->get_gate(gate_tp)->is_toffoli_gate())
      continue;
    if (dst_gate_set.find(gate_tp) == dst_gate_set.end()) {
      std::vector<Command> cmds;
      Command src_cmd;
      assert(
          rule_parser->find_convert_commands(dst_ctx, gate_tp, src_cmd, cmds));

      tp_2_xfer[gate_tp] =
          GraphXfer::create_single_gate_GraphXfer(union_ctx, src_cmd, cmds);
    }
  }
  std::shared_ptr<Graph> src_graph(new Graph(*this));
  std::shared_ptr<Graph> dst_graph(nullptr);
  for (auto it = tp_2_xfer.begin(); it != tp_2_xfer.end(); ++it) {
    while ((dst_graph = it->second->run_1_time(0, src_graph.get())) !=
           nullptr) {
      src_graph = dst_graph;
    }
  }
  return src_graph;
}

float Graph::total_cost(void) const {
  // Uncomment to use circuit depth as the cost
  // return circuit_depth();
  size_t cnt = 0;
  for (const auto &it : inEdges) {
    if (it.first.ptr->is_quantum_gate())
      cnt++;
  }
  return (float)cnt;
}

int Graph::gate_count() const {
  int cnt = 0;
  for (const auto &it : inEdges) {
    if (it.first.ptr->tp != GateType::input_qubit &&
        it.first.ptr->tp != GateType::input_param)
      cnt++;
  }
  return cnt;
}

int Graph::specific_gate_count(GateType gate_type) const {
  int cnt = 0;
  for (const auto &it : inEdges) {
    if (it.first.ptr->tp == gate_type)
      cnt++;
  }
  return cnt;
}

int Graph::circuit_depth() const {
  std::unordered_map<Op, std::vector<int>, OpHash> op_2_qubit_idx;
  std::vector<int> depth(get_num_qubits(), 0);
  std::queue<Op> gates;
  std::unordered_map<Op, int, OpHash> gate_indegree;
  for (const auto &it: outEdges) {
    if (it.first.ptr->tp == GateType::input_qubit) {
      auto idx = input_qubit_op_2_qubit_idx.find(it.first);
      op_2_qubit_idx[it.first] = std::vector<int>(1, idx->second);
      gates.push(it.first);
    }
  }
  while (!gates.empty()) {
    const auto &gate = gates.front();
    gates.pop();
    if (outEdges.count(gate) == 0) {
      continue;
    }
    for (auto &edge : outEdges.find(gate)->second) {
      if (gate_indegree.count(edge.dstOp) == 0) {
        gate_indegree[edge.dstOp] = edge.dstOp.ptr->num_qubits;
        op_2_qubit_idx[edge.dstOp] = std::vector<int>(edge.dstOp.ptr->num_qubits, 0);
      }
      gate_indegree[edge.dstOp]--;
      op_2_qubit_idx[edge.dstOp][edge.dstIdx] = op_2_qubit_idx[edge.srcOp][edge.srcIdx];
      if (!gate_indegree[edge.dstOp]) {
        // Append the gate
        int max_previous_depth = 0;
        for (auto &idx: op_2_qubit_idx[edge.dstOp]) {
          max_previous_depth = std::max(max_previous_depth, depth[idx]);
        }
        // Update the depth
        for (auto &idx: op_2_qubit_idx[edge.dstOp]) {
          depth[idx] = max_previous_depth + 1;
        }
        gates.push(edge.dstOp);
      }
    }
  }
  int max_depth = *std::max_element(depth.begin(), depth.end());
  return max_depth;
}

void Graph::remove_node(Op oldOp) {
  assert(oldOp.ptr->tp != GateType::input_qubit);
  int num_qubits = oldOp.ptr->get_num_qubits();
  if (inEdges.find(oldOp) != inEdges.end()) {
    auto in_edges = inEdges[oldOp];
    for (auto edge : in_edges) {
      auto src_op = edge.srcOp;
      assert(outEdges.find(src_op) != outEdges.end());
      auto out_edges = outEdges[src_op];
      for (auto out_edge : out_edges) {
        if (out_edge.dstOp == oldOp) {
          outEdges[src_op].erase(out_edge);
          if (outEdges[src_op].empty())
            outEdges.erase(src_op);
          break;
        }
      }
    }
  }
  if (outEdges.find(oldOp) != outEdges.end()) {
    auto out_edges = outEdges[oldOp];
    for (auto out_edge : out_edges) {
      auto dst_op = out_edge.dstOp;
      assert(inEdges.find(dst_op) != inEdges.end());
      auto in_edges = inEdges[dst_op];
      for (auto in_edge : in_edges) {
        if (in_edge.srcOp == oldOp) {
          inEdges[dst_op].erase(in_edge);
          if (inEdges[dst_op].empty())
            inEdges.erase(dst_op);
          break;
        }
      }
    }
  }
  if (num_qubits != 0) {
    // Add edges between the inputs and outputs of the to-be removed
    // node Only add edges that connect qubits
    if (inEdges.find(oldOp) != inEdges.end() &&
        outEdges.find(oldOp) != outEdges.end()) {
      auto input_edges = inEdges[oldOp];
      auto output_edges = outEdges[oldOp];
      for (auto in_edge : input_edges) {
        for (auto out_edge : output_edges) {
          if (in_edge.dstIdx < num_qubits &&
              in_edge.dstIdx == out_edge.srcIdx) {
            add_edge(in_edge.srcOp, out_edge.dstOp, in_edge.srcIdx,
                     out_edge.dstIdx);
          }
        }
      }
    }
  }
  inEdges.erase(oldOp);
  outEdges.erase(oldOp);
  constant_param_values.erase(oldOp);
}

void Graph::remove_edge(Op srcOp, Op dstOp) {
  if (inEdges.find(dstOp) != inEdges.end()) {
    auto &edge_list = inEdges[dstOp];
    for (auto edge : edge_list) {
      if (edge.srcOp == srcOp) {
        edge_list.erase(edge);
        break;
      }
    }
    if (inEdges[dstOp].empty())
      inEdges.erase(dstOp);
  }
  if (outEdges.find(srcOp) != outEdges.end()) {
    auto &edge_list = outEdges[srcOp];
    for (auto edge : edge_list) {
      if (edge.dstOp == dstOp) {
        edge_list.erase(edge);
        break;
      }
    }
    if (outEdges[srcOp].empty())
      outEdges.erase(srcOp);
  }
}

// Merge constant parameters
// Eliminate rotation with parameter 0
void Graph::constant_and_rotation_elimination() {
  std::queue<Op> op_queue;
  // Compute the hash value for input ops
  for (auto it = outEdges.cbegin(); it != outEdges.cend(); it++) {
    if (it->first.ptr->tp == GateType::input_qubit ||
        it->first.ptr->tp == GateType::input_param) {
      op_queue.push(it->first);
    }
  }

  // Construct in-degree map
  std::map<Op, size_t> op_in_edges_cnt;
  for (auto it = inEdges.cbegin(); it != inEdges.cend(); ++it) {
    op_in_edges_cnt[it->first] = it->second.size();
  }

  while (!op_queue.empty()) {
    auto op = op_queue.front();
    op_queue.pop();
    if (outEdges.find(op) != outEdges.end()) {
      std::set<Edge, EdgeCompare> list = outEdges[op];
      for (auto it2 = list.cbegin(); it2 != list.cend(); it2++) {
        auto e = *it2;
        op_in_edges_cnt[e.dstOp]--;
        if (op_in_edges_cnt[e.dstOp] == 0) {
          op_queue.push(e.dstOp);
        }
      }
    }
    // Won't remove node in op_queue
    // Remove node won't change the in-degree of other nodes
    // because we only remove poped nodes and their predecessors
    if (op.ptr->is_parameter_gate()) {
      // Parameter gate, check if all its params are constant
      assert(inEdges.find(op) != inEdges.end());
      bool all_constants = true;
      auto list = inEdges[op];
      for (auto it = list.begin(); it != list.end(); ++it) {
        auto src_op = it->srcOp;
        if (constant_param_values.find(src_op) == constant_param_values.end()) {
          all_constants = false;
          break;
        }
      }
      if (all_constants) {
        if (op.ptr->tp == GateType::add) {
          ParamType params[2], result = 0;
          for (auto it = list.begin(); it != list.end(); ++it) {
            auto edge = *it;
            params[edge.dstIdx] = constant_param_values[edge.srcOp];
            remove_node(edge.srcOp);
          }
          result = params[0] + params[1];

          assert(outEdges[op].size() == 1);
          auto output_dst_op = (*outEdges[op].begin()).dstOp;
          auto output_dst_idx = (*outEdges[op].begin()).dstIdx;
          remove_node(op);

          Op merged_op(context->next_global_unique_id(),
                       context->get_gate(GateType::input_param));
          add_edge(merged_op, output_dst_op, 0, output_dst_idx);
          constant_param_values[merged_op] = result;
        } else if (op.ptr->tp == GateType::neg) {
          ParamType params[2], result = 0;
          for (auto it = list.begin(); it != list.end(); ++it) {
            auto edge = *it;
            params[edge.dstIdx] = constant_param_values[edge.srcOp];
            remove_node(edge.srcOp);
          }
          result = params[0] - params[1];

          assert(outEdges[op].size() == 1);
          auto output_dst_op = (*outEdges[op].begin()).dstOp;
          auto output_dst_idx = (*outEdges[op].begin()).dstIdx;
          remove_node(op);

          Op merged_op(context->next_global_unique_id(),
                       context->get_gate(GateType::input_param));
          add_edge(merged_op, output_dst_op, 0, output_dst_idx);
          constant_param_values[merged_op] = result;
        } else {
          assert(false && "Unimplemented parameter gates");
        }
      }
    } else if (op.ptr->is_parametrized_gate()) {
      // Eliminate 0 rotation gates
      auto input_edges = inEdges[op];
      bool all_parameter_is_0 = true;
      int num_qubits = op.ptr->get_num_qubits();
      for (auto in_edge : input_edges) {
        if (in_edge.dstIdx >= num_qubits &&
            in_edge.srcOp.ptr->is_parameter_gate()) {
          all_parameter_is_0 = false;
          break;
        } else if (in_edge.dstIdx >= num_qubits) {
          if (constant_param_values.find(in_edge.srcOp) ==
              constant_param_values.end()) {
            // Not a constant parameter
            all_parameter_is_0 = false;
            break;
          } else {
            // A constant parameter
            if (!equal_to_2k_pi(constant_param_values[in_edge.srcOp])) {
              // The constant parameter is not 2kpi
              all_parameter_is_0 = false;
              break;
            }
          }
        }
      }
      if (all_parameter_is_0) {
        // Delete all parameter nodes, they are all 0
        for (const auto &e : input_edges) {
          if (e.dstIdx > num_qubits) {
            remove_node(e.srcOp);
          }
        }
        remove_node(op);
      }
    }
  }
}

uint64_t Graph::xor_bitmap(uint64_t src_bitmap, int src_idx,
                           uint64_t dst_bitmap, int dst_idx) {
  uint64_t dst_bit = 1 << dst_idx; // Get mask, only dst_idx is 1
  dst_bit &= dst_bitmap;           // Get dst_idx bit
  dst_bit >>= dst_idx;
  dst_bit <<= src_idx;
  return src_bitmap ^= dst_bit;
}

void Graph::expand(Pos pos, bool left, GateType target_rotation,
                   std::unordered_set<Pos, PosHash> &covered,
                   std::unordered_map<int, Pos> &anchor_point,
                   std::unordered_map<Pos, int, PosHash> pos_to_qubits,
                   std::queue<int> &todo_qubits) {
  covered.insert(pos);
  while (true) {
    if (!move_forward(pos, left))
      return;
    if (pos.op.ptr->tp == GateType::cx) {
      // Insert the other side of cnot to anchor_points
      if (anchor_point.find(pos_to_qubits[Pos(pos.op, pos.idx ^ 1)]) ==
          anchor_point.end()) {
        anchor_point[pos_to_qubits[Pos(pos.op, pos.idx ^ 1)]] =
            Pos(pos.op, pos.idx ^ 1);
        todo_qubits.push(pos_to_qubits[Pos(pos.op, pos.idx ^ 1)]);
      }
      covered.insert(pos);
    } else if (moveable(pos.op.ptr->tp)) {
      covered.insert(pos);
      continue;
    } else {
      break;
    }
  }
}

bool Graph::move_forward(Pos &pos, bool left) {
  if (left) {
    if (inEdges.find(pos.op) == inEdges.end())
      return false;
    else {
      auto in_edges = inEdges[pos.op];
      for (const auto &edge : in_edges) {
        if (edge.dstIdx == pos.idx) {
          pos.op = edge.srcOp;
          pos.idx = edge.srcIdx;
          return true;
        }
      }
    }
  } else {
    if (outEdges.find(pos.op) == outEdges.end()) {
      return false;
    } else {
      auto out_edges = outEdges[pos.op];
      for (const auto &edge : out_edges) {
        if (edge.srcIdx == pos.idx) {
          pos.op = edge.dstOp;
          pos.idx = edge.dstIdx;
          return true;
        }
      }
      return false; // Output qubit
    }
  }
  assert(false); // Should not reach here
}

bool Graph::moveable(GateType tp) {
  if (tp == GateType::cx || tp == GateType::x || tp == GateType::rz ||
      tp == GateType::u1)
    return true;
  return false;
}

void Graph::explore(Pos pos, bool left,
                    std::unordered_set<Pos, PosHash> &covered) {
  while (true) {
    if (covered.find(pos) == covered.end())
      return;
    if (pos.op.ptr->tp == GateType::cx && pos.idx == 1 &&
        covered.find(Pos(pos.op, 0)) == covered.end()) {
      // pos is the target qubit of a cnot
      remove(pos, left, covered);
    } else {
      if (!move_forward(pos, left)) {
        return;
      }
    }
  }
}

void Graph::remove(Pos pos, bool left,
                   std::unordered_set<Pos, PosHash> &covered) {
  if (covered.find(pos) == covered.end())
    return;
  covered.erase(pos);
  if (pos.op.ptr->tp == GateType::cx &&
      pos.idx == 0) /*pos is the control qubit of a cnot*/
    remove(Pos(pos.op, 1), left, covered);
  if (!move_forward(pos, left))
    return;
  remove(pos, left, covered);
}

bool Graph::merge_2_rotation_op(Op op_0, Op op_1) {
  // Marge rotation op_1 to rotation op_0
  int num_qubits = op_0.ptr->get_num_qubits();
  int num_params = op_0.ptr->get_num_parameters();

  std::map<int, Op> param_idx_2_op_0;
  std::map<int, Op> param_idx_2_op_1;

  assert(inEdges.find(op_0) != inEdges.end());
  assert(inEdges.find(op_1) != inEdges.end());
  auto input_edges_0 = inEdges[op_0];
  for (auto it = input_edges_0.begin(); it != input_edges_0.end(); ++it) {
    auto edge_0 = *it;
    if (edge_0.dstIdx >= num_qubits) {
      param_idx_2_op_0[edge_0.dstIdx] = edge_0.srcOp;
    }
  }
  auto input_edges_1 = inEdges[op_1];
  for (auto it = input_edges_1.begin(); it != input_edges_1.end(); ++it) {
    auto edge_1 = *it;
    if (edge_1.dstIdx >= num_qubits) {
      // Which means that it is a parameter input
      param_idx_2_op_1[edge_1.dstIdx] = edge_1.srcOp;
    }
  }
  for (int i = num_qubits; i < num_qubits + num_params; ++i) {
    if (constant_param_values.find(param_idx_2_op_0[i]) !=
            constant_param_values.end() &&
        constant_param_values.find(param_idx_2_op_1[i]) !=
            constant_param_values.end()) {
      // Index i parameter at both Ops are constant
      ParamType sum = constant_param_values[param_idx_2_op_0[i]] +
                      constant_param_values[param_idx_2_op_1[i]];
      remove_node(param_idx_2_op_0[i]);
      remove_node(param_idx_2_op_1[i]);
      Op new_constant_op(context->next_global_unique_id(),
                         context->get_gate(GateType::input_param));
      add_edge(new_constant_op, op_0, 0, i);
      constant_param_values[new_constant_op] = sum;
    } else {
      // Add a add gate
      Op new_add_op(context->next_global_unique_id(),
                    context->get_gate(GateType::add));
      add_edge(param_idx_2_op_0[i], new_add_op, 0, 0);
      add_edge(param_idx_2_op_1[i], new_add_op, 0, 1);
      add_edge(new_add_op, op_0, 0, i);
      remove_edge(param_idx_2_op_0[i], op_0);
      remove_edge(param_idx_2_op_1[i], op_1);
    }
  }
  remove_node(op_1);
  return true;
}

void Graph::rotation_merging(GateType target_rotation) {
  // Step 1: calculate the bitmask of each operator
  std::unordered_map<Pos, uint64_t, PosHash> bitmasks;
  std::unordered_map<Pos, int, PosHash> pos_to_qubits;
  std::queue<Op> todos;

  // For all input_qubits, initialize its bitmap, and assign it a idx
  for (const auto &it : outEdges) {
    if (it.first.ptr->tp == GateType::input_qubit) {
      todos.push(it.first);
      int qubit_idx = input_qubit_op_2_qubit_idx[it.first];
      bitmasks[Pos(it.first, 0)] = 1 << qubit_idx;
      pos_to_qubits[Pos(it.first, 0)] = qubit_idx;
    } else if (it.first.ptr->tp == GateType::input_param) {
      todos.push(it.first);
    }
  }

  // Construct in-degree map
  std::map<Op, size_t> op_in_edges_cnt;
  for (auto it = inEdges.begin(); it != inEdges.end(); ++it) {
    op_in_edges_cnt[it->first] = it->second.size();
  }

  // Traverse the graph with topological order
  // Construct the bitmap for all position
  while (!todos.empty()) {
    auto op = todos.front();
    todos.pop();
    // Explore the outEdges of op
    if (op.ptr->tp == GateType::cx) {
      auto in_edge_list = inEdges[op];
      std::vector<Pos> pos_list(2); // Two inputs for cx gate
      for (const auto &edge : in_edge_list) {
        pos_list[edge.dstIdx] = Pos(edge.srcOp, edge.srcIdx);
      }
      bitmasks[Pos(op, 0)] = bitmasks[pos_list[0]];
      bitmasks[Pos(op, 1)] = bitmasks[pos_list[0]] ^ bitmasks[pos_list[1]];
      //    xor_bitmap(bitmasks[pos_list[1]],
      //    pos_to_qubits[pos_list[1]],
      //               bitmasks[pos_list[0]],
      //               pos_to_qubits[pos_list[0]]);
      pos_to_qubits[Pos(op, 0)] = pos_to_qubits[pos_list[0]];
      pos_to_qubits[Pos(op, 1)] = pos_to_qubits[pos_list[1]];
    } else if (op.ptr->tp != GateType::input_qubit &&
               op.ptr->tp != GateType::input_param) {
      auto in_edge_list = inEdges[op];
      int num_qubits = op.ptr->get_num_qubits();
      std::vector<Pos> pos_list(num_qubits);
      for (const auto &edge : in_edge_list) {
        if (edge.dstIdx < num_qubits) {
          pos_list[edge.dstIdx] = Pos(edge.srcOp, edge.srcIdx);
        }
      }
      for (int i = 0; i < num_qubits; ++i) {
        bitmasks[Pos(op, i)] = bitmasks[pos_list[i]];
        pos_to_qubits[Pos(op, i)] = pos_to_qubits[pos_list[i]];
      }
    }

    if (outEdges.find(op) != outEdges.end()) {
      std::set<Edge, EdgeCompare> list = outEdges[op];
      std::set<Edge, EdgeCompare>::const_iterator it2;
      for (it2 = list.begin(); it2 != list.end(); it2++) {
        auto e = *it2;
        op_in_edges_cnt[e.dstOp]--;
        if (op_in_edges_cnt[e.dstOp] == 0) {
          todos.push(e.dstOp);
        }
      }
    }
  }

  // Step 2: Propagate all CNOTs
  std::queue<Op> todo_cx;
  std::unordered_set<Op, OpHash> visited_cx;
  for (const auto &it : inEdges)
    if (it.first.ptr->tp == GateType::cx) {
      todo_cx.push(it.first);
    }
  while (!todo_cx.empty()) {
    const auto cx = todo_cx.front();
    todo_cx.pop();
    if (visited_cx.find(cx) != visited_cx.end())
      continue;
    std::unordered_map<int, Pos> anchor_point;
    std::queue<int> todo_qubits;
    std::unordered_set<Pos, PosHash> covered;
    anchor_point[pos_to_qubits[Pos(cx, 0)]] = Pos(cx, 0);
    anchor_point[pos_to_qubits[Pos(cx, 1)]] = Pos(cx, 1);
    todo_qubits.push(pos_to_qubits[Pos(cx, 0)]);
    todo_qubits.push(pos_to_qubits[Pos(cx, 1)]);
    while (!todo_qubits.empty()) {
      int qid = todo_qubits.front();
      todo_qubits.pop();
      expand(anchor_point[qid], true, target_rotation, covered, anchor_point,
             pos_to_qubits, todo_qubits); // expand left
      expand(anchor_point[qid], false, target_rotation, covered, anchor_point,
             pos_to_qubits,
             todo_qubits); // expand right
    }

    // Step 3: deal with partial cnot
    for (const auto &it : anchor_point) {
      auto pos = it.second;
      explore(pos, true, covered);
      explore(pos, false, covered);
    }

    // Step 4: merge rotations with the same bitmasks on the same qubit
    std::unordered_map<
        int, std::unordered_map<uint64_t, std::unordered_set<Pos, PosHash>>>
        qubit_2_bm_2_pos;
    for (const auto &pos : covered) {
      if (pos.op.ptr->tp == GateType::cx) {
        visited_cx.insert(pos.op);
      }
      if (pos.op.ptr->tp == target_rotation) {
        int qubit_idx = pos_to_qubits[pos];
        auto bm = bitmasks[pos];
        if (qubit_2_bm_2_pos.find(qubit_idx) == qubit_2_bm_2_pos.end()) {
          qubit_2_bm_2_pos[qubit_idx];
        }
        if (qubit_2_bm_2_pos[qubit_idx].find(bm) ==
            qubit_2_bm_2_pos[qubit_idx].end()) {
          qubit_2_bm_2_pos[qubit_idx][bm];
        }
        if (qubit_2_bm_2_pos[qubit_idx][bm].find(pos) ==
            qubit_2_bm_2_pos[qubit_idx][bm].end()) {
          qubit_2_bm_2_pos[qubit_idx][bm].insert(pos);
        }
      }
    }

    for (const auto &it_0 : qubit_2_bm_2_pos) {
      for (const auto &it_1 : it_0.second) {
        auto &pos_set = it_1.second;
        assert(pos_set.size() >= 1);
        Pos first;
        bool is_first = true;
        for (auto pos : pos_set) {
          if (is_first) {
            first = pos;
            is_first = false;
          } else {
            merge_2_rotation_op(first.op, pos.op);
            // std::cout << "merging op " << gate_type_name(first.op.ptr->tp)
            //           << "(" << first.op.guid << ")"
            //           << " and " << gate_type_name(pos.op.ptr->tp) << "("
            //           << pos.op.guid << ")" << std::endl;
          }
        }

        auto op = first.op;
        auto in_edges = inEdges[op];
        int num_qubits = op.ptr->get_num_qubits();
        bool all_param_is_0 = true;
        for (auto edge : in_edges) {
          if (edge.dstIdx >= num_qubits) {
            if (constant_param_values.find(edge.srcOp) ==
                    constant_param_values.end() ||
                !equal_to_2k_pi(constant_param_values[edge.srcOp])) {
              all_param_is_0 = false;
              break;
            }
          }
        }
        if (all_param_is_0) {
          for (auto edge : in_edges) {
            if (edge.dstIdx >= num_qubits) {
              remove_node(edge.srcOp);
            }
          }
          remove_node(op);
          //   std::cout << "eliminating op " << gate_type_name(op.ptr->tp) <<
          //   "("
          //             << op.guid << ")" << std::endl;
        }
      }
    }
  }
  constant_and_rotation_elimination();
}

size_t Graph::get_num_qubits() const {
  return input_qubit_op_2_qubit_idx.size();
}

void Graph::print_qubit_ops() {
  std::unordered_map<Pos, int, PosHash> pos_to_qubits;
  std::queue<Op> todos;
  for (const auto &it : outEdges) {
    if (it.first.ptr->tp == GateType::input_qubit) {
      todos.push(it.first);
      int qubit_idx = input_qubit_op_2_qubit_idx[it.first];
      pos_to_qubits[Pos(it.first, 0)] = qubit_idx;
    } else if (it.first.ptr->tp == GateType::input_param) {
      todos.push(it.first);
    }
  }
  // Construct in-degree map
  std::map<Op, size_t> op_in_edges_cnt;
  for (auto it = inEdges.begin(); it != inEdges.end(); ++it) {
    op_in_edges_cnt[it->first] = it->second.size();
  }

  std::map<int, std::vector<int>> qubit_2_op_list;
  int circuit_qubit_num = get_num_qubits();
  for (int i = 0; i < circuit_qubit_num; ++i) {
    qubit_2_op_list[i];
  }

  while (!todos.empty()) {
    auto op = todos.front();
    todos.pop();

    if (op.ptr->tp != GateType::input_qubit &&
        op.ptr->tp != GateType::input_param) {
      int num_qubits = op.ptr->get_num_qubits();
      assert(inEdges.find(op) != inEdges.end());
      auto in_edges = inEdges[op];
      for (auto edge : in_edges) {
        if (edge.dstIdx < num_qubits) {
          int qid = pos_to_qubits[Pos(edge.srcOp, edge.srcIdx)];
          pos_to_qubits[Pos(op, edge.dstIdx)] = qid;
          qubit_2_op_list[qid].push_back(op.guid);
        }
      }
    }

    if (outEdges.find(op) != outEdges.end()) {
      std::set<Edge, EdgeCompare> list = outEdges[op];
      std::set<Edge, EdgeCompare>::const_iterator it2;
      for (it2 = list.begin(); it2 != list.end(); it2++) {
        auto e = *it2;
        op_in_edges_cnt[e.dstOp]--;
        if (op_in_edges_cnt[e.dstOp] == 0) {
          todos.push(e.dstOp);
        }
      }
    }
  }
  for (auto it : qubit_2_op_list) {
    int qid = it.first;
    std::cout << qid << ":\t";
    for (auto op_guid : qubit_2_op_list[qid]) {
      std::cout << op_guid << " ";
    }
    std::cout << std::endl;
  }
}

void Graph::to_qasm(const std::string &save_filename, bool print_result,
                    bool print_guid) const {
  std::ofstream ofs(save_filename);
  ofs << to_qasm(print_result, print_guid);
}

std::string Graph::to_qasm(bool print_result, bool print_guid) const {
  std::ostringstream o;
  std::map<float, std::string> constant_2_pi;
  std::vector<float> multiples;
  for (int i = 1; i <= 8; ++i) {
    multiples.push_back(i * 0.25);
    multiples.push_back(-i * 0.25);
  }
  multiples.push_back(0);
  for (auto f : multiples) {
    constant_2_pi[f * PI] = "pi*" + std::to_string(f);
  }

  o << "OPENQASM 2.0;" << std::endl;
  o << "include \"qelib1.inc\";" << std::endl;
  o << "qreg q[" << get_num_qubits() << "];" << std::endl;

  std::unordered_map<Pos, int, PosHash> pos_to_qubits;
  std::queue<Op> todos;
  for (const auto &it : outEdges) {
    if (it.first.ptr->tp == GateType::input_qubit) {
      todos.push(it.first);
      int qubit_idx = input_qubit_op_2_qubit_idx.find(it.first)->second;
      pos_to_qubits[Pos(it.first, 0)] = qubit_idx;
    } else if (it.first.ptr->tp == GateType::input_param) {
      todos.push(it.first);
    }
  }

  // Construct in-degree map
  std::map<Op, size_t> op_in_edges_cnt;
  for (auto it = inEdges.begin(); it != inEdges.end(); ++it) {
    op_in_edges_cnt[it->first] = it->second.size();
  }

  while (!todos.empty()) {
    auto op = todos.front();
    todos.pop();

    if (op.ptr->tp != GateType::input_qubit &&
        op.ptr->tp != GateType::input_param) {
      assert(op.ptr->is_quantum_gate()); // Should not have any
                                         // arithmetic gates
      std::ostringstream iss;
      iss << gate_type_name(op.ptr->tp);
      int num_qubits = op.ptr->get_num_qubits();
      auto in_edges = inEdges.find(op)->second;
      // Maintain pos_to_qubits
      if (op.ptr->is_parametrized_gate()) {
        iss << '(';
        assert(inEdges.find(op) != inEdges.end());
        int num_params = op.ptr->get_num_parameters();
        std::vector<ParamType> param_values(num_params);
        for (auto edge : in_edges) {
          // Print parameters
          if (edge.dstIdx >= num_qubits) {
            // Parameter inputs
            assert(constant_param_values.find(edge.srcOp) !=
                   constant_param_values.end()); // All parameters should be
                                                 // constant
            param_values[edge.dstIdx - num_qubits] =
                constant_param_values.find(edge.srcOp)->second;
          }
        }
        bool first = true;
        for (auto f : param_values) {
          if (first) {
            first = false;
          } else
            iss << ',';
          bool found = false;
          for (auto it : constant_2_pi) {
            if (std::abs(f - it.first) < eps) {
              iss << it.second;
              found = true;
            }
          }
          if (!found) {
            iss << "pi*" << f / PI;
          }
        }
        iss << ')';
      }
      iss << ' ';
      std::vector<int> q_idx(num_qubits);
      for (auto edge : in_edges) {
        if (edge.dstIdx < num_qubits) {
          assert(pos_to_qubits.find(Pos(edge.srcOp, edge.srcIdx)) !=
                 pos_to_qubits.end());
          q_idx[edge.dstIdx] = pos_to_qubits[Pos(edge.srcOp, edge.srcIdx)];
          pos_to_qubits[Pos(op, edge.dstIdx)] =
              pos_to_qubits[Pos(edge.srcOp, edge.srcIdx)];
        }
      }
      bool first = true;
      for (auto idx : q_idx) {
        if (first)
          first = false;
        else
          iss << ',';
        iss << "q[" << idx << ']';
      }
      if (print_guid)
        iss << "; # guid = " << op.guid << std::endl;
      else
        iss << ';' << std::endl;
      o << iss.str();
    }

    if (outEdges.find(op) != outEdges.end()) {
      std::set<Edge, EdgeCompare> list = outEdges.find(op)->second;
      std::set<Edge, EdgeCompare>::const_iterator it2;
      for (it2 = list.begin(); it2 != list.end(); it2++) {
        auto e = *it2;
        op_in_edges_cnt[e.dstOp]--;
        if (op_in_edges_cnt[e.dstOp] == 0) {
          todos.push(e.dstOp);
        }
      }
    }
  }
  const std::string ostr(o.str());
  if (print_result)
    std::cout << ostr;
  return ostr;
}

template <class _CharT, class _Traits>
std::shared_ptr<Graph>
Graph::_from_qasm_stream(Context *ctx,
                         std::basic_istream<_CharT, _Traits> &qasm_stream) {

  std::shared_ptr<Graph> graph(new Graph(ctx));
  std::string line;
  GateType gate_type;
  std::vector<Pos> pos_on_qubits;
  while (std::getline(qasm_stream, line)) {
    // repleace comma with space
    find_and_replace_all(line, ",", " ");
    find_and_replace_all(line, "(", " ");
    find_and_replace_all(line, ")", " ");
    // ignore semicolon at the end
    find_and_replace_all(line, ";", "");
    std::stringstream ss(line);
    std::string command;
    std::getline(ss, command, ' ');
    if (command == "OPENQASM") {
      continue; // ignore this line
    } else if (command == "include") {
      continue; // ignore this line
    } else if (command == "creg") {
      continue; // ignore this line
    } else if (command == "qreg") {
      std::string token;
      getline(ss, token, ' ');
      size_t num_qubits = string_to_number(token);
      pos_on_qubits.resize(num_qubits);
      for (int i = 0; i < num_qubits; ++i) {
        auto op = graph->add_qubit(i);
        pos_on_qubits[i] = Pos(op, 0);
        // Construct input_qubit_op_2_qubit_idx
        graph->input_qubit_op_2_qubit_idx[op] = i;
      }
      assert(!ss.good());
    } else if (is_gate_string(command, gate_type)) {
      Gate *gate = graph->context->get_gate(gate_type);
      if (!gate) {
        std::cerr << "Unsupported gate in current context: " << command
                  << std::endl;
        return nullptr;
      }
      if (gate->is_parametrized_gate()) {
        auto op = graph->new_gate(gate_type);
        auto num_qubits = graph->context->get_gate(gate_type)->num_qubits;
        auto num_params = graph->context->get_gate(gate_type)->num_parameters;
        for (int i = 0; i < num_params; ++i) {
          assert(ss.good());
          std::string token;
          ss >> token;
          // Currently only support the format of pi*0.123, 0.123*pi or 0.123
          ParamType p;
          if (token.find("pi") == 0) {
            auto d = token.substr(3, std::string::npos);
            p = std::stod(d) * PI;
          }
          else if(token.find("pi") != std::string::npos){
            // 0.123*pi
            auto d = token.substr(0, token.find("*"));
            p = std::stod(d) * PI;
          }
          else {
            p = std::stod(token);
          }
          auto src_op = graph->add_parameter(p);
          int src_idx = 0;
          auto dst_op = op;
          auto dst_idx = num_qubits + i;
          graph->add_edge(src_op, dst_op, src_idx, dst_idx);
        }
        for (int i = 0; i < num_qubits; ++i) {
          assert(ss.good());
          std::string token;
          ss >> token;
          int qubit_idx = string_to_number(token);
          if (qubit_idx != -1) {
            auto src_op = pos_on_qubits[qubit_idx].op;
            auto src_idx = pos_on_qubits[qubit_idx].idx;
            auto dst_op = op;
            auto dst_idx = i;
            graph->add_edge(src_op, dst_op, src_idx, dst_idx);
            pos_on_qubits[qubit_idx] = Pos(dst_op, dst_idx);
          } else
            return nullptr;
        }
      } else if (gate->is_quantum_gate()) {
        auto op = graph->new_gate(gate_type);
        auto num_qubits = graph->context->get_gate(gate_type)->num_qubits;
        for (int i = 0; i < num_qubits; ++i) {
          assert(ss.good());
          std::string token;
          ss >> token;
          int qubit_idx = string_to_number(token);
          if (qubit_idx != -1) {
            auto src_op = pos_on_qubits[qubit_idx].op;
            auto src_idx = pos_on_qubits[qubit_idx].idx;
            auto dst_op = op;
            auto dst_idx = i;
            graph->add_edge(src_op, dst_op, src_idx, dst_idx);
            pos_on_qubits[qubit_idx] = Pos(dst_op, dst_idx);
          } else
            return nullptr;
        }
      }
    } else {
      std::cout << "Unknown gate: " << command << std::endl;
      assert(false);
    }
  }

  graph->_construct_pos_2_logical_qubit();
  return graph;
}

std::shared_ptr<Graph> Graph::from_qasm_file(Context *ctx,
                                             const std::string &filename) {
  std::ifstream fin;
  fin.open(filename, std::ifstream::in);
  if (!fin.is_open()) {
    std::cerr << "Failed to open " << filename << std::endl;
    return nullptr;
  }
  auto graph = _from_qasm_stream(ctx, fin);
  fin.close();
  return graph;
}

std::shared_ptr<Graph> Graph::from_qasm_str(Context *ctx,
                                            const std::string qasm_str) {
  std::stringstream sstream(qasm_str);
  return _from_qasm_stream(ctx, sstream);
}

void Graph::draw_circuit(const std::string &src_file_name,
                         const std::string &save_filename) {

  system(("python python/draw_graph.py " + src_file_name + " " + save_filename)
             .c_str());
}

std::shared_ptr<Graph> Graph::optimize(
    float alpha, int budget, bool print_subst, Context *ctx,
    const std::string &equiv_file_name, bool use_simulated_annealing,
    bool enable_early_stop, bool use_rotation_merging_in_searching,
    GateType target_rotation, std::string circuit_name, int timeout) {
  EquivalenceSet eqs;
  // Load equivalent dags from file
  auto start = std::chrono::steady_clock::now();
  if (!eqs.load_json(ctx, equiv_file_name)) {
    std::cerr << "Failed to load equivalence file: " << equiv_file_name
              << std::endl;
    exit(1);
  }
  auto end = std::chrono::steady_clock::now();
  //   std::cout << std::dec << eqs.num_equivalence_classes()
  //             << " classes of equivalences with " << eqs.num_total_dags()
  //             << " DAGs are loaded in "
  //             <<
  //             (double)std::chrono::duration_cast<std::chrono::milliseconds>(
  //                    end - start)
  //                        .count() /
  //                    1000.0
  //             << " seconds." << std::endl;

  std::vector<GraphXfer *> xfers;
  for (const auto &equiv_set : eqs.get_all_equivalence_sets()) {
    bool first = true;
    DAG *first_dag = nullptr;
    for (const auto &dag : equiv_set) {
      if (first) {
        // Used raw pointer according to the GraphXfer API
        // May switch to smart pointer later
        first_dag = new DAG(*dag);
        first = false;
      } else {
        DAG *other_dag = new DAG(*dag);
        // first_dag is src, others are dst
        // if (first_dag->get_num_gates() !=
        // other_dag->get_num_gates()) {
        //   std::cout << first_dag->get_num_gates() << " "
        //             << other_dag->get_num_gates() << "; ";
        // }
        auto first_2_other =
            GraphXfer::create_GraphXfer(ctx, first_dag, other_dag);
        // first_dag is dst, others are src
        auto other_2_first =
            GraphXfer::create_GraphXfer(ctx, other_dag, first_dag);
        if (first_2_other != nullptr)
          xfers.push_back(first_2_other);
        if (other_2_first != nullptr)
          xfers.push_back(other_2_first);
        delete other_dag;
      }
    }
    delete first_dag;
  }

  //   std::cout << "Number of different transfers is " << xfers.size() << "."
  //             << std::endl;

  int counter = 0;
  int maxNumOps = inEdges.size();

  std::priority_queue<std::shared_ptr<Graph>,
                      std::vector<std::shared_ptr<Graph>>, GraphCompare>
      candidates;
  std::set<size_t> hashmap;
  std::shared_ptr<Graph> bestGraph(new Graph(*this));
  float bestCost = total_cost();
  candidates.push(std::shared_ptr<Graph>(new Graph(*this)));
  hashmap.insert(hash());
  std::vector<GraphXfer *> good_xfers;

  //   printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
  start = std::chrono::steady_clock::now();
  // TODO: add optional rotation merging in sa
  if (use_simulated_annealing) {
    const double kSABeginTemp = bestCost;
    const double kSAEndTemp = kSABeginTemp / 1e6;
    const double kSACoolingFactor = 1.0 - 1e-1;
    const int kNumKeepGraph = 20;
    constexpr bool always_run_rotation_merging = true;
    const double kRunRotationMergingRate = -1;
    constexpr bool always_delete_original_circuit = true;
    const double kDeleteOriginalCircuitRate = -1;
    // <cost, graph>
    std::vector<std::pair<float, std::shared_ptr<Graph>>> sa_candidates;
    sa_candidates.reserve(kNumKeepGraph);
    sa_candidates.emplace_back(bestCost, this);
    int num_iteration = 0;
    std::cout << "Begin simulated annealing with " << xfers.size() << " xfers."
              << std::endl;
    for (double T = kSABeginTemp; T > kSAEndTemp; T *= kSACoolingFactor) {
      num_iteration++;
      hashmap.clear();
      std::vector<std::pair<float, std::shared_ptr<Graph>>> new_candidates;
      new_candidates.reserve(sa_candidates.size() * xfers.size());
      int num_possible_new_candidates = 0;
      int num_candidates_kept = 0;
      for (auto &candidate : sa_candidates) {
        const auto current_cost = candidate.first;
        std::vector<std::shared_ptr<Graph>> current_new_candidates;
        current_new_candidates.reserve(xfers.size());
        bool stop_search = false;
        for (auto &xfer : xfers) {
          xfer->run(0, candidate.second.get(), current_new_candidates, hashmap,
                    bestCost * alpha, 2 * maxNumOps, enable_early_stop,
                    stop_search);
        }
        num_possible_new_candidates += current_new_candidates.size();
        for (auto &new_candidate : current_new_candidates) {
          if (use_rotation_merging_in_searching &&
              (always_run_rotation_merging ||
               ctx->random_number() <
                   1 - std::exp(kRunRotationMergingRate / T))) {
            new_candidate->rotation_merging(target_rotation);
          }
          const auto new_cost = new_candidate->total_cost();
          if (new_cost < bestCost) {
            bestGraph = new_candidate;
            bestCost = new_cost;
          }
          // Apply the criteria of simulated annealing.
          // Cost is the smaller the better here.
          if (new_cost < current_cost ||
              ctx->random_number() < std::exp((current_cost - new_cost) / T)) {
            // Accept the new candidate.
            new_candidates.emplace_back(new_cost, new_candidate);
          } else {
            new_candidate.reset();
          }
        }
        if (!always_delete_original_circuit &&
            ctx->random_number() < std::exp(kDeleteOriginalCircuitRate / T) &&
            hashmap.find(candidate.second->hash()) == hashmap.end()) {
          // Keep the original candidate.
          new_candidates.emplace_back(candidate);
          hashmap.insert(candidate.second->hash());
          num_candidates_kept++;
        } else {
          if (candidate.second != bestGraph && candidate.second.get() != this) {
            candidate.second.reset();
          }
        }
      }

      // Compute some statistical information to output, can be
      // commented when verbose=false
      const auto num_new_candidates = new_candidates.size();
      if (new_candidates.empty()) {
        std::cout << "No new candidates. Early stopping." << std::endl;
        break;
      }
      assert(!new_candidates.empty());
      auto min_cost = new_candidates[0].first;
      auto max_cost = new_candidates[0].first;
      for (const auto &new_candidate : new_candidates) {
        min_cost = std::min(min_cost, new_candidate.first);
        max_cost = std::max(max_cost, new_candidate.first);
      }

      if (new_candidates.size() > kNumKeepGraph) {
        // Prune some candidates.
        // TODO: make sure the candidates kept are far from each
        // other
        // TODO: use hashmap to avoid keep searching for the same
        // graphs
        std::partial_sort(new_candidates.begin(),
                          new_candidates.begin() + kNumKeepGraph,
                          new_candidates.end());
        for (int i = kNumKeepGraph; i < (int)new_candidates.size(); i++) {
          if (new_candidates[i].second.get() != this &&
              new_candidates[i].second != bestGraph) {
            new_candidates[i].second.reset();
          }
        }
        new_candidates.resize(kNumKeepGraph);
      }
      sa_candidates = std::move(new_candidates);

      std::cout << "Iteration " << num_iteration << ": T = " << std::fixed
                << std::setprecision(2) << T << ", bestcost = " << bestCost
                << ", " << num_candidates_kept << " candidates kept, "
                << num_new_candidates - num_candidates_kept << " out of "
                << num_possible_new_candidates
                << " possible new candidates accepted, cost ranging ["
                << min_cost << ", " << max_cost << "]" << std::endl;
    }
  } else {
    while (!candidates.empty()) {
      auto subGraph = candidates.top();
      if (use_rotation_merging_in_searching) {
        subGraph->rotation_merging(target_rotation);
      }
      candidates.pop();
      if (subGraph->total_cost() < bestCost) {
        bestCost = subGraph->total_cost();
        bestGraph = subGraph;
      }
      if (counter > budget) {
        // TODO: free all remaining candidates when budget exhausted
        //   break;
        ;
      }
      counter++;
      subGraph->constant_and_rotation_elimination();
      end = std::chrono::steady_clock::now();
      if (circuit_name != "")
        std::cout << circuit_name << ": ";
      if ((int)std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                  .count() /
              1000.0 >
          timeout) {
        std::cout << "Timeout. Program terminated. Best cost is "
                  << bestCost << std::endl;
        bestGraph->constant_and_rotation_elimination();
        return bestGraph;
      }
      fprintf(stdout, "bestCost(%.4lf) candidates(%zu) after %.4lf seconds\n",
              bestCost, candidates.size(),
              (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                  end - start)
                      .count() /
                  1000.0);
      fflush(stdout);

      //   std::vector<Graph *> new_candidates;
      bool stop_search = false;
      for (auto &xfer : xfers) {
        std::vector<std::shared_ptr<Graph>> new_candidates;
        xfer->run(0, subGraph.get(), new_candidates, hashmap, bestCost * alpha,
                  2 * maxNumOps, enable_early_stop, stop_search);
        // auto front_gate_count = candidates.top()->gate_count();
        for (auto &candidate : new_candidates) {
          candidates.push(candidate);
        }
        // auto new_front_gate_count = candidates.top()->gate_count();
        // if (new_front_gate_count < front_gate_count) {
        //   good_xfers.push_back(xfer);
        //   }
      }
    }
  }
  //   printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");
  // Print results
  //   std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::iterator it;
  //   for (it = bestGraph->inEdges.begin(); it !=
  //   bestGraph->inEdges.end();
  //   ++it) {
  // 	std::cout << gate_type_name(it->first.ptr->tp) << std::endl;
  //   }
  bestGraph->constant_and_rotation_elimination();
  return bestGraph;
} // namespace quartz

std::shared_ptr<Graph> Graph::optimize(std::vector<GraphXfer *> xfers,
                                       double cost_upper_bound,
                                       std::string circuit_name,
                                       bool print_message, int timeout) {
  auto start = std::chrono::steady_clock::now();
  std::priority_queue<std::shared_ptr<Graph>,
                      std::vector<std::shared_ptr<Graph>>, GraphCompare>
      candidates;
  std::set<size_t> hashmap;
  std::shared_ptr<Graph> best_graph(new Graph(*this));
  auto best_cost = total_cost();

  candidates.push(best_graph);
  hashmap.insert(hash());

  int invoke_cnt = 0;

  while (!candidates.empty()) {
    auto graph = candidates.top();
    candidates.pop();
    std::vector<Op> all_nodes;
    graph->topology_order_ops(all_nodes);
    for (auto xfer : xfers) {
      for (auto const &node : all_nodes) {
        invoke_cnt++;
        auto new_graph =
            graph->apply_xfer(xfer, node, context->has_parameterized_gate());
        auto end = std::chrono::steady_clock::now();
        if ((int)std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                    .count() /
                1000.0 >
            timeout) {
          std::cout << "Timeout. Program terminated. Best cost is "
                    << best_cost << std::endl;
          return best_graph;
        }
        if (new_graph == nullptr)
          continue;

        auto new_hash = new_graph->hash();
        auto new_cost = new_graph->total_cost();
        if(new_cost > cost_upper_bound)
            continue;
        if (hashmap.find(new_hash) == hashmap.end()) {
          hashmap.insert(new_hash);
          candidates.push(new_graph);
          if (new_cost < best_cost) {
            best_cost = new_cost;
            best_graph = new_graph;
          }
        } else
          continue;
      }
    }
    auto end = std::chrono::steady_clock::now();
    if (print_message) {
      std::cout << "[" << circuit_name << "] "
                << "Best cost: " << best_cost
                << "\tcandidate number: " << candidates.size() << "\tafter "
                << (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - start)
                           .count() /
                       1000.0
                << " seconds." << std::endl;
    }
  }
  return best_graph;
}

std::shared_ptr<Graph> Graph::ccz_flip_t(Context *ctx) {
  // Transform ccz to t, an naive solution
  // Simply 1 normal 1 inverse
  auto xfers = GraphXfer::ccz_cx_t_xfer(ctx);
  Graph *graph = this;
  bool flip = false;
  while (true) {
    std::shared_ptr<Graph> new_graph(nullptr);
    if (flip) {
      new_graph = xfers.first->run_1_time(0, graph);
      flip = false;
    } else {
      new_graph = xfers.second->run_1_time(0, graph);
      flip = true;
    }
    if (new_graph.get() == nullptr) {
      return std::shared_ptr<Graph>(graph);
    }
    if (graph != this)
      delete graph;
    graph = new_graph.get();
  }
  assert(false); // Should never reach here
}

std::shared_ptr<Graph> Graph::toffoli_flip_greedy(GateType target_rotation,
                                                  GraphXfer *xfer,
                                                  GraphXfer *inverse_xfer) {
  std::shared_ptr<Graph> temp_graph(new Graph(*this));
  while (true) {
    auto new_graph_0 = xfer->run_1_time(0, temp_graph.get());
    auto new_graph_1 = inverse_xfer->run_1_time(0, temp_graph.get());
    if (new_graph_0.get() == nullptr) {
      assert(new_graph_1.get() == nullptr);
      return temp_graph;
    }
    new_graph_0->rotation_merging(target_rotation);
    new_graph_1->rotation_merging(target_rotation);
    if (new_graph_0->gate_count() <= new_graph_1->gate_count()) {
      temp_graph = new_graph_0;
    } else {
      temp_graph = new_graph_1;
    }
  }
  assert(false); // Should never reach here
}

void Graph::toffoli_flip_greedy_with_trace(GateType target_rotation,
                                           GraphXfer *xfer,
                                           GraphXfer *inverse_xfer,
                                           std::vector<int> &trace) {
  std::shared_ptr<Graph> temp_graph(new Graph(*this));
  while (true) {
    auto new_graph_0 = xfer->run_1_time(0, temp_graph.get());
    auto new_graph_1 = inverse_xfer->run_1_time(0, temp_graph.get());
    if (new_graph_0.get() == nullptr) {
      assert(new_graph_1.get() == nullptr);
      return;
    }
    new_graph_0->rotation_merging(target_rotation);
    new_graph_1->rotation_merging(target_rotation);
    if (new_graph_0->gate_count() <= new_graph_1->gate_count()) {
      temp_graph = new_graph_0;
      trace.push_back(0);
    } else {
      temp_graph = new_graph_1;
      trace.push_back(1);
    }
  }
  assert(false); // Should never reach here
}

std::shared_ptr<Graph>
Graph::toffoli_flip_by_instruction(GateType target_rotation, GraphXfer *xfer,
                                   GraphXfer *inverse_xfer,
                                   std::vector<int> instruction) {
  std::shared_ptr<Graph> graph(new Graph(*this));
  std::shared_ptr<Graph> new_graph(nullptr);
  for (const auto direction : instruction) {
    if (direction == 0) {
      new_graph = xfer->run_1_time(0, graph.get());
    } else {
      new_graph = inverse_xfer->run_1_time(0, graph.get());
    }
    graph = new_graph;
  }
  return graph;
}

std::shared_ptr<Graph> Graph::ccz_flip_greedy_rz() {
  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(context);
  std::vector<int> trace;
  toffoli_flip_greedy_with_trace(GateType::rz, xfer_pair.first,
                                 xfer_pair.second, trace);
  auto graph_ccz_flipped = toffoli_flip_by_instruction(
      GateType::rz, xfer_pair.first, xfer_pair.second, trace);
  return graph_ccz_flipped;
}
// std::shared_ptr<Graph> Graph::ccz_flip_greedy_u1() {}

bool Graph::xfer_appliable(GraphXfer *xfer, Op op) const {
  if (!xfer->can_match(*xfer->srcOps.begin(), op, this)) {
    return false;
  }
  std::unordered_set<OpX *> mapped_opx;
  std::deque<std::pair<OpX *, Op>> opx_op_dq;
  xfer->match(*xfer->srcOps.begin(), op, this);
  mapped_opx.insert(*xfer->srcOps.begin());
  opx_op_dq.push_back(std::make_pair(*xfer->srcOps.begin(), op));
  // If an OpX is mapped to an Op, check whether their corresponding input OpX,
  // input Op, output OpX, output Op can match. Because the source graphs is
  // connected, by doing this we can traverse all nodes.
  bool fail = false;
  size_t idx = 0;
  while (idx < opx_op_dq.size()) {
    auto opx_op_pair = opx_op_dq[idx];
    idx++;
    auto opx_ = opx_op_pair.first;
    auto op_ = opx_op_pair.second;
    auto num_input = opx_->inputs.size();
    auto num_output = opx_->outputs.size();
    // Get all input and output Op of op_ in an ordered list
    std::vector<Op> input_ops(num_input, Op::INVALID_OP);
    std::vector<Op> output_ops(num_output, Op::INVALID_OP);
    std::vector<OpX *> output_opxs(num_output, nullptr);
    if (inEdges.find(op_) != inEdges.end()) {
      for (auto &e : inEdges.find(op_)->second) {
        assert((size_t)e.dstIdx < num_input);
        input_ops[e.dstIdx] = e.srcOp;
      }
    }
    for (size_t i = 0; i < num_input; ++i) {
      auto input_opx = opx_->inputs[i].op;
      if (input_opx != nullptr &&
          mapped_opx.find(input_opx) == mapped_opx.end()) {
        if (!xfer->can_match(input_opx, input_ops[i], this)) {
          fail = true;
          break;
        } else {
          xfer->match(input_opx, input_ops[i], this);
          mapped_opx.insert(input_opx);
          opx_op_dq.push_back(std::make_pair(input_opx, input_ops[i]));
        }
      }
    }
    if (fail) {
      break;
    }
    if (outEdges.find(op_) != outEdges.end()) {
      for (auto &e : outEdges.find(op_)->second) {
        assert((size_t)e.srcIdx < num_output);
        output_ops[e.srcIdx] = e.dstOp;
      }
    }
    // Get all output OpX because we don't have output edges in GraphXfer
    for (auto &opx : xfer->srcOps) {
      for (auto &input_tensor : opx->inputs) {
        if (input_tensor.op == opx_) {
          assert((size_t)input_tensor.idx < num_output);
          output_opxs[input_tensor.idx] = opx;
        }
      }
    }
    for (size_t i = 0; i < num_output; ++i) {
      auto output_opx = output_opxs[i];
      if (output_opx != nullptr &&
          mapped_opx.find(output_opx) == mapped_opx.end()) {
        if (!xfer->can_match(output_opx, output_ops[i], this)) {
          fail = true;
          break;
        } else {
          xfer->match(output_opx, output_ops[i], this);
          mapped_opx.insert(output_opx);
          opx_op_dq.push_back(std::make_pair(output_opx, output_ops[i]));
        }
      }
    }
    if (fail) {
      break;
    }
  }
  if (!fail) {
    // Check qubit consistancy
    std::set<int> qubits;
    for (auto it = xfer->mappedInputs.cbegin(); it != xfer->mappedInputs.cend();
         ++it) {
      if (it->second.first.ptr->is_quantum_gate() ||
          it->second.first.ptr->tp == GateType::input_qubit) {
        // Only check inputs on a qubit
        // Excluding input_param gates and arithmetic gates
        Pos p = Pos(it->second.first, it->second.second);
        auto q = pos_2_logical_qubit.find(p)->second;
        if (qubits.find(q) != qubits.end()) {
          fail = true;
          break;
        } else {
          qubits.insert(q);
        }
      }
    }
  }
  if (!fail) {
    for (auto dst_it = xfer->dstOps.cbegin(); dst_it != xfer->dstOps.cend();
         ++dst_it) {
      if (!fail) {
        OpX *dstOp = *dst_it;
        fail = !xfer->create_new_operator(dstOp, dstOp->mapOp);
      }
    }
  }
  if (!fail) {
    // Check that output tensors with external edges are mapped
    for (auto mapped_ops_it = xfer->mappedOps.cbegin();
         mapped_ops_it != xfer->mappedOps.cend() && !fail; ++mapped_ops_it) {
      if (outEdges.find(mapped_ops_it->first) != outEdges.end()) {
        const std::set<Edge, EdgeCompare> &list =
            outEdges.find(mapped_ops_it->first)->second;
        for (auto edge_it = list.cbegin(); edge_it != list.cend() && !fail;
             ++edge_it)
          if (xfer->mappedOps.find(edge_it->dstOp) == xfer->mappedOps.end()) {
            // dstOp is external, (srcOp, srcIdx) must be in
            // mappedOutputs
            TensorX srcTen;
            srcTen.op = mapped_ops_it->second;
            srcTen.idx = edge_it->srcIdx;
            if (xfer->mappedOutputs.find(srcTen) == xfer->mappedOutputs.end()) {
              fail = true;
            }
          }
      }
    }
  }
  if (!fail) {
    auto new_graph = xfer->create_new_graph(this);
    if (new_graph->has_loop()) {
      fail = true;
    }
  }
  while (!opx_op_dq.empty()) {
    auto opx_op_pair = opx_op_dq.back();
    opx_op_dq.pop_back();
    xfer->unmatch(opx_op_pair.first, opx_op_pair.second, this);
  }
  if (!fail) {
    assert(mapped_opx.size() == xfer->srcOps.size());
  }
  return !fail;
}

std::shared_ptr<Graph> Graph::apply_xfer(GraphXfer *xfer, Op op,
                                         bool eliminate_rotation) {
  if (!xfer->can_match(*xfer->srcOps.begin(), op, this)) {
    return nullptr;
  }
  std::unordered_set<OpX *> mapped_opx;
  std::deque<std::pair<OpX *, Op>> opx_op_dq;
  xfer->match(*xfer->srcOps.begin(), op, this);
  mapped_opx.insert(*xfer->srcOps.begin());
  opx_op_dq.push_back(std::make_pair(*xfer->srcOps.begin(), op));
  // If an OpX is mapped to an Op, check whether their corresponding input OpX,
  // input Op, output OpX, output Op can match. Because the source graphs is
  // connected, by doing this we can traverse all nodes.
  bool fail = false;
  size_t idx = 0;
  while (idx < opx_op_dq.size()) {
    auto opx_op_pair = opx_op_dq[idx];
    idx++;
    auto opx_ = opx_op_pair.first;
    auto op_ = opx_op_pair.second;
    auto num_input = opx_->inputs.size();
    auto num_output = opx_->outputs.size();
    // Get all input and output Op of op_ in an ordered list
    std::vector<Op> input_ops(num_input, Op::INVALID_OP);
    std::vector<Op> output_ops(num_output, Op::INVALID_OP);
    std::vector<OpX *> output_opxs(num_output, nullptr);
    if (inEdges.find(op_) != inEdges.end()) {
      for (auto &e : inEdges.find(op_)->second) {
        assert((size_t)e.dstIdx < num_input);
        input_ops[e.dstIdx] = e.srcOp;
      }
    }
    for (size_t i = 0; i < num_input; ++i) {
      auto input_opx = opx_->inputs[i].op;
      if (input_opx != nullptr &&
          mapped_opx.find(input_opx) == mapped_opx.end()) {
        if (!xfer->can_match(input_opx, input_ops[i], this)) {
          fail = true;
          break;
        } else {
          xfer->match(input_opx, input_ops[i], this);
          mapped_opx.insert(input_opx);
          opx_op_dq.push_back(std::make_pair(input_opx, input_ops[i]));
        }
      }
    }
    if (fail) {
      break;
    }
    if (outEdges.find(op_) != outEdges.end()) {
      for (auto &e : outEdges.find(op_)->second) {
        assert((size_t)e.srcIdx < num_output);
        output_ops[e.srcIdx] = e.dstOp;
      }
    }
    // Get all output OpX because we don't have output edges in GraphXfer
    for (auto &opx : xfer->srcOps) {
      for (auto &input_tensor : opx->inputs) {
        if (input_tensor.op == opx_) {
          assert((size_t)input_tensor.idx < num_output);
          output_opxs[input_tensor.idx] = opx;
        }
      }
    }
    for (size_t i = 0; i < num_output; ++i) {
      auto output_opx = output_opxs[i];
      if (output_opx != nullptr &&
          mapped_opx.find(output_opx) == mapped_opx.end()) {
        if (!xfer->can_match(output_opx, output_ops[i], this)) {
          fail = true;
          break;
        } else {
          xfer->match(output_opx, output_ops[i], this);
          mapped_opx.insert(output_opx);
          opx_op_dq.push_back(std::make_pair(output_opx, output_ops[i]));
        }
      }
    }
    if (fail) {
      break;
    }
  }
  if (!fail) {
    // Check qubit consistancy
    std::set<int> qubits;
    for (auto it = xfer->mappedInputs.cbegin(); it != xfer->mappedInputs.cend();
         ++it) {
      if (it->second.first.ptr->is_quantum_gate() ||
          it->second.first.ptr->tp == GateType::input_qubit) {
        // Only check inputs on a qubit
        // Excluding input_param gates and arithmetic gates
        Pos p = Pos(it->second.first, it->second.second);
        auto q = pos_2_logical_qubit.find(p)->second;
        if (qubits.find(q) != qubits.end()) {
          fail = true;
          break;
        } else {
          qubits.insert(q);
        }
      }
    }
  }
  if (!fail) {
    for (auto dst_it = xfer->dstOps.cbegin(); dst_it != xfer->dstOps.cend();
         ++dst_it) {
      if (!fail) {
        OpX *dstOp = *dst_it;
        fail = !xfer->create_new_operator(dstOp, dstOp->mapOp);
      }
    }
  }
  if (!fail) {
    // Check that output tensors with external edges are mapped
    for (auto mapped_ops_it = xfer->mappedOps.cbegin();
         mapped_ops_it != xfer->mappedOps.cend() && !fail; ++mapped_ops_it) {
      if (outEdges.find(mapped_ops_it->first) != outEdges.end()) {
        const std::set<Edge, EdgeCompare> &list =
            outEdges.find(mapped_ops_it->first)->second;
        for (auto edge_it = list.cbegin(); edge_it != list.cend() && !fail;
             ++edge_it)
          if (xfer->mappedOps.find(edge_it->dstOp) == xfer->mappedOps.end()) {
            // dstOp is external, (srcOp, srcIdx) must be in
            // mappedOutputs
            TensorX srcTen;
            srcTen.op = mapped_ops_it->second;
            srcTen.idx = edge_it->srcIdx;
            if (xfer->mappedOutputs.find(srcTen) == xfer->mappedOutputs.end()) {
              fail = true;
            }
          }
      }
    }
  }
  std::shared_ptr<Graph> new_graph(nullptr);
  if (!fail) {
    new_graph = xfer->create_new_graph(this);
    if (new_graph->has_loop()) {
      new_graph.reset();
      fail = true;
    }
  }
  if (!fail) {
    if (eliminate_rotation) {
      new_graph->constant_and_rotation_elimination();
    }
  }
  while (!opx_op_dq.empty()) {
    auto opx_op_pair = opx_op_dq.back();
    opx_op_dq.pop_back();
    xfer->unmatch(opx_op_pair.first, opx_op_pair.second, this);
  }
  if (!fail)
    assert(mapped_opx.size() == xfer->srcOps.size());
  return new_graph;
}

std::pair<std::shared_ptr<Graph>, std::vector<int>>
Graph::apply_xfer_and_track_node(GraphXfer *xfer, Op op,
                                 bool eliminate_rotation) {
  // When eliminate_rotation is true, this function will eliminate all rotation
  // whose parameters are all 0
  if (!xfer->can_match(*xfer->srcOps.begin(), op, this)) {
    return std::make_pair(std::shared_ptr<Graph>(nullptr), std::vector<int>());
  }
  std::unordered_set<OpX *> mapped_opx;
  std::deque<std::pair<OpX *, Op>> opx_op_dq;
  xfer->match(*xfer->srcOps.begin(), op, this);
  mapped_opx.insert(*xfer->srcOps.begin());
  opx_op_dq.push_back(std::make_pair(*xfer->srcOps.begin(), op));
  // If an OpX is mapped to an Op, check whether their corresponding input OpX,
  // input Op, output OpX, output Op can match. Because the source graphs is
  // connected, by doing this we can traverse all nodes.
  bool fail = false;
  size_t idx = 0;
  while (idx < opx_op_dq.size()) {
    auto opx_op_pair = opx_op_dq[idx];
    idx++;
    auto opx_ = opx_op_pair.first;
    auto op_ = opx_op_pair.second;
    auto num_input = opx_->inputs.size();
    auto num_output = opx_->outputs.size();
    // Get all input and output Op of op_ in an ordered list
    std::vector<Op> input_ops(num_input, Op::INVALID_OP);
    std::vector<Op> output_ops(num_output, Op::INVALID_OP);
    std::vector<OpX *> output_opxs(num_output, nullptr);
    if (inEdges.find(op_) != inEdges.end()) {
      for (auto &e : inEdges.find(op_)->second) {
        assert((size_t)e.dstIdx < num_input);
        input_ops[e.dstIdx] = e.srcOp;
      }
    }
    for (size_t i = 0; i < num_input; ++i) {
      auto input_opx = opx_->inputs[i].op;
      if (input_opx != nullptr &&
          mapped_opx.find(input_opx) == mapped_opx.end()) {
        if (!xfer->can_match(input_opx, input_ops[i], this)) {
          fail = true;
          break;
        } else {
          xfer->match(input_opx, input_ops[i], this);
          mapped_opx.insert(input_opx);
          opx_op_dq.push_back(std::make_pair(input_opx, input_ops[i]));
        }
      }
    }
    if (fail) {
      break;
    }
    if (outEdges.find(op_) != outEdges.end()) {
      for (auto &e : outEdges.find(op_)->second) {
        assert((size_t)e.srcIdx < num_output);
        output_ops[e.srcIdx] = e.dstOp;
      }
    }
    // Get all output OpX because we don't have output edges in GraphXfer
    for (auto &opx : xfer->srcOps) {
      for (auto &input_tensor : opx->inputs) {
        if (input_tensor.op == opx_) {
          assert((size_t)input_tensor.idx < num_output);
          output_opxs[input_tensor.idx] = opx;
        }
      }
    }
    for (size_t i = 0; i < num_output; ++i) {
      auto output_opx = output_opxs[i];
      if (output_opx != nullptr &&
          mapped_opx.find(output_opx) == mapped_opx.end()) {
        if (!xfer->can_match(output_opx, output_ops[i], this)) {
          fail = true;
          break;
        } else {
          xfer->match(output_opx, output_ops[i], this);
          mapped_opx.insert(output_opx);
          opx_op_dq.push_back(std::make_pair(output_opx, output_ops[i]));
        }
      }
    }
    if (fail) {
      break;
    }
  }
  if (!fail) {
    // Check qubit consistancy
    std::set<int> qubits;
    for (auto it = xfer->mappedInputs.cbegin(); it != xfer->mappedInputs.cend();
         ++it) {
      if (it->second.first.ptr->is_quantum_gate() ||
          it->second.first.ptr->tp == GateType::input_qubit) {
        // Only check inputs on a qubit
        // Excluding input_param gates and arithmetic gates
        Pos p = Pos(it->second.first, it->second.second);
        auto q = pos_2_logical_qubit.find(p)->second;
        if (qubits.find(q) != qubits.end()) {
          fail = true;
          break;
        } else {
          qubits.insert(q);
        }
      }
    }
  }
  if (!fail) {
    for (auto dst_it = xfer->dstOps.cbegin(); dst_it != xfer->dstOps.cend();
         ++dst_it) {
      if (!fail) {
        OpX *dstOp = *dst_it;
        fail = !xfer->create_new_operator(dstOp, dstOp->mapOp);
      }
    }
  }
  if (!fail) {
    // Check that output tensors with external edges are mapped
    for (auto mapped_ops_it = xfer->mappedOps.cbegin();
         mapped_ops_it != xfer->mappedOps.cend() && !fail; ++mapped_ops_it) {
      if (outEdges.find(mapped_ops_it->first) != outEdges.end()) {
        const std::set<Edge, EdgeCompare> &list =
            outEdges.find(mapped_ops_it->first)->second;
        for (auto edge_it = list.cbegin(); edge_it != list.cend() && !fail;
             ++edge_it)
          if (xfer->mappedOps.find(edge_it->dstOp) == xfer->mappedOps.end()) {
            // dstOp is external, (srcOp, srcIdx) must be in
            // mappedOutputs
            TensorX srcTen;
            srcTen.op = mapped_ops_it->second;
            srcTen.idx = edge_it->srcIdx;
            if (xfer->mappedOutputs.find(srcTen) == xfer->mappedOutputs.end()) {
              fail = true;
            }
          }
      }
    }
  }
  std::shared_ptr<Graph> new_graph(nullptr);
  std::vector<int> node_trace;
  if (!fail) {
    new_graph = xfer->create_new_graph(this);
    if (new_graph->has_loop()) {
      new_graph.reset();
      fail = true;
    }
  }
  if (!fail) {
    std::unordered_set<Op, OpHash> op_set;
    // The destination graph in xfer is not an empty graph
    // Add all nodes in the destination graph to op_set
    if (!xfer->dstOps.empty()) {
      for (const auto &opx : xfer->dstOps) {
        if (opx->mapOp.ptr->is_quantum_gate()) {
          op_set.insert(opx->mapOp);
        }
      }
    }
    // Add all 1-hop predecessors to op_set
    for (auto it = xfer->srcOps.cbegin(); it != xfer->srcOps.cend(); ++it) {
      if (inEdges.find((*it)->mapOp) != inEdges.end()) {
        auto in_es = inEdges.find((*it)->mapOp)->second;
        for (auto e_it = in_es.cbegin(); e_it != in_es.cend(); ++e_it) {
          if (e_it->srcOp.ptr->is_quantum_gate()) {
            op_set.insert(e_it->srcOp);
          }
        }
      }
    }
    std::vector<Op> all_ops;
    if (eliminate_rotation) {
      new_graph->constant_and_rotation_elimination();
    }
    new_graph->topology_order_ops(all_ops);
    auto ops_num = all_ops.size();
    for (size_t i = 0; i < ops_num; ++i) {
      if (op_set.find(all_ops[i]) != op_set.end()) {
        node_trace.push_back(i);
      }
    }
  }
  while (!opx_op_dq.empty()) {
    auto opx_op_pair = opx_op_dq.back();
    opx_op_dq.pop_back();
    xfer->unmatch(opx_op_pair.first, opx_op_pair.second, this);
  }
  if (!fail)
    assert(mapped_opx.size() == xfer->srcOps.size());
  return std::make_pair(new_graph, node_trace);
}

std::vector<size_t>
Graph::appliable_xfers(Op op, const std::vector<GraphXfer *> &xfer_v) const {
  std::vector<size_t> appliable_xfer_v;
  auto xfer_v_s = xfer_v.size();
  for (size_t i = 0; i < xfer_v_s; ++i)
    if (xfer_appliable(xfer_v[i], op)) {
      appliable_xfer_v.push_back(i);
    }
  return appliable_xfer_v;
}

std::vector<size_t>
Graph::appliable_xfers_parallel(Op op,
                                const std::vector<GraphXfer *> &xfer_v) const {
  // cannot use std::vector<bool> here because it's not thread-safe to write
  // different elements of it
  std::vector<int> xfer_is_appliable(xfer_v.size(), false);
#pragma omp parallel for schedule(runtime) default(none)                       \
    shared(xfer_is_appliable, xfer_v, op)
  for (size_t i = 0; i < xfer_v.size(); i++) {
    xfer_is_appliable[i] = xfer_appliable(xfer_v[i], op);
  }
  std::vector<size_t> appliable_xfer_v;
  for (size_t i = 0; i < xfer_is_appliable.size(); i++) {
    if (xfer_is_appliable[i]) {
      appliable_xfer_v.emplace_back(i);
    }
  }
  return appliable_xfer_v;
}

// bool Graph::xfer_appliable(GraphXfer *xfer, Op op) const {
//   for (auto it = xfer->srcOps.begin(); it != xfer->srcOps.end(); ++it) {
//     // Find a match for the given Op
//     if (xfer->can_match(*it, op, this)) {
//       xfer->match(*it, op, this);
//       bool rest_match = _match_rest_ops(xfer, 0, it - xfer->srcOps.begin(),
//                                         op.guid) != nullptr;
//       xfer->unmatch(*it, op, this);
//       if (rest_match)
//         return true;
//     }
//   }
//   return false;
// }

// std::shared_ptr<Graph> Graph::_match_rest_ops(GraphXfer *xfer, size_t depth,
//                                               size_t ignore_depth,
//                                               size_t min_guid) const {
//   // The parameter min_guid is the guid of the first mapped Op
//   if (depth == xfer->srcOps.size()) {
//     // Create dst operators
//     bool pass = true;
//     for (auto dst_it = xfer->dstOps.cbegin(); dst_it != xfer->dstOps.cend();
//          ++dst_it) {
//       if (pass) {
//         OpX *dstOp = *dst_it;
//         pass = (pass & xfer->create_new_operator(dstOp, dstOp->mapOp));
//       }
//     }
//     if (!pass)
//       return std::shared_ptr<Graph>(nullptr);
//     // Check that output tensors with external edges are mapped
//     for (auto mapped_ops_it = xfer->mappedOps.cbegin();
//          mapped_ops_it != xfer->mappedOps.cend(); ++mapped_ops_it) {
//       if (outEdges.find(mapped_ops_it->first) != outEdges.end()) {
//         const std::set<Edge, EdgeCompare> &list =
//             outEdges.find(mapped_ops_it->first)->second;
//         for (auto edge_it = list.cbegin(); edge_it != list.cend(); ++edge_it)
//           if (xfer->mappedOps.find(edge_it->dstOp) == xfer->mappedOps.end())
//           {
//             // dstOp is external, (srcOp, srcIdx) must be in
//             // mappedOutputs
//             TensorX srcTen;
//             srcTen.op = mapped_ops_it->second;
//             srcTen.idx = edge_it->srcIdx;
//             if (xfer->mappedOutputs.find(srcTen) ==
//             xfer->mappedOutputs.end()) {
//               return std::shared_ptr<Graph>(nullptr);
//             }
//           }
//       }
//     }

//     auto new_graph = xfer->create_new_graph(this);
//     if (new_graph->has_loop()) {
//       new_graph.reset();
//       return new_graph;
//     }
//     return new_graph;
//   }
//   if (depth == ignore_depth) {
//     return _match_rest_ops(xfer, depth + 1, ignore_depth, min_guid);
//   }
//   OpX *srcOp = xfer->srcOps[depth];
//   for (auto it = inEdges.cbegin(); it != inEdges.cend(); ++it) {
//     if (it->first.guid < min_guid)
//       continue;
//     if ((xfer->mappedOps.find(it->first) == xfer->mappedOps.end()) &&
//         xfer->can_match(srcOp, it->first, this)) {
//       Op match_op = it->first;
//       // Check mapOutput
//       xfer->match(srcOp, match_op, this);
//       auto new_graph = _match_rest_ops(xfer, depth + 1, ignore_depth,
//       min_guid); xfer->unmatch(srcOp, match_op, this); if (new_graph !=
//       nullptr)
//         return new_graph;
//     }
//   }
//   return std::shared_ptr<Graph>(nullptr);
// }

// std::shared_ptr<Graph> Graph::apply_xfer(GraphXfer *xfer, Op op) {
//   for (auto it = xfer->srcOps.begin(); it != xfer->srcOps.end(); ++it) {
//     // Find a match for the given Op
//     if (xfer->can_match(*it, op, this)) {
//       xfer->match(*it, op, this);
//       auto new_graph =
//           _match_rest_ops(xfer, 0, it - xfer->srcOps.begin(), op.guid);
//       xfer->unmatch(*it, op, this);
//       if (new_graph.get() != nullptr) {
//         return new_graph;
//       }
//     }
//   }
//   return std::shared_ptr<Graph>(nullptr);
// }

void Graph::all_ops(std::vector<Op> &ops) {
  for (auto it = inEdges.cbegin(); it != inEdges.cend(); ++it) {
    ops.push_back(it->first);
  }
}

void Graph::all_edges(std::vector<Edge> &edges) {
  for (auto it = outEdges.cbegin(); it != outEdges.cend(); ++it) {
    if (it->first.ptr->tp != GateType::input_qubit &&
        it->first.ptr->tp != GateType::input_param)
      edges.insert(edges.end(), it->second.begin(), it->second.end());
  }
}

void Graph::topology_order_ops(std::vector<Op> &ops) const {
  std::unordered_map<Op, int, OpHash> op_in_degree;
  std::queue<Op> op_q;
  for (auto it = outEdges.cbegin(); it != outEdges.cend(); ++it) {
    if (it->first.ptr->tp == GateType::input_qubit ||
        it->first.ptr->tp == GateType::input_param) {
      op_q.push(it->first);
    }
  }

  for (auto it = inEdges.cbegin(); it != inEdges.cend(); ++it) {
    op_in_degree[it->first] = it->second.size();
  }

  while (!op_q.empty()) {
    auto op = op_q.front();
    op_q.pop();
    if (outEdges.find(op) != outEdges.end()) {
      auto op_out_edges = outEdges.find(op)->second;
      for (auto e_it = op_out_edges.cbegin(); e_it != op_out_edges.cend();
           ++e_it) {
        assert(op_in_degree[e_it->dstOp] > 0);
        op_in_degree[e_it->dstOp]--;
        if (op_in_degree[e_it->dstOp] == 0) {
          ops.push_back(e_it->dstOp);
          op_q.push(e_it->dstOp);
        }
      }
    }
  }
}

// TODO
bool Graph::equal(const Graph &other) const { return true; }

void Graph::init_physical_mapping(InitialMappingType mapping_type, const std::shared_ptr<DeviceTopologyGraph>& device,
                                  int pass, bool use_extensive, double w_value) {
    // STEP 1: initial mapping
    switch (mapping_type) {
        case InitialMappingType::TRIVIAL:
            _trivial_mapping();
            break;
        case InitialMappingType::SABRE:
            _sabre_mapping(device, pass, use_extensive, w_value);
            break;
        default:
            std::cout << "Unrecognized initial mapping type\n";
            assert(false);
    }

    // STEP 2: propagate through the circuit
    propagate_mapping();
}

void Graph::_trivial_mapping() {
    // Trivial mapping: input gate i -> phy = log = i
    int qubit_index = 0;
    for (const auto& op_out_edge : outEdges) {
        if (op_out_edge.first.ptr->tp == GateType::input_qubit) {
            qubit_mapping_table.insert({op_out_edge.first, std::pair<int, int>(qubit_index, qubit_index)});
            qubit_index += 1;
        }
    }
}

void Graph::_sabre_mapping(const std::shared_ptr<DeviceTopologyGraph>& device, int pass,
                           bool use_extensive, double w_value) {
    // set a random initial mapping
    _trivial_mapping();
    size_t num_logical_qubits = qubit_mapping_table.size();
    size_t num_physical_qubits = device->get_num_qubits();

    // generate a random permutation
    std::vector<int> permutation;
    permutation.reserve(num_physical_qubits);
    for (int i = 0; i < num_physical_qubits; ++i) {
        permutation.emplace_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(permutation.begin(), permutation.end(), g);

    // generate a random initial mapping from this permutation
    std::vector<int> l2p;
    l2p.reserve(num_logical_qubits);
    for (int i = 0; i < num_logical_qubits; ++i) {
        l2p.emplace_back(permutation[i]);
    }
    set_physical_mapping(l2p);

    // perform sabre search
    for (int i = 0; i < pass; ++i) {
        QubitMappingTable sabre_mapping = calculate_sabre_mapping(*this, device, use_extensive, w_value);
        qubit_mapping_table = sabre_mapping;
    }
}

void Graph::propagate_mapping() {
    std::queue<Edge> query_list;
    // initialize edges directed connected to input_qubits and put them into query list
    for (const auto& input_qubits_edges: qubit_mapping_table) {
        auto cur_input_qubit = input_qubits_edges.first;
        if (outEdges.find(cur_input_qubit) != outEdges.end() &&
            !outEdges[cur_input_qubit].empty()) {
            auto& initial_edge = *(outEdges[input_qubits_edges.first].begin());
            initial_edge.logical_qubit_idx = input_qubits_edges.second.first;
            initial_edge.physical_qubit_idx = input_qubits_edges.second.second;
            query_list.push(initial_edge);
        } else {
            std::cout << "A qubit in mapping table has no successor gates" << std::endl;
            assert(false);
        }
        // std::cout << initial_edge.dstOp.guid << std::endl;
    }
    // scan through the circuit until the end
    while (!query_list.empty()) {
        // get a mapped edge
        auto cur_edge = query_list.front();
        query_list.pop();
        // set mapping info for its target op
        auto target_op = cur_edge.dstOp;
        auto port_idx = cur_edge.dstIdx;
        // std::cout << "Target Op is" << target_op.guid << " port " << port_idx << std::endl;
        if (cur_edge.dstOp != Op::INVALID_OP){
            // set in edge of target op
            for (auto& in_edge : inEdges[target_op]) {
                if (in_edge.dstIdx == port_idx) {
                    in_edge.physical_qubit_idx = cur_edge.physical_qubit_idx;
                    in_edge.logical_qubit_idx = cur_edge.logical_qubit_idx;
                }
            }
            // set out edge of target op (need to differentiate swap and other gates)
            if (target_op.ptr->tp == GateType::swap) {
                if (outEdges.find(target_op) != outEdges.end()) {
                    for (auto& edge : outEdges[target_op]) {
                        // notice that a swap has two outputs (can have less due to termination)
                        if (edge.srcIdx == port_idx) edge.logical_qubit_idx = cur_edge.logical_qubit_idx;
                        else edge.physical_qubit_idx = cur_edge.physical_qubit_idx;
                    }
                }
            } else {
                if (outEdges.find(target_op) != outEdges.end()) {
                    for (auto& edge : outEdges[target_op]) {
                        if (edge.srcIdx != port_idx) continue;
                        edge.logical_qubit_idx = cur_edge.logical_qubit_idx;
                        edge.physical_qubit_idx = cur_edge.physical_qubit_idx;
                    }
                }
            }
            // put out edge into query list
            if (target_op.ptr->tp == GateType::swap) {
                // swap gates put out edges when both out edges are finished
                int finished_edge = 0;
                if (outEdges.find(target_op) != outEdges.end()) {
                    for (const auto& edge : outEdges[target_op]) {
                        if (edge.physical_qubit_idx != -1 && edge.logical_qubit_idx != -1) finished_edge++;
                    }
                    if (finished_edge == outEdges[target_op].size()) {
                        for (const auto& edge : outEdges[target_op]) {
                            query_list.push(edge);
                        }
                    }
                }
            } else {
                if (outEdges.find(target_op) != outEdges.end()) {
                    for (const auto& edge : outEdges[target_op]) {
                        if (edge.srcIdx == port_idx) query_list.push(edge);
                    }
                }
            }
        }
    }
}

MappingStatus Graph::check_mapping_correctness() {
    // build a map of Op -> <in_edges, out_edges>
    struct EdgePacket{
        EdgePacket() = default;
        const std::set<Edge, EdgeCompare>* in_edges;
        const std::set<Edge, EdgeCompare>* out_edges;
    };
    std::map<Op, EdgePacket, OpCompare> edge_map;
    for (const auto &op_out: outEdges) edge_map[op_out.first].out_edges = &op_out.second;
    for (const auto &op_in: inEdges) edge_map[op_in.first].in_edges = &op_in.second;

    // check correctness on each Op
    for (const auto &op_edge: edge_map) {
        // std::cout << op_edge.first.guid << std::endl;
        // input_param always passes
        if (op_edge.first.ptr->tp == GateType::input_param) continue;

        // final gates also passes
        if (!op_edge.second.out_edges || op_edge.second.out_edges->empty()) {
            // case for gates with no output (i.e. final gates)
            continue;
        }

        // input_qubit only needs to check out edges
        if (op_edge.first.ptr->tp == GateType::input_qubit) {
            // each input qubit has one out edge
            if (op_edge.second.out_edges->size() != 1) return MappingStatus::INPUT_QUBIT_TOO_MANY_OUTPUTS;
            // each input qubit has no input edge
            if (op_edge.second.in_edges && !op_edge.second.in_edges->empty()) return MappingStatus::INPUT_QUBIT_HAS_INPUT;
            // check if input qubit's physical and logical idx match the edge
            if (op_edge.second.out_edges->begin()->physical_qubit_idx != qubit_mapping_table[op_edge.first].second ||
                op_edge.second.out_edges->begin()->logical_qubit_idx != qubit_mapping_table[op_edge.first].first)
                return MappingStatus::INPUT_QUBIT_MAPPING_MISMATCH;
            continue;
        }

        // other Ops
        // check own consistency
        bool is_swap = (op_edge.first.ptr->tp == GateType::swap);
        for (const auto& in_edge : *op_edge.second.in_edges) {
            for (const auto& out_edge : *op_edge.second.out_edges) {
                if (is_swap) {
                    if (in_edge.dstIdx == 1 - out_edge.srcIdx &&
                        in_edge.physical_qubit_idx != out_edge.physical_qubit_idx)
                        return MappingStatus::SWAP_PHYSICAL_MISMATCH;
                    if (in_edge.dstIdx == out_edge.srcIdx &&
                        in_edge.logical_qubit_idx != out_edge.logical_qubit_idx)
                        return MappingStatus::SWAP_LOGICAL_MISMATCH;
                } else {
                    if (in_edge.dstIdx == out_edge.srcIdx) {
                        if (in_edge.physical_qubit_idx != out_edge.physical_qubit_idx)
                            return MappingStatus::NON_SWAP_PHYSICAL_MISMATCH;
                        if (in_edge.logical_qubit_idx != out_edge.logical_qubit_idx)
                            return MappingStatus::NON_SWAP_LOGICAL_MISMATCH;
                    }
                }
            }
        }
        // check consistency with next step
        for (const auto& out_edge : *op_edge.second.out_edges) {
            if (out_edge.dstOp != Op::INVALID_OP) {
                auto& target_op = out_edge.dstOp;
                for (const auto& next_in_edge : inEdges[target_op]) {
                    if (next_in_edge.dstIdx == out_edge.dstIdx) {
                        if (next_in_edge.logical_qubit_idx != out_edge.logical_qubit_idx ||
                            next_in_edge.physical_qubit_idx != out_edge.physical_qubit_idx)
                            return MappingStatus::SERIAL_MISMATCH;
                    }
                }
            }
        }
    }
    return MappingStatus::VALID;
}

double Graph::circuit_implementation_cost(const std::shared_ptr<DeviceTopologyGraph>& device) {
    double total_cost = 0;
    // Part1: cost of gates
    auto gate_count_cost = gate_count();
    total_cost += gate_count_cost;

    // Part2: minus cost of initial swaps gates, which are defined as follows:
    // 1. it's a swap
    // 2. both inputs have not gone through any gates other than swap
    std::unordered_set<Op, OpHash> initial_swap_set;
    std::unordered_set<Op, OpHash> candidate_0;
    std::unordered_set<Op, OpHash> candidate_1;
    for (const auto& input_qubit_mapping : qubit_mapping_table) {
        // start from input qubits
        const Edge* input_qubit_edge = &*outEdges[input_qubit_mapping.first].begin();
        bool has_next = true;
        while (has_next && input_qubit_edge->dstOp.ptr->tp == GateType::swap) {
            // store the swap gate in corresponding candidate set
            if (input_qubit_edge->dstIdx == 0) candidate_0.insert(input_qubit_edge->dstOp);
            if (input_qubit_edge->dstIdx == 1) candidate_1.insert(input_qubit_edge->dstOp);
            // move one step forward
            has_next = false;
            if (outEdges.find(input_qubit_edge->dstOp) != outEdges.end()) {
                for (const auto& next_edge : outEdges[input_qubit_edge->dstOp]) {
                    if (next_edge.srcIdx == input_qubit_edge->dstIdx) {
                        has_next = true;
                        input_qubit_edge = &next_edge;
                        break;
                    }
                }
            }
        }
    }
    for (const auto& candidate_op : candidate_0) {
        // an initial swap must be in both candidate0 and candidate1
        if (candidate_1.find(candidate_op) != candidate_1.end()) {
            // it's in candidate1
            initial_swap_set.insert(candidate_op);
        }
    }
    total_cost -= static_cast<double>(initial_swap_set.size());

    // Part3: calculate additional swap cost based on qubit distance in device
    for (const auto& Op_edge : inEdges) {
        // ignore the gate if it is initial swap
        if (initial_swap_set.find(Op_edge.first) != initial_swap_set.end()) { continue; }
        // ignore the gate if it has only one input
        if (Op_edge.second.size() == 1) { continue; }
        // calculate the min swap cost (only consider input 0 and input 1)
        // we assume that qubit 0 is moved to somewhere adjacent to qubit 1
        double min_swap_cost = device->unconnected_swap_penalty;
        const auto& first_qubit = Op_edge.second.begin();
        const auto& second_qubit = std::next(Op_edge.second.begin());
        auto input_neighbours = device->get_input_neighbours(second_qubit->physical_qubit_idx);
        for (const auto& neighbour : input_neighbours) {
            double swap_cost = 2 * device->cal_swap_cost(first_qubit->physical_qubit_idx, neighbour);
            min_swap_cost = std::min(min_swap_cost, swap_cost);
        }
        total_cost += min_swap_cost;
    }

    return total_cost;
}

void Graph::add_swap(const Edge& e1, const Edge& e2) {
    // TODO: check compatibility with other parts
    // create a swap gate and insert it into inEdges and outEdges
    auto new_swap_op = Op(context->next_global_unique_id(), context->get_gate(GateType::swap));
    auto in_edge_0 = Edge(e1.srcOp, new_swap_op, e1.srcIdx, 0);
    auto in_edge_1 = Edge(e2.srcOp, new_swap_op, e2.srcIdx, 1);
    auto out_edge_0 = Edge(new_swap_op, e1.dstOp, 0, e1.dstIdx);
    auto out_edge_1 = Edge(new_swap_op, e2.dstOp, 1, e2.dstIdx);
    inEdges[new_swap_op].insert(in_edge_0);
    inEdges[new_swap_op].insert(in_edge_1);
    outEdges[new_swap_op].insert(out_edge_0);
    outEdges[new_swap_op].insert(out_edge_1);

    // change e1 and e2's endpoints
    // e1 src
    for (auto& edge : outEdges[e1.srcOp]) {
        if (edge.srcIdx == e1.srcIdx) {
            edge.dstOp = new_swap_op;
            edge.dstIdx = 0;
        }
    }
    // e1 dst
    for (auto& edge : inEdges[e1.dstOp]) {
        if (edge.dstIdx == e1.dstIdx) {
            edge.srcOp = new_swap_op;
            edge.srcIdx = 0;
        }
    }
    // e2 src
    for (auto& edge : outEdges[e2.srcOp]) {
        if (edge.srcIdx == e2.srcIdx) {
            edge.dstOp = new_swap_op;
            edge.dstIdx = 1;
        }
    }
    // e2 dst
    for (auto& edge : inEdges[e2.dstOp]) {
        if (edge.dstIdx == e2.dstIdx) {
            edge.srcOp = new_swap_op;
            edge.srcIdx = 1;
        }
    }

    // re-calculate physical mapping
    propagate_mapping();
}

void Graph::set_physical_mapping(const std::vector<int>& logical2physical) {
    // set a trivial mapping
    qubit_mapping_table.clear();
    _trivial_mapping();
    // modify it to our target and propagate
    QubitMappingTable new_table;
    for (const auto& op_pair : qubit_mapping_table) {
        int logical_idx = op_pair.second.first;
        int physical_idx = logical2physical[logical_idx];
        new_table.insert({op_pair.first, std::pair<int, int>(logical_idx, physical_idx)});
    }
    qubit_mapping_table = new_table;
    propagate_mapping();
}

GraphState Graph::convert_circuit_to_state(int num_layers, bool include_forward_edge) {
    GraphState graph_state = GraphState();
    // only consider ops within #layers from front for better performance on GNN side
    std::set<Op, OpCompare> front_set = get_front_layers(num_layers, include_forward_edge);

    // map each op to a different node represented by id
    std::map<Op, int, OpCompare> op2node_id;
    int cur_node_id = 0;
    if (!outEdges.empty()) {
        // input qubit first
        for (const auto& op_edge: outEdges) {
            if (op_edge.first.ptr->tp == GateType::input_qubit) {
                op2node_id[op_edge.first] = cur_node_id;
                graph_state.node_id.emplace_back(cur_node_id);
                graph_state.is_input.emplace_back(true);
                graph_state.input_logical_idx.emplace_back(qubit_mapping_table[op_edge.first].first);
                graph_state.input_physical_idx.emplace_back(qubit_mapping_table[op_edge.first].second);
                graph_state.node_type.emplace_back(static_cast<int>(op_edge.first.ptr->tp));
                cur_node_id += 1;
            }
        }
    }
    if (!inEdges.empty()) {
        // all other gates (this is enough since each gate other than input qubit must have input)
        for (const auto& op_edge: inEdges) {
            // ignore ops not in front layers
            if (front_set.find(op_edge.first) == front_set.end()) continue;
            // insert node features otherwise
            assert(op_edge.first.ptr->tp != GateType::input_qubit);
            assert(op2node_id.find(op_edge.first) == op2node_id.end());
            op2node_id[op_edge.first] = cur_node_id;
            graph_state.node_id.emplace_back(cur_node_id);
            graph_state.is_input.emplace_back(false);
            graph_state.input_logical_idx.emplace_back(-1);
            graph_state.input_physical_idx.emplace_back(-1);
            graph_state.node_type.emplace_back(static_cast<int>(op_edge.first.ptr->tp));
            cur_node_id += 1;
        }
    }
    graph_state.number_of_nodes = cur_node_id;
    assert(graph_state.node_id.size() == cur_node_id);
    assert(graph_state.is_input.size() == cur_node_id);
    assert(graph_state.input_logical_idx.size() == cur_node_id);
    assert(graph_state.input_physical_idx.size() == cur_node_id);
    assert(graph_state.node_type.size() == cur_node_id);
    assert(front_set.size() == cur_node_id);

    // store all edges
    std::set<Edge, EdgeCompare> finished_edge_set;
    int edge_count = 0;
    if (!outEdges.empty()) {
        for (const auto& op_edge: outEdges) {
            assert(!op_edge.second.empty());
            // ignore ops not in front layers (since each valid edge must have both ends in front set)
            if (front_set.find(op_edge.first) == front_set.end()) continue;
            // consider the op's out edges otherwise
            for (const Edge& edge : op_edge.second) {
                if (finished_edge_set.find(edge) == finished_edge_set.end() &&
                    front_set.find(edge.dstOp) != front_set.end()) {
                    finished_edge_set.insert(edge);
                    // extract information
                    int logical_idx = edge.logical_qubit_idx;
                    int physical_idx = edge.physical_qubit_idx;
                    int src_node_idx = op2node_id[edge.srcOp];
                    int dst_node_idx = op2node_id[edge.dstOp];
                    // add edge to graph state
                    if (include_forward_edge) {
                        graph_state.edge_from.emplace_back(src_node_idx);
                        graph_state.edge_to.emplace_back(dst_node_idx);
                        graph_state.edge_logical_idx.emplace_back(logical_idx);
                        graph_state.edge_physical_idx.emplace_back(physical_idx);
                        graph_state.edge_reversed.emplace_back(false);
                        edge_count += 1;
                    }
                    // add reverse edge to graph state
                    graph_state.edge_from.emplace_back(dst_node_idx);
                    graph_state.edge_to.emplace_back(src_node_idx);
                    graph_state.edge_logical_idx.emplace_back(logical_idx);
                    graph_state.edge_physical_idx.emplace_back(physical_idx);
                    graph_state.edge_reversed.emplace_back(true);
                    edge_count += 1;
                }
            }
        }
    }
    graph_state.number_of_edges = edge_count;
    assert(graph_state.edge_from.size() == edge_count);
    assert(graph_state.edge_to.size() == edge_count);
    assert(graph_state.edge_logical_idx.size() == edge_count);
    assert(graph_state.edge_physical_idx.size() == edge_count);
    assert(graph_state.edge_reversed.size() == edge_count);
    if (!inEdges.empty()) {
        for (const auto& op_edge: inEdges) {
            assert(!op_edge.second.empty());
            if (front_set.find(op_edge.first) == front_set.end()) continue;
            for (const Edge& edge : op_edge.second) {
                if (front_set.find(edge.srcOp) == front_set.end()) continue;
                assert(finished_edge_set.find(edge) != finished_edge_set.end());
            }
        }
    }

    // return
    return graph_state;
}

std::map<Op, int, OpCompare> Graph::get_topology_ordering() {
    /// This function gives a topology ordering for each op in the circuit.
    /// Input qubits has ordering 0. Each layer of gates after input qubits has topology ordering +1.
    // initialize
    Graph tmp_graph = *this;
    std::map<Op, int, OpCompare> topology_ordering;

    // set ordering of input qubits as 0
    if (!outEdges.empty()) {
        // input qubit first
        for (const auto& op_edge: outEdges) {
            if (op_edge.first.ptr->tp == GateType::input_qubit) {
                assert(topology_ordering.find(op_edge.first) == topology_ordering.end());
                topology_ordering[op_edge.first] = 0;
            }
        }
    }

    // loop through the rest
    bool is_finished = false;
    int cur_topology_ordering = 1;
    while (!is_finished) {
        // find front gates and set ordering for them
        std::set<Op, OpCompare> front_gate_set = find_front_gates(tmp_graph);
        for (const Op& front_gate : front_gate_set) {
            assert(topology_ordering.find(front_gate) == topology_ordering.end());
            topology_ordering[front_gate] = cur_topology_ordering;
            execute_front_gate(tmp_graph, front_gate);
        }

        // execute gates and continue
        cur_topology_ordering += 1;
        is_finished = is_circuit_finished(tmp_graph);
    }

    return std::move(topology_ordering);
}

std::set<Op, OpCompare> Graph::get_front_layers(int num_layers, bool include_input) {
    /// This function returns all gates within #layers from circuit front.
    /// Note: 1. This function assumes that the graph has already been initialized.
    ///       2. This function is different from first #layers in topology ordering.
    ///          (use this in RL for faster computation)
    ///       3. set include_input = true if we consider both original and reversed edges
    ///          set include_input = false if we only considers reversed edges
    std::set<Op, OpCompare> front_set;

    // loop through all input qubits for front set
    for (const auto& op_mapping : qubit_mapping_table) {
        // initialize query set of current qubit
        std::set<Op> query_set;
        query_set.insert(op_mapping.first);

        // iterative query (input qubit layer is 0, so we need n+1 loops)
        for (int layer_idx = 0; layer_idx <= num_layers; ++layer_idx) {
            std::set<Op> next_query_set;

            // loop through all ops in current layer
            for (const Op& cur_op : query_set) {
                // put op in current layer into front set
                front_set.insert(cur_op);
                // put following ops into next layer's query list
                if (outEdges.find(cur_op) != outEdges.end() && !outEdges[cur_op].empty()) {
                    for (const Edge& edge : outEdges[cur_op]) {
                        auto dst_op = edge.dstOp;
                        next_query_set.insert(dst_op);
                    }
                }
                // also include input (since they may not be considered)
                if (include_input && inEdges.find(cur_op) != inEdges.end()) {
                    for (const Edge& edge: inEdges[cur_op]) {
                        auto src_op = edge.srcOp;
                        next_query_set.insert(src_op);
                    }
                }
            }

            // set query set for next layer
            query_set.clear();
            query_set = next_query_set;
        }
    }

    return front_set;
}
}; // namespace quartz
