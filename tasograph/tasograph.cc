#include "tasograph.h"
#include "substitution.h"
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

Graph::Graph(Context *ctx)
    : context(ctx), special_op_guid(0), totalCost(0.0f) {}

Graph::Graph(Context *ctx, const DAG &dag) : context(ctx), special_op_guid(0) {
  // Guid for input qubit and input parameter nodes
  int num_input_qubits = dag.get_num_qubits();
  int num_input_params = dag.get_num_input_parameters();
  // Currently only 100 vacant guid
  assert(num_input_qubits + num_input_params <= 100);
  std::vector<Op> input_qubits_op;
  std::vector<Op> input_params_op;
  input_qubits_op.reserve(num_input_qubits);
  input_params_op.reserve(num_input_params);
  for (int i = 0; i < num_input_qubits; ++i)
	input_qubits_op.push_back(
	    Op(get_next_special_op_guid(), ctx->get_gate(GateType::input_qubit)));
  for (int i = 0; i < num_input_params; ++i)
	input_params_op.push_back(
	    Op(get_next_special_op_guid(), ctx->get_gate(GateType::input_param)));

  // Map all edges in dag to Op
  std::map<DAGHyperEdge *, Op> edge_2_op;
  for (auto &edge : dag.edges) {
	auto e = edge.get();
	if (edge_2_op.find(e) == edge_2_op.end()) {
	  Op op(ctx->next_global_unique_id(), edge->gate);
	  edge_2_op[e] = op;
	}
  }

  //   std::cout << edge_2_op.size() << std::endl;

  for (auto &node : dag.nodes) {
	int srcIdx = -1; // Assumption: a node can have at most 1 input
	Op srcOp;
	if (node->type == DAGNode::input_qubit) {
	  srcOp = input_qubits_op[node->index];
	  srcIdx = 0;
	}
	else if (node->type == DAGNode::input_param) {
	  srcOp = input_params_op[node->index];
	  srcIdx = 0;
	}
	else {
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
	  int dstIdx;
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

  totalCost = total_cost();
}

size_t Graph::get_next_special_op_guid() {
  special_op_guid++;
  assert(special_op_guid < 100);
  return special_op_guid;
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

bool Graph::has_edge(const Op &srcOp, const Op &dstOp, int srcIdx, int dstIdx) {
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return (inEdges[dstOp].find(e) != inEdges[dstOp].end());
}

Edge::Edge(void)
    : srcOp(Op::INVALID_OP), dstOp(Op::INVALID_OP), srcIdx(-1), dstIdx(-1) {}

Edge::Edge(const Op &_srcOp, const Op &_dstOp, int _srcIdx, int _dstIdx)
    : srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx) {}

bool Graph::has_loop(void) {
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
	int cnt = 0;
	std::set<Edge, EdgeCompare> inList = it->second;
	std::set<Edge, EdgeCompare>::const_iterator it2;
	for (it2 = inList.begin(); it2 != inList.end(); it2++) {
	  if (it2->srcOp.guid > GUID_PRESERVED)
		cnt++;
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
	  todos[it2->dstOp]--;
	  if (todos[it2->dstOp] == 0) {
		opList.push_back(it2->dstOp);
	  }
	}
  }
  return (opList.size() < inEdges.size());
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
  //   std::cout << total << std::endl;
  return total;
}

Graph *Graph::context_shift(Context *src_ctx, Context *dst_ctx,
                            RuleParser *rule_parser) {
  auto src_gates = src_ctx->get_supported_gates();
  auto dst_gate_set = std::set<GateType>(dst_ctx->get_supported_gates().begin(),
                                         dst_ctx->get_supported_gates().end());
  std::map<GateType, GraphXfer *> tp_2_xfer;
  for (auto gate_tp : src_gates) {
	if (dst_gate_set.find(gate_tp) == dst_gate_set.end()) {
	  std::vector<Command> cmds;
	  Command src_cmd;
	  assert(
	      rule_parser->find_convert_commands(dst_ctx, gate_tp, src_cmd, cmds));

	  tp_2_xfer[gate_tp] =
	      GraphXfer::create_single_gate_GraphXfer(src_cmd, dst_ctx, cmds);
	}
  }
  Graph *src_graph = this;
  Graph *dst_graph = nullptr;
  for (auto it = tp_2_xfer.begin(); it != tp_2_xfer.end(); ++it) {
	while ((dst_graph = it->second->run_1_time(0, src_graph)) != nullptr) {
	  if (src_graph != this)
		delete src_graph;
	  src_graph = dst_graph;
	}
  }
  return src_graph;
}

float Graph::total_cost(void) const {
  size_t cnt = 0;
  for (const auto &it : inEdges) {
	if (it.first.ptr->is_quantum_gate())
	  cnt++;
  }
  return (float)cnt;
}

void Graph::remove_node(Op oldOp) {
  assert(oldOp.ptr->tp != GateType::input_qubit);
  // Add edges between the inputs and outputs of the to-be removed node
  // Only add edges that connect qubits
  if (inEdges.find(oldOp) != inEdges.end() &&
      outEdges.find(oldOp) != outEdges.end()) {
	auto input_edges = inEdges[oldOp];
	auto output_edges = outEdges[oldOp];
	int num_qubits = oldOp.ptr->get_num_qubits();
	for (auto in_edge : input_edges) {
	  for (auto out_edge : output_edges) {
		if (in_edge.dstIdx == out_edge.srcIdx) {
		  if (in_edge.dstIdx < num_qubits) {
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

// Merge constant parameters
// Eliminate rotation with parameter 0
void Graph::constant_and_rotation_elimination() {
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::unordered_map<size_t, size_t> hash_values;
  std::queue<Op> op_queue;
  // Compute the hash value for input ops
  for (it = outEdges.begin(); it != outEdges.end(); it++) {
	if (it->first.ptr->tp == GateType::input_qubit ||
	    it->first.ptr->tp == GateType::input_param) {
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

		  Op merged_op(get_next_special_op_guid(),
		               context->get_gate(GateType::input_param));
		  add_edge(merged_op, output_dst_op, 0, output_dst_idx);
		  constant_param_values[merged_op] = result;
		}
		else if (op.ptr->tp == GateType::neg) {
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

		  Op merged_op(get_next_special_op_guid(),
		               context->get_gate(GateType::input_param));
		  add_edge(merged_op, output_dst_op, 0, output_dst_idx);
		  constant_param_values[merged_op] = result;
		}
		else {
		  assert(false && "Unimplemented parameter gates");
		}
	  }
	}
	else if (op.ptr->is_parametrized_gate()) {
	  // Rotation merging
	  auto input_edges = inEdges[op];
	  for (auto it = input_edges.begin(); it != input_edges.end(); ++it) {
		auto edge = *it;
		if (edge.srcOp.ptr->tp == op.ptr->tp) {
		  // Same rotation found, merge them
		  int num_qubits = op.ptr->get_num_qubits();
		  int num_params = op.ptr->get_num_parameters();
		  auto pre_rotation_op = edge.srcOp;
		  std::map<int, Op> pre_param_idx_2_op;
		  std::map<int, Op> param_idx_2_op;
		  assert(inEdges.find(pre_rotation_op) != inEdges.end());
		  auto pre_rotation_input_edges = inEdges[pre_rotation_op];
		  for (auto it = pre_rotation_input_edges.begin();
		       it != pre_rotation_input_edges.end(); ++it) {
			auto pre_rotation_edge = *it;
			if (pre_rotation_edge.dstIdx >= num_qubits) {
			  pre_param_idx_2_op[pre_rotation_edge.dstIdx] =
			      pre_rotation_edge.srcOp;
			}
		  }
		  for (auto it = input_edges.begin(); it != input_edges.end(); ++it) {
			auto edge = *it;
			if (edge.dstIdx >= num_qubits) {
			  param_idx_2_op[edge.dstIdx] = edge.srcOp;
			}
		  }
		  for (int i = num_qubits; i < num_qubits + num_params; ++i) {
			if (constant_param_values.find(pre_param_idx_2_op[i]) !=
			        constant_param_values.end() &&
			    constant_param_values.find(param_idx_2_op[i]) !=
			        constant_param_values.end()) {
			  ParamType sum = constant_param_values[pre_param_idx_2_op[i]] +
			                  constant_param_values[param_idx_2_op[i]];
			  remove_node(pre_param_idx_2_op[i]);
			  remove_node(param_idx_2_op[i]);
			  Op new_constant_op(get_next_special_op_guid(),
			                     context->get_gate(GateType::input_param));
			  add_edge(new_constant_op, op, 0, i);
			}
			else {
			  Op new_add_op(context->next_global_unique_id(),
			                context->get_gate(GateType::add));
			  add_edge(pre_param_idx_2_op[i], new_add_op, 0, 0);
			  add_edge(param_idx_2_op[i], new_add_op, 0, 1);
			}
		  }
		}
	  }

	  // Rotation gate
	  bool all_parameter_is_0 = true;
	  int num_qubits = op.ptr->get_num_qubits();
	  int num_params = op.ptr->get_num_parameters();
	  for (auto in_edge : input_edges) {
		if (in_edge.dstIdx >= num_qubits &&
		    in_edge.srcOp.ptr->is_parameter_gate()) {
		  all_parameter_is_0 = false;
		  break;
		}
		else if (in_edge.dstIdx >= num_qubits) {
		  if (constant_param_values.find(in_edge.srcOp) ==
		      constant_param_values.end()) {
			// Not a constant parameter
			all_parameter_is_0 = false;
			break;
		  }
		  else {
			// A constant parameter
			if (std::abs(constant_param_values[in_edge.srcOp]) > eps) {
			  // The constant parameter is not 0
			  all_parameter_is_0 = false;
			  break;
			}
		  }
		}
	  }
	  if (all_parameter_is_0) {
		remove_node(op);
	  }
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
}

#ifdef DEADCODE
void Graph::expand(std::pair<Op, int> pos, bool left,
                   std::unordered_set<std::pair<Op, int>> &covered) {
  covered.insert(pos);
  while (true) {
	if (left) {
	  if (!move_left(pos))
		return;
	}
	else {
	  if (!move_right(pos))
		return;
	}
	if (pos.first.ptr->tp == GateType::cnot) {
	  // Insert the other side of cnot to anchor_points;
	}
	else if (moveable(pos.frist.ptr->tp)) {
	  continue;
	}
	else {
	  break;
	}
  }
}

void Graph::remove(std::pair<Op, int> pos, bool left,
                   std::unordered_set<std::pair<Op, int>> &covered) {
  if (covered.find(pos) == covered.end())
	return;
  covered.remove(pos);
  if (/*pos is the control qubit of a cnot*/)
	remove(target_pos, left, covered);
  if (left) {
	if (!move_left(pos))
	  return;
  }
  else {
	if (!move_right(pos))
	  return;
  }
  remove(pos, covered);
}

void Graph::explore(std::pair<Op, int> pos, bool left,
                    std::unordered_set<std::pair<Op, int>> &covered) {
  while (true) {
	if (covered.find(pos) == covered.end())
	  return;
	if (/*pos is the target qubit of a cnot*/) {
	  remove(pos, left, covered);
	}
	else {
	  if (left) {
		if (!move_left(pos))
		  return;
	  }
	  else {
		if (!move_right(pos))
		  return;
	  }
	}
  }
}

bool Graph::rotation_merging(void) {
  std::unordered_set<Op> visited_cnot;
  // Step 1: calculate the bitmask of each operator
  std::unordered_map<std::pair<Op, int>, uint_128t> bitmasks;
  std::unordered_map<std::pair<Op, int>, int> op_to_qubits;
  std::queue<Op> todos;
  for (const auto &it : inEdges) {
	if (it.second.size() == 0) {
	  todos.push(it.first);
	  bitmasks[std::make_pair(it.first, 0)] = 1 << it.first.ptr->index;
	  op_to_qubits[std::make_pair(it.first, 0)] = it.first.ptr->index;
	}
  }
  while (todos.size() > 0) {
	auto op = todos.front();
	todos.pop();
	// TODO: explore the outEdges of op
	//
	if (op.ptr->tp == GateType::cnot) {
	  bitmasks[std::make_pair(op, 0)] = bitmasks[in0];
	  bitmasks[std::make_pair(op, 1)] = bitmasks[in0] xor bitmasks[in1];
	  op_to_qubits[std::make_pair(op, 0)] = op_to_qubits[in0];
	  op_to_qubits[std::make_pair(op, 1)] = op_to_qubits[in1];
	}
	else if (op.ptr->tp == GateType::x) {
	  bitmasks[std::makr_pair(op, 0)] = bitmasks[in0];
	  op_to_qubits[std::make_pair(op, 0)] = op_to_qubits[in0];
	}
	else {
	  bitmasks[std::make_pair(op, 0)] = bitmasks[in0];
	  op_to_qubits[std::make_pair(op, 0)] = op_to_qubits[in0];
	}
  }

  // Step 2: Propagate all CNOTs
  std::queue<Op> todo_cnot;
  for (const auto &it : inEdges)
	if (it.first.ptr->tp == GateType::cnot) {
	  todo_cnot.push(it.first);
	}
  while (todo_cnot.size() != 0) {
	const auto cnot = todo_cnot.front();
	todo_cnot.pop();
	if (visited_cnot.find(cnot) != visited_cnot.end())
	  continue;
	std::unordered_map<int, std::pair<Op, int>> anchor_point;
	anchor_point[op_to_qubits[std::make_pair(cnot, 0)]] =
	    std::make_pair(cnot, 0);
	anchor_point[op_to_qubits[std::make_pair(cnot, 1)]] =
	    std::make_pair(cnot, 1);
	std::queue<int> todo_qubits;
	todo_qubits.push(op_to_qubits[std::make_pair(cnot, 0)]);
	todo_qubits.push(op_to_qubits[std::make_pair(cnot, 1)]);
	std::unordered_set<std::pair<Op, int>> covered;
	while (todo_qubits.size() > 0) {
	  int qid = todo_qubits.front();
	  todo_qubits.pop();
	  expand(anchor_point[qid], true, covered);
	  expand(anchor_point[qid], false, covered);
	}
  }
  // Step 3: deal with partial cnot
  for (const auto &start_pos : anchor_point) {
	pos = start_pos;
	explore(pos, true, covered);
	explore(pos, false, covered);
  }
  // Step 4: merge rotations with the same bitmasks
  std::unordered_map<uint128_t, std::pair<Op, int>> bitmask_to_pos;
  for (const auto &it : covered) {
	if (it.first.op.ptr->ty == GateType::rz) {
	  uint128_t bm = bitmasks[it.first];
	  if (bitmask_to_pos(bm) != bitmask_to_pos.end()) {
		std::pair<Op, int> old_pos = bitmask_to_pos[bm];
		// remove it from the graph
		//
		// change the degree of old_pos
		old_pos.first.ptr->degree += it.first.ptr->degree;
	  }
	  else {
		bitmask_to_pos[bm] = it;
	  }
	}
  }
}
#endif

Graph *Graph::optimize(float alpha, int budget, bool print_subst, Context *ctx,
                       const std::string &equiv_file_name,
                       bool use_simulated_annealing) {
  EquivalenceSet eqs;
  // Load equivalent dags from file
  auto start = std::chrono::steady_clock::now();
  if (!eqs.load_json(ctx, equiv_file_name)) {
	std::cout << "Failed to load equivalence file." << std::endl;
	assert(false);
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << std::dec << eqs.num_equivalence_classes()
            << " classes of equivalences with " << eqs.num_total_dags()
            << " DAGs are loaded in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;

  //   start = std::chrono::steady_clock::now();
  //   auto num_equiv_class_inserted = eqs.simplify(ctx);
  //   end = std::chrono::steady_clock::now();
  //   std::cout << std::dec << eqs.num_equivalence_classes()
  //             << " classes of equivalences remain after simplication after "
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
	  }
	  else {
		DAG *other_dag = new DAG(*dag);
		// first_dag is src, others are dst
		if (first_dag->get_num_gates() != other_dag->get_num_gates()) {
		  std::cout << first_dag->get_num_gates() << " "
		            << other_dag->get_num_gates() << "; ";
		}
		auto first_2_other =
		    GraphXfer::create_GraphXfer(ctx, first_dag, other_dag);
		// first_dag is dst, others are src
		auto other_2_first =
		    GraphXfer::create_GraphXfer(ctx, other_dag, first_dag);
		if (first_2_other != nullptr)
		  xfers.push_back(first_2_other);
		else
		  std::cout << "nullptr"
		            << " ";
		if (other_2_first != nullptr)
		  xfers.push_back(other_2_first);
		else
		  std::cout << "nullptr"
		            << " ";
		delete other_dag;
	  }
	}
	delete first_dag;
  }

  std::cout << "Number of different transfers is " << xfers.size() << "."
            << std::endl;

  int counter = 0;
  int maxNumOps = inEdges.size();

  std::priority_queue<Graph *, std::vector<Graph *>, GraphCompare> candidates;
  std::set<size_t> hashmap;
  candidates.push(this);
  hashmap.insert(hash());
  Graph *bestGraph = this;
  float bestCost = total_cost();

  printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
  if (use_simulated_annealing) {
<<<<<<< HEAD
	const double kSABeginTemp = bestCost;
	const double kSAEndTemp = kSABeginTemp / 1e6;
	const double kSACoolingFactor = 1.0 - 1e-3;
	const int kNumKeepGraph = 50;
	// <cost, graph>
	std::vector<std::pair<float, Graph *>> sa_candidates;
	sa_candidates.reserve(kNumKeepGraph);
	sa_candidates.emplace_back(bestCost, this);
	int num_iteration = 0;
	std::cout << "Begin simulated annealing with " << xfers.size() << " xfers."
	          << std::endl;
	for (double T = kSABeginTemp; T > kSAEndTemp; T *= kSACoolingFactor) {
	  num_iteration++;
	  std::vector<std::pair<float, Graph *>> new_candidates;
	  new_candidates.reserve(sa_candidates.size() * xfers.size());
	  int num_possible_new_candidates = 0;
	  for (auto &candidate : sa_candidates) {
		const auto current_cost = candidate.first;
		std::vector<Graph *> current_new_candidates;
		current_new_candidates.reserve(xfers.size());
		for (auto &xfer : xfers) {
		  xfer->run(0, candidate.second, current_new_candidates, hashmap,
		            bestCost * alpha, 2 * maxNumOps);
		}
		num_possible_new_candidates += current_new_candidates.size();
		for (auto &new_candidate : current_new_candidates) {
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
		  }
		  else {
			delete new_candidate;
		  }
		}
	  }
	  std::cout << "Iteration " << num_iteration << ": bestcost = " << bestCost
	            << ", " << new_candidates.size() << " out of "
	            << num_possible_new_candidates
	            << " possible new candidates accepted." << std::endl;
	  if (new_candidates.size() > kNumKeepGraph) {
		// Prune some candidates.
		// TODO: make sure the candidates kept are far from each other
		// TODO: use hashmap to avoid keep searching for the same graphs
		std::partial_sort(new_candidates.begin(),
		                  new_candidates.begin() + kNumKeepGraph,
		                  new_candidates.end());
		new_candidates.resize(kNumKeepGraph);
	  }
	  sa_candidates = std::move(new_candidates);
	}
  }
  else {
	while (!candidates.empty()) {
	  Graph *subGraph = candidates.top();
	  candidates.pop();
	  if (subGraph->total_cost() < bestCost) {
		if (bestGraph != this)
		  delete bestGraph;
		bestCost = subGraph->total_cost();
		bestGraph = subGraph;
	  }
	  if (counter > budget) {
		// TODO: free all remaining candidates when budget exhausted
		//   break;
		;
	  }
	  counter++;

	  std::cout << bestCost << " " << std::flush;

	  std::vector<Graph *> new_candidates;
	  for (auto &xfer : xfers) {
		xfer->run(0, subGraph, new_candidates, hashmap, bestCost * alpha,
		          2 * maxNumOps);
	  }
	  for (auto &candidate : new_candidates) {
		candidates.push(candidate);
	  }
	  if (bestGraph != subGraph) {
		delete subGraph;
	  }
	}
=======
	const double kSABeginTemp = bestCost;
	const double kSAEndTemp = kSABeginTemp / 1e6;
	const double kSACoolingFactor = 1.0 - 1e-1;
	const int kNumKeepGraph = 50;
	// <cost, graph>
	std::vector<std::pair<float, Graph *>> sa_candidates;
	sa_candidates.reserve(kNumKeepGraph);
	sa_candidates.emplace_back(bestCost, this);
	int num_iteration = 0;
	std::cout << "Begin simulated annealing with " << xfers.size() << " xfers."
	          << std::endl;
	for (double T = kSABeginTemp; T > kSAEndTemp; T *= kSACoolingFactor) {
	  num_iteration++;
	  hashmap.clear();
	  std::vector<std::pair<float, Graph *>> new_candidates;
	  new_candidates.reserve(sa_candidates.size() * xfers.size());
	  int num_possible_new_candidates = 0;
	  for (auto &candidate : sa_candidates) {
		const auto current_cost = candidate.first;
		std::vector<Graph *> current_new_candidates;
		current_new_candidates.reserve(xfers.size());
		for (auto &xfer : xfers) {
		  xfer->run(0, candidate.second, current_new_candidates, hashmap,
		            bestCost * alpha, 2 * maxNumOps);
		}
		num_possible_new_candidates += current_new_candidates.size();
		for (auto &new_candidate : current_new_candidates) {
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
		  }
		  else {
			delete new_candidate;
		  }
		}
	  }

	  // Compute some statistical information to output, can be commented
	  // when verbose=false
	  const auto num_new_candidates = new_candidates.size();
	  assert(!new_candidates.empty());
	  auto min_cost = new_candidates[0].first;
	  auto max_cost = new_candidates[0].first;
	  for (const auto &new_candidate : new_candidates) {
		min_cost = std::min(min_cost, new_candidate.first);
		max_cost = std::max(max_cost, new_candidate.first);
	  }

	  if (new_candidates.size() > kNumKeepGraph) {
		// Prune some candidates.
		// TODO: make sure the candidates kept are far from each other
		// TODO: use hashmap to avoid keep searching for the same graphs
		std::partial_sort(new_candidates.begin(),
		                  new_candidates.begin() + kNumKeepGraph,
		                  new_candidates.end());
		for (int i = kNumKeepGraph; i < (int)new_candidates.size(); i++) {
		  if (new_candidates[i].second != this &&
		      new_candidates[i].second != bestGraph) {
			delete new_candidates[i].second;
		  }
		}
		new_candidates.resize(kNumKeepGraph);
	  }
	  sa_candidates = std::move(new_candidates);

	  std::cout << "Iteration " << num_iteration << ": T = " << std::fixed
	            << std::setprecision(2) << T << ", bestcost = " << bestCost
	            << ", " << num_new_candidates << " out of "
	            << num_possible_new_candidates
	            << " possible new candidates accepted, cost ranging ["
	            << min_cost << ", " << max_cost << "]" << std::endl;
	}
  }
  else {
	while (!candidates.empty()) {
	  Graph *subGraph = candidates.top();
	  candidates.pop();
	  if (subGraph->total_cost() < bestCost) {
		if (bestGraph != this)
		  delete bestGraph;
		bestCost = subGraph->total_cost();
		bestGraph = subGraph;
	  }
	  if (counter > budget) {
		// TODO: free all remaining candidates when budget exhausted
		//   break;
		;
	  }
	  counter++;

	  std::cout << bestCost << " " << std::flush;

	  std::vector<Graph *> new_candidates;
	  for (auto &xfer : xfers) {
		xfer->run(0, subGraph, new_candidates, hashmap, bestCost * alpha,
		          2 * maxNumOps);
	  }
	  for (auto &candidate : new_candidates) {
		candidates.push(candidate);
	  }
	  if (bestGraph != subGraph) {
		delete subGraph;
	  }
	}
>>>>>>> 42c6d79d51f597c2d5c45657fc5f42a2a0491c15
  }
  printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");
  // Print results
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::iterator it;
  for (it = bestGraph->inEdges.begin(); it != bestGraph->inEdges.end(); ++it) {
	std::cout << gate_type_name(it->first.ptr->tp) << std::endl;
  }
  return bestGraph;
}
}; // namespace TASOGraph
