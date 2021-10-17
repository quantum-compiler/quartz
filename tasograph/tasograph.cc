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

Graph::Graph() : totalCost(0.0f) {}

Graph::Graph(Context *ctx, const DAG &dag) {
  // Guid for input qubit and input parameter nodes
  size_t special_op_guid = 0;
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
	    Op(special_op_guid++, ctx->get_gate(GateType::input_qubit)));
  for (int i = 0; i < num_input_params; ++i)
	input_params_op.push_back(
	    Op(special_op_guid++, ctx->get_gate(GateType::input_param)));

  // Map all edges in dag to Op
  std::map<DAGHyperEdge *, Op> edge_2_op;
  for (auto &edge : dag.edges) {
	auto e = edge.get();
	if (edge_2_op.find(e) == edge_2_op.end()) {
	  Op op(ctx->next_global_unique_id(), edge->gate);
	  edge_2_op[e] = op;
	}
  }

  std::cout << edge_2_op.size() << std::endl;

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

	// for (auto input_edge : node->input_edges) {
	//   int srcIdx;
	//   bool found = false;
	//   for (srcIdx = 0; srcIdx < input_edge->output_nodes.size(); ++srcIdx) {
	// 	if (node.get() == input_edge->output_nodes[srcIdx]) {
	// 	  found = true;
	// 	  break;
	// 	}
	//   }
	//   assert(found);
	//   assert(edge_2_op.find(input_edge) != edge_2_op.end());
	//   auto srcOp = edge_2_op[input_edge];

	//   for (auto output_edge : node->output_edges) {
	// 	int dstIdx;
	// 	bool found = false;
	// 	for (dstIdx = 0; dstIdx < output_edge->input_nodes.size(); ++dstIdx) {
	// 	  if (node.get() == output_edge->input_nodes[dstIdx]) {
	// 		found = true;
	// 		break;
	// 	  }
	// 	}
	// 	assert(found);
	// 	assert(edge_2_op.find(output_edge) != edge_2_op.end());
	// 	auto dstOp = edge_2_op[output_edge];

	// 	add_edge(srcOp, dstOp, srcIdx, dstIdx);
	//   }
	// }
  }

  totalCost = total_cost();
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

float Graph::total_cost(void) const {
  size_t cnt = 0;
  for (const auto &it : inEdges) {
	if (it.first.ptr->is_quantum_gate())
	  cnt++;
  }
  return (float)cnt;
}

Graph *Graph::optimize(float alpha, int budget, bool print_subst, Context *ctx,
                       const std::string &equiv_file_name) {
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

  // Normalize dags to minimal representations
  start = std::chrono::steady_clock::now();
  eqs.normalize_to_minimal_representations(ctx);
  end = std::chrono::steady_clock::now();
  std::cout << "After normalizing to minimal representations in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds, " << eqs.num_equivalence_classes()
            << " classes of equivalences with " << eqs.num_total_dags()
            << " DAGs are found." << std::endl;

  std::vector<GraphXfer *> xfers;
  for (const auto &item : eqs.dataset) {
	for (const auto &equiv_set : item.second) {
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
		  xfers.push_back(new GraphXfer(ctx, first_dag, other_dag));
		  // first_dag is dst, others are src
		  xfers.push_back(new GraphXfer(ctx, other_dag, first_dag));
		  delete other_dag;
		}
	  }
	  delete first_dag;
	}
  }

  int counter = 0;
  int maxNumOps = inEdges.size();

  std::priority_queue<Graph *, std::vector<Graph *>, GraphCompare> candidates;
  std::set<size_t> hashmap;
  candidates.push(this);
  hashmap.insert(hash());
  Graph *bestGraph = this;
  float bestCost = total_cost();

  printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
  while (!candidates.empty()) {
	Graph *subGraph = candidates.top();
	candidates.pop();
	if (subGraph->total_cost() < bestCost) {
	  delete bestGraph;
	  bestCost = subGraph->total_cost();
	  bestGraph = subGraph;
	}
	if (counter > budget) {
	  // TODO: free all remaining candidates when budget exhausted
	  break;
	}
	counter++;
	for (auto &xfer : xfers) {
	  xfer->run(0, subGraph, candidates, hashmap, bestCost * alpha,
	            2 * maxNumOps);
	}
	if (bestGraph != subGraph) {
	  delete subGraph;
	}
  }
  printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");
  // Print results
  return bestGraph;
}
}; // namespace TASOGraph
