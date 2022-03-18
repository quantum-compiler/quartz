#include "substitution.h"

namespace quartz {

OpX::OpX(const OpX &_op)
    : type(_op.type), mapOp(_op.mapOp), inputs(_op.inputs),
      outputs(_op.outputs) {}

OpX::OpX(GateType _type) : type(_type) {}

void OpX::add_input(const TensorX &input) { inputs.push_back(input); }

void OpX::add_output(const TensorX &output) { outputs.push_back(output); }

GraphXfer::GraphXfer(Context *_context) : context(_context), tensorId(10) {}

GraphXfer *GraphXfer::create_GraphXfer(Context *_context, const DAG *src_graph,
                                       const DAG *dst_graph) {
  // Remove common unused qubits
  assert(src_graph->get_num_qubits() == dst_graph->get_num_qubits());
  auto qubit_num = src_graph->get_num_qubits();
  std::vector<int> unused_qubits;
  for (int i = 0; i < qubit_num; ++i) {
    if (!src_graph->qubit_used(i) && !dst_graph->qubit_used(i))
      unused_qubits.push_back(i);
  }
  DAG *src_dag = new DAG(*src_graph), *dst_dag = new DAG(*dst_graph);
  bool ret = src_dag->remove_unused_qubits(unused_qubits);
  assert(ret);
  ret = dst_dag->remove_unused_qubits(unused_qubits);
  assert(ret);

  // TODO: remove common unused input parameters?

  // Eliminate transfers where src dag has unused qubits
  auto src_num_qubits = src_dag->get_num_qubits();
  for (int i = 0; i < src_num_qubits; ++i) {
    if (!src_dag->qubit_used(i))
      return nullptr;
  }

  // Eliminate transfers where src dag has unused input parameters
  auto src_num_input_params = src_dag->get_num_input_parameters();
  for (int i = 0; i < src_num_input_params; ++i) {
    if (!src_dag->input_param_used(i))
      return nullptr;
  }

  assert(src_dag->get_num_qubits() == dst_dag->get_num_qubits());
  assert(src_dag->get_num_input_parameters() ==
         dst_dag->get_num_input_parameters());
  GraphXfer *graphXfer = new GraphXfer(_context);
  std::unordered_map<DAGNode *, TensorX> src_to_tx, dst_to_tx;
  int cnt = 0;
  for (int i = 0; i < src_dag->get_num_qubits(); i++) {
    DAGNode *src_node = src_dag->nodes[cnt].get();
    DAGNode *dst_node = dst_dag->nodes[cnt++].get();
    assert(src_node->is_qubit());
    assert(dst_node->is_qubit());
    assert(src_node->index == i);
    assert(dst_node->index == i);
    TensorX qubit_tensor = graphXfer->new_tensor();
    src_to_tx[src_node] = qubit_tensor;
    dst_to_tx[dst_node] = qubit_tensor;
  }
  for (int i = 0; i < src_dag->get_num_input_parameters(); i++) {
    DAGNode *src_node = src_dag->nodes[cnt].get();
    DAGNode *dst_node = dst_dag->nodes[cnt++].get();
    assert(src_node->is_parameter());
    assert(dst_node->is_parameter());
    assert(src_node->index == i);
    assert(dst_node->index == i);
    TensorX parameter_tensor = graphXfer->new_tensor();
    src_to_tx[src_node] = parameter_tensor;
    dst_to_tx[dst_node] = parameter_tensor;
  }
  for (size_t i = 0; i < src_dag->edges.size(); i++) {
    DAGHyperEdge *e = src_dag->edges[i].get();
    OpX *op = new OpX(e->gate->tp);
    for (size_t j = 0; j < e->input_nodes.size(); j++) {
      assert(src_to_tx.find(e->input_nodes[j]) != src_to_tx.end());
      TensorX input = src_to_tx[e->input_nodes[j]];
      op->add_input(input);
    }
    for (size_t j = 0; j < e->output_nodes.size(); j++) {
      //   if (e->output_nodes[j]->is_qubit()) {
      //     TensorX output(op, j);
      //     op->add_output(output);
      //     src_to_tx[e->output_nodes[j]] = output;
      //   }
      TensorX output(op, j);
      op->add_output(output);
      src_to_tx[e->output_nodes[j]] = output;
    }
    graphXfer->srcOps.push_back(op);
  }
  for (size_t i = 0; i < dst_dag->edges.size(); i++) {
    DAGHyperEdge *e = dst_dag->edges[i].get();
    OpX *op = new OpX(e->gate->tp);
    for (size_t j = 0; j < e->input_nodes.size(); j++) {
      TensorX input = dst_to_tx[e->input_nodes[j]];
      op->add_input(input);
    }
    for (size_t j = 0; j < e->output_nodes.size(); j++) {
      //   if (e->output_nodes[j]->is_qubit()) {
      TensorX output(op, j);
      op->add_output(output);
      dst_to_tx[e->output_nodes[j]] = output;
      //   }
    }
    graphXfer->dstOps.push_back(op);
  }
  for (int i = 0; i < src_dag->get_num_qubits(); i++) {
    assert(src_to_tx.find(src_dag->outputs[i]) != src_to_tx.end());
    assert(dst_to_tx.find(dst_dag->outputs[i]) != dst_to_tx.end());
    graphXfer->map_output(src_to_tx[src_dag->outputs[i]],
                          dst_to_tx[dst_dag->outputs[i]]);
  }
  delete src_dag;
  delete dst_dag;
  return graphXfer;
}

GraphXfer *
GraphXfer::create_single_gate_GraphXfer(Context *union_ctx, Command src_cmd,
                                        std::vector<Command> dst_cmds) {
  // Currently only support source command with no constant parameters
  // Assume the only added parameters are constant parameters
  // Assume the number of non constant parameters are equal
  GateType src_tp = src_cmd.get_gate_type();
  GraphXfer *graphXfer = new GraphXfer(union_ctx);

  Gate *gate = union_ctx->get_gate(src_tp);
  auto num_qubit = gate->get_num_qubits();
  auto num_non_constant_params = gate->get_num_parameters();

  OpX *src_op = new OpX(src_tp);
  std::map<int, TensorX> dst_qubits_2_tensorx;
  std::map<int, TensorX> dst_params_2_tensorx;

  for (int i = 0; i < num_qubit; ++i) {
    TensorX qubit_tensor = graphXfer->new_tensor();
    src_op->add_input(qubit_tensor);
    dst_qubits_2_tensorx[i] = qubit_tensor;
  }

  for (int i = 0; i < num_non_constant_params; ++i) {
    TensorX param_tensor = graphXfer->new_tensor();
    src_op->add_input(param_tensor);
    dst_params_2_tensorx[i] = param_tensor;
  }

  for (int i = 0; i < num_qubit; ++i) {
    TensorX tensor(src_op, i);
    src_op->add_output(tensor);
  }
  graphXfer->srcOps.push_back(src_op);

  for (auto cmd : dst_cmds) {
    OpX *op = new OpX(cmd.get_gate_type());
    auto num_qubit = cmd.qubit_idx.size();
    for (size_t i = 0; i < num_qubit; ++i) {
      assert(dst_qubits_2_tensorx.find(cmd.qubit_idx[i]) !=
             dst_qubits_2_tensorx.end());
      op->add_input(dst_qubits_2_tensorx[cmd.qubit_idx[i]]);
      TensorX tensor(op, i);
      op->add_output(tensor);
      // Update output tensors
      dst_qubits_2_tensorx[cmd.qubit_idx[i]] = tensor;
    }
    auto num_params = cmd.param_idx.size();
    for (size_t i = 0; i < num_params; ++i) {
      // Non-constant parameters
      if (cmd.param_idx[i] != -1) {
        assert(dst_params_2_tensorx.find(cmd.param_idx[i]) !=
               dst_params_2_tensorx.end());
        op->add_input(dst_params_2_tensorx[cmd.param_idx[i]]);
      }
      // Constant parameters
      else {
        TensorX constant_param = graphXfer->new_tensor();
        graphXfer->paramValues[constant_param.idx] = cmd.constant_params[i];
        op->add_input(constant_param);
      }
    }
    graphXfer->dstOps.push_back(op);
  }
  for (int i = 0; i < num_qubit; ++i) {
    graphXfer->map_output(src_op->outputs[i],
                          dst_qubits_2_tensorx[src_cmd.qubit_idx[i]]);
  }
  return graphXfer;
}

std::pair<GraphXfer *, GraphXfer *> GraphXfer::ccz_cx_rz_xfer(Context *ctx) {
  Context dst_ctx({GateType::rz, GateType::cx, GateType::input_qubit,
                   GateType::input_param});
  std::pair<RuleParser *, RuleParser *> toffoli_rules =
      RuleParser::ccz_cx_rz_rules();
  std::vector<Command> cmds;
  Command cmd;
  toffoli_rules.first->find_convert_commands(&dst_ctx, GateType::ccz, cmd,
                                             cmds);
  GraphXfer *xfer_0 = create_single_gate_GraphXfer(ctx, cmd, cmds);
  toffoli_rules.second->find_convert_commands(&dst_ctx, GateType::ccz, cmd,
                                              cmds);
  GraphXfer *xfer_1 = create_single_gate_GraphXfer(ctx, cmd, cmds);
  delete toffoli_rules.first;
  delete toffoli_rules.second;
  return std::make_pair(xfer_0, xfer_1);
}

std::pair<GraphXfer *, GraphXfer *> GraphXfer::ccz_cx_u1_xfer(Context *ctx) {
  Context dst_ctx({GateType::u1, GateType::cx, GateType::input_qubit,
                   GateType::input_param});
  std::pair<RuleParser *, RuleParser *> toffoli_rules =
      RuleParser::ccz_cx_u1_rules();
  std::vector<Command> cmds;
  Command cmd;
  toffoli_rules.first->find_convert_commands(&dst_ctx, GateType::ccz, cmd,
                                             cmds);
  GraphXfer *xfer_0 = create_single_gate_GraphXfer(ctx, cmd, cmds);
  toffoli_rules.second->find_convert_commands(&dst_ctx, GateType::ccz, cmd,
                                              cmds);
  GraphXfer *xfer_1 = create_single_gate_GraphXfer(ctx, cmd, cmds);
  delete toffoli_rules.first;
  delete toffoli_rules.second;
  return std::make_pair(xfer_0, xfer_1);
}

std::pair<GraphXfer *, GraphXfer *> GraphXfer::ccz_cx_t_xfer(Context *ctx) {
  Context dst_ctx({GateType::t, GateType::tdg, GateType::cx,
                   GateType::input_qubit, GateType::input_param});
  std::pair<RuleParser *, RuleParser *> toffoli_rules =
      RuleParser::ccz_cx_t_rules();
  std::vector<Command> cmds;
  Command cmd;
  toffoli_rules.first->find_convert_commands(&dst_ctx, GateType::ccz, cmd,
                                             cmds);
  GraphXfer *xfer_0 = create_single_gate_GraphXfer(ctx, cmd, cmds);
  toffoli_rules.second->find_convert_commands(&dst_ctx, GateType::ccz, cmd,
                                              cmds);
  GraphXfer *xfer_1 = create_single_gate_GraphXfer(ctx, cmd, cmds);
  delete toffoli_rules.first;
  delete toffoli_rules.second;
  return std::make_pair(xfer_0, xfer_1);
}

GraphXfer::GraphXfer(Context *_context, const DAG *src_graph,
                     const DAG *dst_graph)
    : context(_context), tensorId(10) {
  assert(src_graph->get_num_qubits() == dst_graph->get_num_qubits());
  assert(src_graph->get_num_input_parameters() ==
         dst_graph->get_num_input_parameters());
  std::unordered_map<DAGNode *, TensorX> src_to_tx, dst_to_tx;
  int cnt = 0;
  for (int i = 0; i < src_graph->get_num_qubits(); i++) {
    DAGNode *src_node = src_graph->nodes[cnt].get();
    DAGNode *dst_node = dst_graph->nodes[cnt++].get();
    assert(src_node->is_qubit());
    assert(dst_node->is_qubit());
    assert(src_node->index == i);
    assert(dst_node->index == i);
    TensorX qubit_tensor = new_tensor();
    src_to_tx[src_node] = qubit_tensor;
    dst_to_tx[dst_node] = qubit_tensor;
  }
  for (int i = 0; i < src_graph->get_num_input_parameters(); i++) {
    DAGNode *src_node = src_graph->nodes[cnt].get();
    DAGNode *dst_node = dst_graph->nodes[cnt++].get();
    assert(src_node->is_parameter());
    assert(dst_node->is_parameter());
    assert(src_node->index == i);
    assert(dst_node->index == i);
    TensorX parameter_tensor = new_tensor();
    src_to_tx[src_node] = parameter_tensor;
    dst_to_tx[dst_node] = parameter_tensor;
  }
  for (size_t i = 0; i < src_graph->edges.size(); i++) {
    DAGHyperEdge *e = src_graph->edges[i].get();
    OpX *op = new OpX(e->gate->tp);
    for (size_t j = 0; j < e->input_nodes.size(); j++) {
      assert(src_to_tx.find(e->input_nodes[j]) != src_to_tx.end());
      TensorX input = src_to_tx[e->input_nodes[j]];
      op->add_input(input);
    }
    for (size_t j = 0; j < e->output_nodes.size(); j++) {
      //   if (e->output_nodes[j]->is_qubit()) {
      //     TensorX output(op, j);
      //     op->add_output(output);
      //     src_to_tx[e->output_nodes[j]] = output;
      //   }
      TensorX output(op, j);
      op->add_output(output);
      src_to_tx[e->output_nodes[j]] = output;
    }
    srcOps.push_back(op);
  }
  for (size_t i = 0; i < dst_graph->edges.size(); i++) {
    DAGHyperEdge *e = dst_graph->edges[i].get();
    OpX *op = new OpX(e->gate->tp);
    for (size_t j = 0; j < e->input_nodes.size(); j++) {
      TensorX input = dst_to_tx[e->input_nodes[j]];
      op->add_input(input);
    }
    for (size_t j = 0; j < e->output_nodes.size(); j++) {
      //   if (e->output_nodes[j]->is_qubit()) {
      TensorX output(op, j);
      op->add_output(output);
      dst_to_tx[e->output_nodes[j]] = output;
      //   }
    }
    dstOps.push_back(op);
  }
  for (int i = 0; i < src_graph->get_num_qubits(); i++) {
    assert(src_to_tx.find(src_graph->outputs[i]) != src_to_tx.end());
    assert(dst_to_tx.find(dst_graph->outputs[i]) != dst_to_tx.end());
    map_output(src_to_tx[src_graph->outputs[i]],
               dst_to_tx[dst_graph->outputs[i]]);
  }
}

TensorX GraphXfer::new_tensor(void) {
  TensorX t;
  t.op = NULL;
  t.idx = tensorId++;
  return t;
}

bool GraphXfer::map_output(const TensorX &src, const TensorX &dst) {
  mappedOutputs[src] = dst;
  return true;
}

bool GraphXfer::can_match(OpX *srcOp, Op op, const Graph *graph) const {
  // This function takes in an OpX, and will check all its input and
  // output tensors. If there are tensors connecting it with other already
  // mapped ops, check whether these edges exists in the given Graph. No
  // need to call this function with topological order. Because once both
  // the src op and the dst op are mapped, the edge connecting them will
  // be checked. This gauarentee that every edges are checked at the end.
  // Check gate type
  if (srcOp->type != op.ptr->tp)
    return false;
  // Check num input tensors
  if ((int)srcOp->inputs.size() !=
      op.ptr->get_num_qubits() + op.ptr->get_num_parameters())
    return false;
  // Check inputs
  std::unordered_map<int, std::pair<Op, int>> newMapInputs;
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) { // Input tensor
      auto it = mappedInputs.find(in.idx);
      if (it != mappedInputs.end()) {
        // Input is already mapped
        Op mappedOp = it->second.first;
        int mappedIdx = it->second.second;
        if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
          return false;
      } else {
        // Input haven't been mapped
        auto newit = newMapInputs.find(in.idx);
        if (newit != newMapInputs.end()) {
          Op mappedOp = newit->second.first;
          int mappedIdx = newit->second.second;
          if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
            return false;
        } else {
          std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
          std::set<Edge, EdgeCompare>::const_iterator it2;
          for (it2 = list.begin(); it2 != list.end(); it2++) {
            Edge e = *it2;
            if (e.dstIdx == (int)i) {
              newMapInputs.insert(
                  std::make_pair(in.idx, std::make_pair(e.srcOp, e.srcIdx)));
            }
          }
        }
      }
    } else {
      // Intermediate tensor
      // If the src op of the edge is not mapped, skip it
      if (in.op->mapOp.ptr == nullptr)
        continue;
      else if (!(graph->has_edge(in.op->mapOp, op, in.idx, i)))
        return false;
    }
    // Check output
    for (size_t i = 0; i < srcOp->outputs.size(); i++) {
      TensorX out = srcOp->outputs[i];
      auto it = mappedOutputs.find(out);
      // If out is in mappedOutputs, it represents an external edge,
      // we don't check it here
      if (it == mappedOutputs.end()) {
        // We have to find out the dst op of this output edge and
        // check its existence in the Graph
        bool found = false;
        for (auto mapped_ops_it = mappedOps.cbegin();
             mapped_ops_it != mappedOps.cend(); ++mapped_ops_it) {
          OpX *output_src_opx = mapped_ops_it->second;
          for (size_t j = 0; j < output_src_opx->inputs.size(); ++j) {
            auto input_tensor = output_src_opx->inputs[j];
            if (input_tensor.op == out.op && input_tensor.idx == out.idx) {
              found = true;
              if (!graph->has_edge(srcOp->mapOp, mapped_ops_it->first, i, j)) {
                return false;
              } else
                break;
            }
          }
          if (found)
            break;
        }
      }
    }
  }
  return true;
}

void GraphXfer::match(OpX *srcOp, Op op, const Graph *graph) {
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == nullptr) {
      // Input TensorX
      // Update mappedInputs
      std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
      std::set<Edge, EdgeCompare>::const_iterator it2;
      for (it2 = list.begin(); it2 != list.end(); it2++) {
        Edge e = *it2;
        if (e.dstIdx == (int)i) {
          mappedInputs.insert(
              std::make_pair(in.idx, std::make_pair(e.srcOp, e.srcIdx)));
        }
      }
    }
  }
  // Map srcOp to Op
  srcOp->mapOp = op;
  mappedOps[op] = srcOp;
}

void GraphXfer::unmatch(OpX *srcOp, Op op, const Graph *graph) {
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == nullptr) {
      // Update mappedInputsa
      auto it = mappedInputs.find(in.idx);
      mappedInputs.erase(it);
    }
  }
  // Unmap op
  mappedOps.erase(op);
  srcOp->mapOp.guid = 0;
  srcOp->mapOp.ptr = nullptr;
}

std::shared_ptr<Graph> GraphXfer::run_1_time(int depth, Graph *src_graph) {
  if (depth >= (int)srcOps.size()) {
    // Create dst operators
    bool pass = true;
    std::vector<OpX *>::const_iterator dstIt;
    for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
      if (pass) {
        OpX *dstOp = *dstIt;
        pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
      }
    if (!pass)
      return nullptr;
    // Check that output tensors with external edges are mapped
    for (auto opIt = mappedOps.cbegin(); opIt != mappedOps.cend(); opIt++) {
      const std::set<Edge, EdgeCompare> &list =
          src_graph->outEdges[opIt->first];
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mappedOps.find(it->dstOp) == mappedOps.end()) {
          // dstOp is external, (srcOp, srcIdx) must be in
          // mappedOutputs
          TensorX srcTen;
          srcTen.op = opIt->second;
          srcTen.idx = it->srcIdx;
          if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
            pass = false;
            return nullptr;
          }
        }
    }
    // Generate a new graph by applying xfer rule
    auto dst_graph = create_new_graph(src_graph);
    // Check that the new graph should not have any loop
    if (dst_graph->has_loop()) {
      std::cout << "Found a new graph with LOOP!!!!\n" << std::endl;
      //   delete dst_graph;
      return nullptr;
    }
    // TODO: remove me for better performance
    assert(dst_graph->check_correctness());
    if (dst_graph->hash() == src_graph->hash()) {
      return nullptr;
    }
    return dst_graph;
  } else {
    OpX *srcOp = srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = src_graph->inEdges.begin(); it != src_graph->inEdges.end();
         ++it) {
      // printf("can_match(%d)\n", can_match(srcOp, it->first,
      // graph));
      if ((mappedOps.find(it->first) == mappedOps.end()) &&
          can_match(srcOp, it->first, src_graph)) {
        Op op = it->first;
        // Check mapOutput
        match(srcOp, op, src_graph);
        auto dst_graph = run_1_time(depth + 1, src_graph);
        unmatch(srcOp, op, src_graph);
        if (dst_graph.get() != nullptr)
          return dst_graph;
      }
    }
  }
  return nullptr;
}

void GraphXfer::run(int depth, Graph *graph,
                    std::vector<std::shared_ptr<Graph>> &new_candidates,
                    std::set<size_t> &hashmap, float threshold, int maxNumOps,
                    bool enable_early_stop, bool &stop_search) {
  if (stop_search)
    return;
  // printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu)
  // candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(),
  // candidates.size());
  if (depth >= (int)srcOps.size()) {
    // Create dst operators
    bool pass = true;
    std::vector<OpX *>::const_iterator dstIt;
    for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
      if (pass) {
        OpX *dstOp = *dstIt;
        pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
      }
    if (!pass)
      return;
    // Check that all external edges are mapped outputs
    for (auto opIt = mappedOps.cbegin(); opIt != mappedOps.cend(); opIt++) {
      const std::set<Edge, EdgeCompare> &list = graph->outEdges[opIt->first];
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mappedOps.find(it->dstOp) == mappedOps.end()) {
          // dstOp is external, (srcOp, srcIdx) must be in
          // mappedOutputs
          TensorX srcTen;
          srcTen.op = opIt->second;
          srcTen.idx = it->srcIdx;
          if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
            pass = false;
            return;
          }
        }
    }
    // Generate a new graph by applying xfer rule
    // Graph *newGraph = create_new_graph(graph);
    std::shared_ptr<Graph> newGraph = create_new_graph(graph);
    // Check that the new graph should not have any loop
    if (newGraph->has_loop()) {
      // printf("Found a new graph with LOOP!!!!\n");
      return;
    }
    // TODO: remove me for better performance
    assert(newGraph->check_correctness());
    if (newGraph->total_cost() < threshold &&
        (int)newGraph->inEdges.size() < maxNumOps) {
      if (hashmap.find(newGraph->hash()) == hashmap.end()) {
        hashmap.insert(newGraph->hash());
        new_candidates.push_back(newGraph);
        if (enable_early_stop)
          stop_search = true;
        // std::cout << newGraph->total_cost() << " ";
      }
    }
  } else {
    OpX *srcOp = srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      // printf("can_match(%d)\n", can_match(srcOp, it->first,
      // graph));
      if ((mappedOps.find(it->first) == mappedOps.end()) &&
          can_match(srcOp, it->first, graph)) {
        Op op = it->first;
        // Check mapOutput
        match(srcOp, op, graph);
        run(depth + 1, graph, new_candidates, hashmap, threshold, maxNumOps,
            enable_early_stop, stop_search);
        unmatch(srcOp, op, graph);
      }
    }
  }
}

bool GraphXfer::create_new_operator(const OpX *opx, Op &op) {
  Gate *gate = context->get_gate(opx->type);
  op.ptr = gate;
  op.guid = context->next_global_unique_id();
  if (op == Op::INVALID_OP)
    return false;
  return true;
}

std::shared_ptr<Graph> GraphXfer::create_new_graph(const Graph *graph) const {
  std::shared_ptr<Graph> newGraph(new Graph(*graph));
  newGraph->inEdges.clear();
  newGraph->outEdges.clear();
  // Step 1: map dst ops
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
  std::vector<OpX *>::const_iterator dstIt;
  // Step 2: add edges to the graph
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mappedOps.find(opIt->first) == mappedOps.end()) {
      // Unmapped ops
      const std::set<Edge, EdgeCompare> &list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mappedOps.find(it->srcOp) != mappedOps.end()) {
          // mapped src -> unmapped dst
          TensorX srcTen;
          srcTen.op = mappedOps.find(it->srcOp)->second;
          srcTen.idx = it->srcIdx;
          assert(mappedOutputs.find(srcTen) != mappedOutputs.end());
          TensorX dstTen = mappedOutputs.find(srcTen)->second;
          if (dstTen.op == NULL) {
            // mappedOutput is an input --- this indicates
            // an empty target graph
            auto it2 = mappedInputs.find(dstTen.idx);
            assert(it2 != mappedInputs.end());
            std::pair<Op, int> srcEdge = it2->second;
            newGraph->add_edge(srcEdge.first, it->dstOp, srcEdge.second,
                               it->dstIdx);
          } else {
            newGraph->add_edge(dstTen.op->mapOp, it->dstOp, dstTen.idx,
                               it->dstIdx);
          }
        } else {
          // unmapped src -> unmmaped dst
          newGraph->add_edge(it->srcOp, it->dstOp, it->srcIdx, it->dstIdx);
        }
    }
  // Step 3: add edges for mapped ops
  for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++) {
    OpX *dstOp = *dstIt;
    for (size_t i = 0; i < dstOp->inputs.size(); i++)
      if (dstOp->inputs[i].op == NULL) {
        // unmapped src -> mapped dst
        if (paramValues.find(dstOp->inputs[i].idx) != paramValues.end()) {
          // New constant parameters
          Op input_constant_param_op(newGraph->get_next_special_op_guid(),
                                     context->get_gate(GateType::input_param));
          newGraph->constant_param_values[input_constant_param_op] =
              paramValues.find(dstOp->inputs[i].idx)->second;
          newGraph->add_edge(input_constant_param_op, dstOp->mapOp, 0, i);
          continue;
        };
        auto it = mappedInputs.find(dstOp->inputs[i].idx);
        assert(it != mappedInputs.end());
        std::pair<Op, int> srcEdge = it->second;
        newGraph->add_edge(srcEdge.first, dstOp->mapOp, srcEdge.second, i);
      } else {
        // mapped src -> mapped dst
        OpX *srcOp = dstOp->inputs[i].op;
        int srcIdx = dstOp->inputs[i].idx;
        newGraph->add_edge(srcOp->mapOp, dstOp->mapOp, srcIdx, i);
      }
  }
  return newGraph;
}
int GraphXfer::num_src_op() { return srcOps.size(); }
int GraphXfer::num_dst_op() { return dstOps.size(); }

}; // namespace quartz
