#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "quartz_api.h"
using namespace quartz;

// Construct contexts
Context gctxt({GateType::h, GateType::x, GateType::rz, GateType::add,
                  GateType::cx, GateType::input_qubit, GateType::input_param});
auto gcost_function = [](Graph *graph) { return graph->total_cost(); };

int write_qasm_to_buffer (std::string cqasm, char* buffer, int buff_size) {
  int blen = static_cast<int>(strlen(cqasm.c_str()));
  if (blen > buff_size) {
    return -1 * blen;
  } else {
    strcpy(buffer, cqasm.c_str());
    return blen;
  }
}

extern "C" long unsigned int load_eqset_ (const char* eqset_fn_, unsigned char** store) {
  std::string eqset_fn(eqset_fn_);
  EquivalenceSet* eqs = new EquivalenceSet();
  if (!eqs->load_json(&gctxt, eqset_fn)) {
    std::cout << "Failed to load equivalence file \"" << eqset_fn
              << "\"." << std::endl;
    assert(false);
  }
  std::vector<std::vector<CircuitSeq *>> eccs = eqs->get_all_equivalence_sets();
  std::vector<CircuitSeq *> *arr = new std::vector<CircuitSeq*>[eccs.size()];
  std::copy(eccs.begin(), eccs.end(), arr);
  *store = reinterpret_cast<unsigned char*> (arr);
  return eccs.size();
}

extern "C" long unsigned int load_greedy_xfers_ (const char* eqset_fn_, unsigned char** store) {
  std::string eqset_fn(eqset_fn_);
  EquivalenceSet* eqs = new EquivalenceSet();
  Context *ctxt = new Context({GateType::h, GateType::x, GateType::rz, GateType::add,
                  GateType::cx, GateType::input_qubit, GateType::input_param});
  if (!eqs->load_json(ctxt, eqset_fn)) {
    std::cout << "Failed to load equivalence file \"" << eqset_fn
              << "\"." << std::endl;
    assert(false);
  }
  std::vector<std::vector<CircuitSeq *>> eccs = eqs->get_all_equivalence_sets();
  std::vector<GraphXfer *> xfers;

  for (const auto &ecc : eccs) {
    const int ecc_size = (int)ecc.size();
    std::vector<Graph>* graphs = new std::vector<Graph>[ecc_size];
    std::vector<int> graph_cost;
    // graphs.reserve(ecc_size);
    graph_cost.reserve(ecc_size);
    for (auto &circuit : ecc) {
      graphs->emplace_back(ctxt, circuit);
      graph_cost.emplace_back(gcost_function(&(graphs->back())));
    }
    int representative_id =
        (int)(std::min_element(graph_cost.begin(), graph_cost.end()) -
              graph_cost.begin());
    for (int i = 0; i < ecc_size; i++) {
      if (graph_cost[i] != graph_cost[representative_id]) {
        auto xfer = GraphXfer::create_GraphXfer(ctxt, ecc[i],
                                                ecc[representative_id], true);
        if (xfer != nullptr) {
          xfers.push_back(xfer);
        }
      }
    }
  }
  std::cout << "greedy_optimize(): Number of xfers that reduce cost: "
              << xfers.size() << std::endl;

  std::vector<GraphXfer *> *vptr = new std::vector<GraphXfer*>(xfers);
  *store = reinterpret_cast<unsigned char*> (vptr);
  return xfers.size();
}

extern "C" int preprocess_ (const char* cqasm_, char* buffer, int buff_size) {

  std::string cqasm(cqasm_);
  // std::cout << "here with qasm "<< cqasm << std::endl;

  Context src_ctx({GateType::u1, GateType::h, GateType::ccz, GateType::rz, GateType::rx, GateType::x, GateType::cx,
                   GateType::input_qubit, GateType::input_param});

  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm_str(cqasm, dag)) {
    std::cout << "Parser failed" << std::endl;
    return -1;
  }
  Graph graph(&src_ctx, dag);

  // decompose ccz as cx and rz
  Context rem_ctx({GateType::u1, GateType::rx, GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param});
  auto imt_ctx = union_contexts(&src_ctx, &rem_ctx);
  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&imt_ctx);
  auto new_graph = graph.toffoli_flip_greedy(GateType::rz, xfer_pair.first, xfer_pair.second);
  // std::cout << "flipping done\n"<<  std::endl;

  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                  GateType::cx, GateType::input_qubit, GateType::input_param});
  auto uctx = union_contexts(&rem_ctx, &dst_ctx);
  // auto y_rule = "y = rz(0.5pi) q0; rz(0.5pi) q0; h q0; rz(0.5pi) q0; rz(0.5pi) q0; h q0;";
  RuleParser rules({"rx q0 p0 = h q0; rz q0 p0; h q0;", "u1 q0 p0 = rz q0 p0;"}); // TODO: check this.
  // std::cout << "contexy shifting " << new_graph->to_qasm (false, false) <<  std::endl;
  auto fin_graph = new_graph->context_shift(&rem_ctx, &dst_ctx, &uctx, &rules, false);
  // std::cout << "ruling done\n"<<  std::endl;

  std::string new_qasm = fin_graph->to_qasm(false, false);
  return write_qasm_to_buffer (new_qasm, buffer, buff_size);
}


extern "C" int opt_circuit_ (const char* cqasm_, char* buffer, int buff_size, unsigned char* xfers_) {

  std::string cqasm(cqasm_);

  std::vector<GraphXfer*>* xfers_ptr = reinterpret_cast<std::vector<GraphXfer*>*> (xfers_);
  std::vector<GraphXfer*> xfers = *xfers_ptr;

  // std::string eqset_fn = "Nam_4_3_complete_ECC_set.json";
  Context* ctxt;
  if (xfers.size () == 0) {
    ctxt = new Context({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param});
  } else {
    ctxt = xfers[0]->context;
  }

  QASMParser qasm_parser(ctxt);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm_str(cqasm, dag)) {
    std::cout << "Parser failed" << std::endl;
    return -1;
  }
  Graph graph(ctxt, dag);

  auto start = std::chrono::steady_clock::now();
  // Assume that the context is same?
  // std::cout << "calling greedy_opt" << std::endl;
  // auto graph_after_search = graph.greedy_optimize(ctxt, eqset_fn, /*print_message=*/ false);
  auto graph_after_search = graph.greedy_optimize_with_xfers(ctxt, xfers, /*print_message=*/ false, gcost_function);
  auto end = std::chrono::steady_clock::now();

  std::cout << " Gate count optimized from: "
            << graph.gate_count() << " to "
            << graph_after_search->gate_count() << ", "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;

  if (graph.total_cost() <= graph_after_search->total_cost()) {
    return -1;
  }
  std::string cqasm2 = graph_after_search->to_qasm(false, false);

  if (xfers.size() == 0) {
    std::cout << "deleting wrongs" << std::endl;
    delete ctxt;
  }
  return write_qasm_to_buffer(cqasm2, buffer, buff_size);
  // std::cout << "circuit after opt = ";
  // std::cout << cqasm2.c_str() << std::endl;
  // int blen = static_cast<int>(strlen(cqasm2.c_str()));
  // if (blen > buff_size) {
  //   return -1 * blen;
  // } else {
  //   strcpy(buffer, cqasm2.c_str());
  //   return blen;
  // }
}
