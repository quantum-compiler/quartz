#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "quartz_api.h"
using namespace quartz;

// Construct contexts
Context gctxt({GateType::h, GateType::x, GateType::rz, GateType::add,
                  GateType::cx, GateType::input_qubit, GateType::input_param});

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
  std::cout << "sice returned = " << eccs.size() << std::endl;
  return eccs.size();
}

extern "C" int preprocess_ (const char* cqasm_, char* buffer, int buff_size) {
  std::string cqasm(cqasm_);
  std::cout << "len of str = " << cqasm.size() << std::endl;
  std::cout << "str = " << cqasm << std::endl;

  Context src_ctx({GateType::h, GateType::ccz, GateType::rz, GateType::x, GateType::cx,
                   GateType::input_qubit, GateType::input_param});
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param});
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);
  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&union_ctx);

  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm_str(cqasm, dag)) {
    std::cout << "Parser failed" << std::endl;
    return -1;
  }
  Graph graph(&src_ctx, dag);
  auto new_graph = graph.toffoli_flip_greedy(GateType::rz, xfer_pair.first, xfer_pair.second);
  std::string new_qasm = new_graph->to_qasm(false, false);
  return write_qasm_to_buffer (new_qasm, buffer, buff_size);
}


extern "C" int opt_circuit_ (const char* cqasm_, char* buffer, int buff_size, unsigned char* ecc_set_, long unsigned int ecc_set_size) {

  std::string cqasm(cqasm_);

  std::string eqset_fn = "../Nam_6_3_complete_ECC_set.json";

  Context ctxt({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param});
  QASMParser qasm_parser(&ctxt);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm_str(cqasm, dag)) {
    std::cout << "Parser failed" << std::endl;
    return -1;
  }
  Graph graph(&ctxt, dag);

  std::vector<CircuitSeq *> *arr = reinterpret_cast<std::vector<CircuitSeq *>*> (ecc_set_);
  std::vector<std::vector<CircuitSeq *>> eccs(arr, arr + ecc_set_size);

  auto start = std::chrono::steady_clock::now();
  auto graph_after_search = graph.greedy_optimize_with_eccs(&ctxt, eccs, /*print_message=*/ true, nullptr);
  auto end = std::chrono::steady_clock::now();

  std::cout << " on Nam's gate set."
            << " Gate count after optimization: "
            << graph_after_search->gate_count() << ", "
            << "Circuit depth: " << graph_after_search->circuit_depth() << ", "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;

  if (graph.total_cost() <= graph_after_search->total_cost()) {
    return -1;
  }
  std::string cqasm2 = graph_after_search->to_qasm(false, false);
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
