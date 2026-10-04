#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace quartz;

int main() {
  const std::string input_fn =
      (kQuartzRootPath / "circuit/example-circuits/ccz_x_ccz.qasm").string();

  ParamInfo param_info;
  Context src_ctx({GateType::ccz, GateType::x, GateType::cx,
                   GateType::input_qubit, GateType::input_param},
                  &param_info);
  Context dst_ctx({GateType::rz, GateType::x, GateType::cx, GateType::add,
                   GateType::input_qubit, GateType::input_param},
                  &param_info);
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&src_ctx, &dst_ctx, &union_ctx);

  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cerr << "Parser failed to load " << input_fn << std::endl;
    return 1;
  }
  Graph graph(&src_ctx, dag);

  // Preprocess via toffoli_flip_greedy
  auto graph_decomposed = graph.toffoli_flip_greedy(
      GateType::rz, xfer_pair.first, xfer_pair.second);

  std::cout << "Original gates: " << graph.total_cost() << std::endl;
  std::cout << "Preprocessed gates: " << graph_decomposed->total_cost()
            << std::endl;

  // In bug #230, rotation_merging mistakenly treated X as moveable and incorrectly
  // merged rotations across the X gate, collapsing the circuit to 13 gates (12 CX + 1 X)
  // with 0 rotations, losing the CZ phase.
  // With the fix, rotations are not merged across X, preserving phase and equivalence.
  assert(graph_decomposed->total_cost() > 13);

  // Verify that rotation gates are preserved in the decomposed graph
  std::vector<Op> ops;
  graph_decomposed->topology_order_ops(ops);
  int rz_count = 0;
  for (const auto &op : ops) {
    if (op.ptr->tp == GateType::rz) {
      rz_count++;
    }
  }
  std::cout << "Preserved Rz rotations count: " << rz_count << std::endl;
  assert(rz_count > 0);

  // Verify functional equivalence by state vector evaluation on all 8 basis states
  auto orig_seq = CircuitSeq::from_qasm_file(&union_ctx, input_fn);
  auto decomp_seq = CircuitSeq::from_qasm_style_string(
      &dst_ctx, graph_decomposed->to_qasm());
  assert(orig_seq != nullptr);
  assert(decomp_seq != nullptr);

  ComplexType global_phase = 0;
  bool phase_initialized = false;
  for (int basis = 0; basis < (1 << 3); ++basis) {
    Vector in_vec(8);
    for (int i = 0; i < 8; ++i) {
      in_vec[i] = (i == basis ? 1.0 : 0.0);
    }
    Vector out_orig, out_decomp;
    orig_seq->evaluate(in_vec, {}, out_orig);
    decomp_seq->evaluate(in_vec, {}, out_decomp);

    ComplexType dot_prod = out_orig.dot(out_decomp);
    double fidelity = std::abs(dot_prod);
    assert(std::abs(fidelity - 1.0) < 1e-5);

    if (!phase_initialized) {
      global_phase = dot_prod;
      phase_initialized = true;
    } else {
      assert(std::abs(dot_prod - global_phase) < 1e-5);
    }
  }
  std::cout << "Unitary equivalence verified on all 8 basis states!" << std::endl;

  std::cout << "Test for Issue #230 passed successfully!" << std::endl;
  return 0;
}
