#include "test/gen_ecc_set.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include <cmath>
#include <filesystem>

using namespace quartz;

static const std::string ecc_set_prefix = "Nam_4_3_";
static const std::string
    ecc_set_name = ecc_set_prefix + "complete_ECC_set.json";

static int num_benchmark = 0;
static double geomean_gate_count = 1;

void benchmark_nam(const std::string &circuit_name) {
  std::string circuit_path = "circuit/nam-benchmarks/" + circuit_name + ".qasm";
  // Construct contexts
  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::input_qubit, GateType::input_param});
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param});
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(circuit_path, dag)) {
    std::cout << "Parser failed" << std::endl;
    return;
  }

  Graph graph(&src_ctx, dag);

  num_benchmark++;
  geomean_gate_count *= graph.gate_count();

  auto start = std::chrono::steady_clock::now();
  // Greedy toffoli flip
  auto graph_before_search = graph.toffoli_flip_greedy(
      GateType::rz, xfer_pair.first, xfer_pair.second);
  //   graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);

  auto end = std::chrono::steady_clock::now();

  std::cout << circuit_name << " after toffoli flip in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start)
                .count() /
                1000.0
            << " seconds: gate count = " << graph_before_search->gate_count()
            << ", cost = " << graph_before_search->total_cost() << std::endl;

  start = std::chrono::steady_clock::now();
  // Optimization
  auto graph_after_search = graph_before_search->optimize(
      0.999,
      0,
      false,
      &dst_ctx,
      ecc_set_name, /*use_simulated_annealing=*/
      false, /*enable_early_stop=*/
      false,
      /*use_rotation_merging_in_searching=*/
      false,
      GateType::rz,
      circuit_name, /*timeout=*/
      10);
  end = std::chrono::steady_clock::now();

  std::cout << circuit_name << " optimization result in "
            << (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start)
                .count() /
                1000.0
            << " seconds: gate count = " << graph_after_search->gate_count()
            << ", cost = " << graph_after_search->total_cost() << std::endl;

  geomean_gate_count /= graph_after_search->gate_count();
  // graph_after_search->to_qasm(output_fn, false, false);
}

int main() {
  if (!std::filesystem::exists(ecc_set_name)) {
    std::cout << "Generating ECC set..." << std::endl;
    gen_ecc_set(
        {GateType::rz, GateType::h, GateType::cx, GateType::x,
         GateType::add}, ecc_set_prefix, true, 3, 2, 4);
    std::cout << "ECC set generated." << std::endl;
  }
  benchmark_nam("tof_3");  // 45 gates
  benchmark_nam("barenco_tof_3");  // 58 gates
  benchmark_nam("mod_mult_55");  // 119 gates
  benchmark_nam("vbe_adder_3");  // 150 gates
  benchmark_nam("gf2^4_mult");  // 225 gates
  if (num_benchmark > 0) {
    std::cout << num_benchmark << " circuits, gate count optimized by "
              << std::pow(geomean_gate_count, 1.0 / num_benchmark) << " times."
              << std::endl;
  }
  return 0;
}
