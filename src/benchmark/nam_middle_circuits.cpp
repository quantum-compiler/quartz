#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "test/gen_ecc_set.h"

#include <cmath>
#include <filesystem>
#include <sstream>

using namespace quartz;

static const std::string ecc_set_prefix = "Nam_6_3_";
static const std::string ecc_set_name =
    ecc_set_prefix + "complete_ECC_set.json";

static int num_benchmark = 0;
static double geomean_gate_count = 1;
static std::stringstream summary_result;

void benchmark_nam(const std::string &circuit_name) {
  std::string circuit_path = "circuit/nam-benchmarks/" + circuit_name + ".qasm";
  // Construct contexts
  ParamInfo param_info;
  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::add, GateType::input_qubit, GateType::input_param},
                  &param_info);
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param},
                  &param_info);
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&src_ctx, &dst_ctx, &union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(circuit_path, dag)) {
    std::cout << "Parser failed" << std::endl;
    return;
  }

  Graph graph(&src_ctx, dag);

  auto start = std::chrono::steady_clock::now();
  // Greedy toffoli flip
  auto graph_before_search = graph.toffoli_flip_greedy(
      GateType::rz, xfer_pair.first, xfer_pair.second);
  //   graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);

  auto end = std::chrono::steady_clock::now();

  num_benchmark++;
  geomean_gate_count *= graph_before_search->gate_count();

  std::cout << circuit_name << " after toffoli flip in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds: gate count = " << graph_before_search->gate_count()
            << ", circuit depth = " << graph_before_search->circuit_depth()
            << ", cost = " << graph_before_search->total_cost() << std::endl;

  start = std::chrono::steady_clock::now();
  // Optimization
  auto graph_after_search = graph_before_search->optimize(
      &dst_ctx, ecc_set_name, circuit_name, /*print_message=*/
      true,                                 /*cost_function=*/
      nullptr,                              /*cost_upper_bound=*/
      -1,                                   /*timeout=*/
      10);
  end = std::chrono::steady_clock::now();

  std::cout << circuit_name << " optimization result in "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds: gate count = " << graph_after_search->gate_count()
            << ", circuit depth = " << graph_after_search->circuit_depth()
            << ", cost = " << graph_after_search->total_cost() << std::endl;

  summary_result
      << circuit_name << " optimization result in "
      << (double)std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                 .count() /
             1000.0
      << " seconds: gate count = " << graph_after_search->gate_count()
      << ", circuit depth = " << graph_after_search->circuit_depth()
      << ", cost = " << graph_after_search->total_cost() << std::endl;

  geomean_gate_count /= graph_after_search->gate_count();
  // graph_after_search->to_qasm(output_fn, false, false);
}

int main() {
  if (!std::filesystem::exists(ecc_set_name)) {
    std::cout << "Generating ECC set..." << std::endl;
    gen_ecc_set(
        {GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add},
        ecc_set_prefix, true, false, 3, 2, 6);
    std::cout << "ECC set generated." << std::endl;
  }
  // Logs are printed to Nam_6_3_barenco_tof_10.log, Nam_6_3_gf2^8_mult.log, ...
  benchmark_nam("barenco_tof_10");  // 450 gates
  benchmark_nam("gf2^8_mult");      // 883 gates
  benchmark_nam("qcla_mod_7");      // 884 gates
  benchmark_nam("adder_8");         // 900 gates
  if (num_benchmark > 0) {
    std::cout << "Summary:" << std::endl;
    std::cout << summary_result.str();
    std::cout << num_benchmark << " circuits, gate count optimized by "
              << std::pow(geomean_gate_count, 1.0 / num_benchmark)
              << " times on average." << std::endl;
  }
  return 0;
}
