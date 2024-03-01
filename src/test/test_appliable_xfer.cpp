//
// Created by Colin on 2022/5/25.
//
#include "quartz/context/context.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace quartz;

template <class T> std::string output_vec(const std::vector<T> &vec) {
  std::ostringstream osstream;
  for (const auto &item : vec) {
    osstream << item << ", ";
  }
  return osstream.str();
}

int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::cx, GateType::t, GateType::tdg},
              &param_info);
  EquivalenceSet eqs;
  if (!eqs.load_json(&ctx, "../bfs_verified_simplified.json",
                     /*from_verifier=*/false)) {
    std::cerr << "Failed to load equivalence file." << std::endl;
    return 1;
  }
  // build all-to-all xfers
  auto ecc = eqs.get_all_equivalence_sets();
  std::vector<GraphXfer *> xfers;
  for (auto eqcs : ecc) {
    for (auto circ_1 : eqcs) {
      for (auto circ_2 : eqcs) {
        if (circ_1 != circ_2) {
          auto xfer = GraphXfer::create_GraphXfer(&ctx, circ_1, circ_2, true);
          if (xfer)
            xfers.emplace_back(xfer);
        }
      }
    }
  }
  std::cout << "number of xfers: " << xfers.size() << std::endl;

  QASMParser qasm_parser(&ctx);
  CircuitSeq *dag = nullptr;
  const std::string qasm_str =
      "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[5];\nh q[4];\ncx "
      "q[3],q[4];\ntdg q[4];\ncx q[2],q[4];\nt q[4];\ncx q[3],q[4];\ntdg "
      "q[4];\ncx q[2],q[4];\ncx q[2],q[3];\nt q[4];\ntdg q[3];\ncx "
      "q[2],q[3];\nt q[2];\nt q[3];\nh q[3];\ncx q[1],q[3];\ntdg q[3];\ncx "
      "q[0],q[3];\nt q[3];\ncx q[1],q[3];\ntdg q[3];\ncx q[0],q[3];\ncx "
      "q[0],q[1];\nt q[3];\ntdg q[1];\nh q[3];\ncx q[0],q[1];\ncx "
      "q[3],q[4];\nt q[0];\nt q[1];\nt q[4];\ncx q[2],q[4];\ntdg q[4];\ncx "
      "q[3],q[4];\nt q[4];\ncx q[2],q[4];\ncx q[2],q[3];\ntdg q[4];\nt "
      "q[3];\nh q[4];\ncx q[2],q[3];\ntdg q[2];\ntdg q[3];\nh q[3];\ncx "
      "q[1],q[3];\nt q[3];\ncx q[0],q[3];\ntdg q[3];\ncx q[1],q[3];\nt "
      "q[3];\ncx q[0],q[3];\ncx q[0],q[1];\ntdg q[3];\nt q[1];\nh q[3];\ncx "
      "q[0],q[1];\ntdg q[0];\ntdg q[1];\n";
  if (!qasm_parser.load_qasm_str(qasm_str, dag)) {
    std::cerr << "Parser failed" << std::endl;
    return 1;
  }
  Graph graph(&ctx, dag);
  std::vector<Op> all_ops;
  graph.topology_order_ops(all_ops);

  const int idx = 7;

  for (int _t = 0; _t < 1e5; _t++) {
    const auto t1_start = std::chrono::high_resolution_clock::now();
    const auto ans_ref = graph.appliable_xfers(all_ops[idx], xfers);
    const auto t1_end = std::chrono::high_resolution_clock::now();

    const auto t2_start = std::chrono::high_resolution_clock::now();
    const auto ans_para = graph.appliable_xfers_parallel(all_ops[idx], xfers);
    const auto t2_end = std::chrono::high_resolution_clock::now();

    const auto dur1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start)
            .count();
    const auto dur2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start)
            .count();
    std::cout << "ans_ref : " << output_vec(ans_ref) << std::endl;
    std::cout << "ans_para: " << output_vec(ans_para) << std::endl;
    std::cout << "dur1 = " << dur1 << "  dur2 = " << dur2 << std::endl
              << std::endl;

    for (size_t i = 0; i < ans_ref.size(); i++) {
      if (i >= ans_para.size()) {
        std::cerr << i << " >= ans_para.size()  "
                  << "ans_ref.size(): " << ans_ref.size()
                  << "  ans_para.size(): " << ans_para.size() << std::endl;
        return 1;
      }
      if (ans_para[i] != ans_ref[i]) {
        std::cerr << "ans_para[i] = " << ans_para[i] << "  "
                  << "ans_ref[i] = " << ans_ref[i] << std::endl;
        return 1;
      }
    }
  }

  return 0;
}
