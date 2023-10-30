#include "quartz/context/context.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace quartz;

// QASMParser qasm_parser(&ctx);

template <class Vec, class VT> void append(Vec &vec, const VT &value) {
  vec.emplace_back(value);
}

// auto graph_from_qasm_file(const std::string& file) {
//   DAG *dag = nullptr;
//   qasm_parser.load_qasm(file, dag);
//   return std::make_shared<Graph>( Graph(&ctx, dag) );
// }

int test() {
  {
    const std::string input_file = "../experiment/circs/nam_circs/adder_8.qasm";

    Context ctx({GateType::h, GateType::cx, GateType::x, GateType::rz,
                 GateType::add, GateType::input_qubit, GateType::input_param});

    // EquivalenceSet eqs;
    // Load equivalent dags from file
    // if (!eqs.load_json(&ctx, "../experiment/ecc_set/nam_ecc.json")) {
    //   std::cout << "Failed to load equivalence file." << std::endl;
    //   assert(false);
    // }
    // // Get xfer from the equivalent set
    // auto ecc = eqs.get_all_equivalence_sets();
    // std::vector<GraphXfer *> xfers;
    // for (auto eqcs : ecc) {
    //   for (auto circ_0 : eqcs) {
    //     for (auto circ_1 : eqcs) {
    //       if (circ_0 != circ_1) {
    //         auto xfer = GraphXfer::create_GraphXfer(&ctx, circ_0, circ_1,
    //         false); if (xfer != nullptr) {
    //           xfers.push_back(xfer);
    //         }
    //       }
    //     }
    //   }
    // }
    // std::cout << "number of xfers: " << xfers.size() << std::endl;
    // std::cout << "Reading graph...  " << std::endl;
    // const auto test_graph = Graph::from_qasm_file(&ctx, input_file);
    // //  const auto test_graph = graph_from_qasm_file(input_file);
    // std::cout << "test_graph.get_num_qubits() = " << test_graph->gate_count()
    // << std::endl;

    // std::vector<Op> all_ops;
    // test_graph->topology_order_ops(all_ops);
    // assert(all_ops.size() == (size_t)test_graph->gate_count());
    // bool stop_loop = false;
    // // (0, 1945)
    // for (size_t i_op = 0; i_op < all_ops.size() && !stop_loop; i_op++) {
    //   auto op = all_ops[i_op];
    // // for (auto op : all_ops) {
    //   for (size_t i_x = 0; i_x < xfers.size(); ++i_x) {
    //     auto xfer = xfers[i_x];
    //     if (test_graph->xfer_appliable(xfer, op)) {
    //       auto new_graph = test_graph->apply_xfer(xfer, op);
    //       std::cout << "available (i_op, i_x): (" << i_op << ", " << i_x <<
    //       ")" << std::endl; stop_loop = true; break;
    //     }
    //   }
    // }

    std::vector<std::shared_ptr<Graph>> buffer;
    // while (true) {
    {
      int delta = 0;
      std::cout << "Input buffer size >  " << std::flush;
      std::cin >> delta;
      std::cout << "Start appending: size(buffer) = " << buffer.size()
                << " delta = " << delta << std::endl;
      for (int i = 0; i < delta; i++) {
        auto tmp_graph = Graph::from_qasm_file(&ctx, input_file);
        append(buffer, tmp_graph);
      }
      std::cout << "Appending finished: size(buffer) = " << buffer.size()
                << "\n\n"
                << std::endl;

      std::cout << "Sleep..." << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(10 * 1000));
      std::cout << "Clear..." << std::endl;
      for (int i = 0; i < buffer.size(); i++) {
        buffer[i].reset();  // unecessary actually; clear can lead to
                            // deconstruction of graphs pointed to
      }
      buffer.clear();
      buffer.shrink_to_fit();
      std::cout << "Clear finished!" << std::endl;
      std::cout << "Sleep..." << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(10 * 1000));
    }
  }
  return 0;
}

int main() {
  while (true) {
    test();
    std::cout << "returned to main" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(5 * 1000));
  }
  return 0;
}
