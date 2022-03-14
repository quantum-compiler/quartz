#include "quartz/context/context.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

int main() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::cx, GateType::t, GateType::tdg});

  // Construct circuit graph from qasm file
  QASMParser qasm_parser(&ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm("circuit/example-circuits/barenco_tof_3.qasm",
                             dag)) {
    std::cout << "Parser failed" << std::endl;
    return 0;
  }
  ctx.get_and_gen_input_dis(dag->get_num_qubits());
  ctx.get_and_gen_hashing_dis(dag->get_num_qubits());
  ctx.get_and_gen_parameters(dag->get_num_input_parameters());
  Graph graph(&ctx, dag);

  std::vector<Op> ops_0;
  graph.topology_order_ops(ops_0);
  std::cout << ops_0.size() << std::endl;

  //   EquivalenceSet eqs;
  //   // Load equivalent dags from file
  //   if (!eqs.load_json(&ctx, "bfs_verified_simplified.json")) {
  //     std::cout << "Failed to load equivalence file." << std::endl;
  //     assert(false);
  //   }

  //   // Get xfer from the equivalent set
  //   auto ecc = eqs.get_all_equivalence_sets();
  //   auto num_equivalent_classes = eqs.num_equivalence_classes();
  //   std::cout << num_equivalent_classes << std::endl;
  //   std::vector<GraphXfer *> xfers;
  //   for (auto eqcs : ecc) {
  //     bool first = true;
  //     // std::cout << eqcs.size() << std::endl;
  //     for (auto circ : eqcs) {
  //       if (first)
  //         first = false;
  //       else {
  //         auto xfer_0 = GraphXfer::create_GraphXfer(&ctx, eqcs[0], circ);
  //         auto xfer_1 = GraphXfer::create_GraphXfer(&ctx, circ, eqcs[0]);
  //         if (xfer_0 != nullptr)
  //           xfers.push_back(xfer_0);
  //         if (xfer_1 != nullptr)
  //           xfers.push_back(xfer_1);
  //       }
  //     }
  //   }

  //   std::vector<Op> ops;
  //   graph.all_ops(ops);
  //   int num_xfers = xfers.size();
  //   std::cout << "number of xfers: " << num_xfers << std::endl;
  //   int num_ops = ops.size();
  //   std::cout << "number of ops: " << num_ops << std::endl;
  //   std::vector<Graph *> graphs;

  //   for (int i = 0; i < num_ops; ++i) {
  //     for (int j = 0; j < num_xfers; ++j) {
  //       if (graph.xfer_appliable(xfers[j], ops[i])) {
  //         std::cout << "Transfer " << j << " appliable to op " << i <<
  //         std::endl; graphs.push_back(graph.apply_xfer(xfers[j], ops[i]));
  //       }
  //     }
  //   }

  //   for (auto g : graphs) {
  //     std::vector<Op> g_ops;
  //     g->all_ops(g_ops);
  //   }
  // for (auto it = ops.begin(); it != ops.end(); ++it) {
  // 	std::cout << gate_type_name(it->ptr->tp) << std::endl;
  // }
  // for (auto it = ops.begin(); it != ops.end(); ++it) {
  // 	bool xfer_ok = graph.xfer_appliable(xfer, &(*it));
  // 	std::cout << (int)xfer_ok << std::endl;
  // }
}