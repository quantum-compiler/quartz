#include "quartz/context/context.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

bool graph_cmp(std::shared_ptr<Graph> a, std::shared_ptr<Graph> b) {
  return a->gate_count() < b->gate_count();
}

int main() {
//   Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
//                GateType::cx, GateType::x, GateType::rz, GateType::add});
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::cx, GateType::x, GateType::rz, GateType::add});

  // Construct circuit graph from qasm file
//   QASMParser qasm_parser(&ctx);
//   DAG *dag = nullptr;
//   if (!qasm_parser.load_qasm(
//           "../experiment/t_tdg_h_cx_toffoli_flip_dataset/barenco_tof_3.qasm.toffoli_flip", dag)) {
//     std::cout << "Parser failed" << std::endl;
    
    // return 0;
//   }
    auto graph = Graph::from_qasm_file(&ctx, "../experiment/t_tdg_h_cx_toffoli_flip_dataset/mod5_4.qasm.toffoli_flip");

//   ctx.get_and_gen_input_dis(dag->get_num_qubits());
//   ctx.get_and_gen_hashing_dis(dag->get_num_qubits());
//   ctx.get_and_gen_parameters(dag->get_num_input_parameters());

//   std::shared_ptr<Graph> graph(new Graph(&ctx, dag));

  EquivalenceSet eqs;
  // Load equivalent dags from file
  if (!eqs.load_json(&ctx, "../Nam_complete_ECC_set.json")) {
    std::cout << "Failed to load equivalence file." << std::endl;
    assert(false);
  }

  // Get xfer from the equivalent set
  auto ecc = eqs.get_all_equivalence_sets();
  std::vector<GraphXfer *> xfers;
  for (auto eqcs : ecc) {
    for (auto circ_0 : eqcs) {
        for(auto circ_1 : eqcs){
            if(circ_0 != circ_1){
                auto xfer = GraphXfer::create_GraphXfer(&ctx, circ_0, circ_1, false);
                if (xfer != nullptr){
                    xfers.push_back(xfer);
                }
            }
        }
    }
  }
  std::cout << "number of xfers: " << xfers.size() << std::endl;

  //   back tracking search
  auto start = std::chrono::steady_clock::now();
  int budget = 1000000;
  std::priority_queue<
      std::shared_ptr<Graph>, std::vector<std::shared_ptr<Graph>>,
      std::function<bool(std::shared_ptr<Graph>, std::shared_ptr<Graph>)>>
      candidate_q(graph_cmp);
  std::set<size_t> hash_mp;
  candidate_q.push(graph);
  std::shared_ptr<Graph> best_graph = graph;
  hash_mp.insert(graph->hash());
  int best_gate_cnt = graph->gate_count();
  while (!candidate_q.empty() && budget >= 0) {
    auto top_graph = candidate_q.top();
    candidate_q.pop();
    std::vector<Op> all_ops;
    top_graph->topology_order_ops(all_ops);
    assert(all_ops.size() == (size_t)top_graph->gate_count());
    for (auto op : all_ops) {
      for (size_t i = 0; i < xfers.size(); ++i) {
        auto xfer = xfers[i];
        //   for (auto xfer : xfers) {
        if (top_graph->xfer_appliable(xfer, op)) {
          auto new_graph = top_graph->apply_xfer(xfer, op);
          if (hash_mp.find(new_graph->hash()) == hash_mp.end()) {
            candidate_q.push(new_graph);
            hash_mp.insert(new_graph->hash());
            if (new_graph->gate_count() < best_gate_cnt) {
              best_graph = new_graph;
              best_gate_cnt = new_graph->gate_count();
            }
            budget--;
            auto end = std::chrono::steady_clock::now();
            if (budget % 1000 == 0) {
              std::cout << "budget: " << budget << " best gate count "
                        << best_gate_cnt << " in "
                        << (double)std::chrono::duration_cast<
                               std::chrono::milliseconds>(end - start)
                                   .count() /
                               1000.0
                        << " seconds." << std::endl;
            }
          }
        }
      }
    }
  }

  best_graph->to_qasm("test.qasm", false, false);
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