#include "quartz/device/device.h"
#include "quartz/tasograph/tasograph.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/game/game.h"
#include <iostream>

using namespace quartz;
using namespace std;

int main() {
    // tof_3_after_heavy.qasm / t_cx_tdg.qasm
    string circuit_file_name = "../tof_3_after_heavy.qasm";
    cout << "This is test for add swap on " << circuit_file_name << ".\n";

    // prepare context
    Context src_ctx({GateType::h, GateType::x, GateType::rz, GateType::add, GateType::swap,
                     GateType::cx, GateType::input_qubit, GateType::input_param, GateType::t, GateType::tdg,
                     GateType::s, GateType::sdg});
    // parse qasm file
    QASMParser qasm_parser(&src_ctx);
    DAG *dag = nullptr;
    if (!qasm_parser.load_qasm(circuit_file_name, dag)) {
        cout << "Parser failed" << endl;
        return -1;
    }
    Graph graph(&src_ctx, dag);
    cout << "Circuit initialized\n";

    // initialize device
    auto device = std::make_shared<quartz::SymmetricUniformDevice>(10);
    // first row
    device->add_edge(0, 1);
    device->add_edge(1, 2);
    device->add_edge(2, 3);
    device->add_edge(3, 4);
    // second row
    device->add_edge(5, 6);
    device->add_edge(6, 7);
    device->add_edge(7, 8);
    device->add_edge(8, 9);
    // col
    device->add_edge(0, 5);
    device->add_edge(1, 6);
    device->add_edge(2, 7);
    device->add_edge(3, 8);
    device->add_edge(4, 9);
    // crossings
    device->add_edge(1, 7);
    device->add_edge(2, 6);
    device->add_edge(3, 9);
    device->add_edge(4, 8);

    // print gate count
    int total_gate_count = graph.gate_count();
    cout << "Gate count: " << total_gate_count << endl;

    // perform a trivial mapping and print cost
    auto trivial_graph = graph;
    trivial_graph.init_physical_mapping(InitialMappingType::TRIVIAL, nullptr,
                                        -1, false, -1);
    MappingStatus succeeded = trivial_graph.check_mapping_correctness();
    if (succeeded == quartz::MappingStatus::VALID) {
        std::cout << "Trivial Mapping has passed correctness check." << endl;
    } else {
        std::cout << "Mapping test failed!" << endl;
    }
    double total_cost = trivial_graph.circuit_implementation_cost(device);
    cout << "Trivial implementation cost is " << total_cost << endl;

    // init sabre mapping and print cost
    // hyper-parameter search
    std::vector<int> iter_list{3};
    std::vector<double> W_list{0.5};
    std::vector<bool> use_extensive_list{true};
    double min_sabre_cost = 100000;
    Graph best_graph = graph;
    for (const auto &iter_cnt: iter_list) {
        for (const auto &w_value: W_list) {
            for (const auto &use_extensive: use_extensive_list) {
                for (int repeat = 0; repeat < 10; ++repeat) {
                    auto tmp_graph = graph;
                    tmp_graph.init_physical_mapping(InitialMappingType::SABRE, device,
                                                    iter_cnt, use_extensive, w_value);
                    MappingStatus succeeded_tmp = tmp_graph.check_mapping_correctness();
                    if (succeeded_tmp != quartz::MappingStatus::VALID) {
                        std::cout << "Mapping test failed!" << endl;
                    }
                    double sabre_cost = tmp_graph.circuit_implementation_cost(device);
                    if (sabre_cost < min_sabre_cost) {
                        min_sabre_cost = sabre_cost;
                        best_graph = tmp_graph;
                    }
                }
            }
        }
    }
    cout << "Sabre search implementation cost is " << min_sabre_cost << endl;

    // create game
    Game new_game(best_graph, device);
    auto action_space_physical_full = new_game.action_space(ActionType::PhysicalFull);
    auto action_space_physical_front = new_game.action_space(ActionType::PhysicalFront);
    auto action_space_logical = new_game.action_space(ActionType::Logical);

    // copy game
    Game copied_game = new_game;
    copied_game.graph.inEdges = std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>();
    copied_game.execution_history[0].logical1=-11000;
};



