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
    string circuit_file_name = "../sabre.qasm";
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
    auto device = std::make_shared<quartz::SymmetricUniformDevice>(3);
    // first row
    device->add_edge(0, 2);
    device->add_edge(2, 1);

    // print gate count
    int total_gate_count = graph.gate_count();
    cout << "Gate count: " << total_gate_count << endl;

    // perform a trivial mapping and print cost
    graph.init_physical_mapping(InitialMappingType::TRIVIAL, nullptr,
                                -1, false, -1);
    MappingStatus succeeded = graph.check_mapping_correctness();
    if (succeeded == quartz::MappingStatus::VALID) {
        std::cout << "Trivial Mapping has passed correctness check." << endl;
    } else {
        std::cout << "Mapping test failed!" << endl;
    }
    double total_cost = graph.circuit_implementation_cost(device);
    cout << "Trivial implementation cost is " << total_cost << endl;

    // simplify circuit
    Game new_game(graph, device);
    Reward reward = new_game.apply_action(Action(ActionType::PhysicalFront, 0, 2));
    cout << "Reward is " << reward << ", game finished " << is_circuit_finished(new_game.graph) << endl;
};

