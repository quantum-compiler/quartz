#include "quartz/device/device.h"
#include "quartz/tasograph/tasograph.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/sabre/sabre.h"
#include <iostream>

using namespace quartz;
using namespace std;

ostream &operator << ( ostream& stream, GateType t )
{
    const string name_list[] = {"h", "x", "y", "rx", "ry", "rz", "cx", "ccx", "add",
                                "neg", "z", "s", "sdg", "t", "tdg", "ch", "swap", "p",
                                "pdg", "rx1", "rx3", "u1", "u2", "u3", "ccz", "cz",
                                "input_qubit", "input_param"
    };
    return stream << name_list[int(t)];
}

int main() {
    // tof_3_after_heavy.qasm / t_cx_tdg.qasm
    string circuit_file_name = "../t_cx_tdg.qasm";
    cout << "This is test for add swap on " << circuit_file_name <<".\n";

    // prepare context
    Context src_ctx({GateType::h, GateType::x, GateType::rz, GateType::add, GateType::swap,
                     GateType::cx, GateType::input_qubit, GateType::input_param, GateType::t, GateType::tdg});
    // parse qasm file
    QASMParser qasm_parser(&src_ctx);
    DAG *dag = nullptr;
    if (!qasm_parser.load_qasm(circuit_file_name, dag)) {
        cout << "Parser failed" << endl;
        return -1;
    }
    Graph graph(&src_ctx, dag);
    cout << "Circuit initialized\n";

    // initialize a device
    auto device = std::make_shared<quartz::SymmetricUniformDevice>(5);
    device->add_edge(0, 2);
    device->add_edge(0, 3);
    device->add_edge(4, 3);
    device->add_edge(4, 1);
    device->add_edge(1, 2);

    // print all Ops
    cout << "Out Edges" << endl;
    for (const auto& Op_edge : graph.outEdges) {
        cout << "Gate: " << Op_edge.first.guid << " has type " << Op_edge.first.ptr->tp << endl;
    }
    cout << "In Edges" << endl;
    for (const auto& Op_edge : graph.inEdges) {
        cout << "Gate: " << Op_edge.first.guid << " has type " << Op_edge.first.ptr->tp << endl;
    }
    cout << endl;

    // test basic sabre heuristic
    std::vector<std::pair<int, int>> front_set;
    front_set.emplace_back(1, 3);
    front_set.emplace_back(2, 3);
    cout << "Heuristic value: " << basic_sabre_heuristic(front_set, device) << endl << endl;

    // init qubit mapping and print cost
    graph.init_physical_mapping(InitialMappingType::SABRE);
    cout << "Mapping has been initialized." << endl;
//    MappingStatus succeeded = graph.check_mapping_correctness();
//    if (succeeded == quartz::MappingStatus::VALID) std::cout << "Mapping has passed correctness check." << endl;
//    else std::cout << "Mapping test failed!\n" << endl;
//    double total_cost = graph.circuit_implementation_cost(device);
//    cout << "Total cost is " << total_cost << endl << endl;

};
