#include "quartz/device/device.h"
#include "quartz/tasograph/tasograph.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
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
    string circuit_file_name = "../t_cx_tdg.qasm";
    cout << "This is test for init_physical_mapping on " << circuit_file_name <<".\n";

    // prepare context
    Context src_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
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

    // print all Ops
    cout << "Out Edges" << endl;
    for (const auto& Op_edge : graph.outEdges) {
        cout << "Gate: " << Op_edge.first.guid << " has type " << Op_edge.first.ptr->tp << endl;
    }
    cout << "In Edges" << endl;
    for (const auto& Op_edge : graph.inEdges) {
        cout << "Gate: " << Op_edge.first.guid << " has type " << Op_edge.first.ptr->tp << endl;
    }

    // test init qubit mapping
    graph.init_physical_mapping();
    cout << "Mapping has been initialized." << endl;
    for (const auto& input_qubit_mapping : graph.qubit_mapping_table) {
        std::cout << "Gate: " << input_qubit_mapping.first.guid << std::endl;
        std::cout << "Logical idx: " << input_qubit_mapping.second.first << std::endl;
        std::cout << "Physical idx: " << input_qubit_mapping.second.second << std::endl;
    }
    std::cout << "Test has passed\n";
};
