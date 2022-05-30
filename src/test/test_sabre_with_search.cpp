#include "quartz/device/device.h"
#include "quartz/tasograph/tasograph.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/sabre/sabre.h"
#include "quartz/sabre/sabre_swap.h"
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
    string circuit_file_name = "../sabre.qasm";
    cout << "This is TEST_SABRE_WITH_SEARCH on " << circuit_file_name <<".\n";

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

    // initialize device IBM Q20 Tokyo
    auto device = std::make_shared<quartz::SymmetricUniformDevice>(20);
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
    // third row
    device->add_edge(10, 11);
    device->add_edge(11, 12);
    device->add_edge(12, 13);
    device->add_edge(13, 14);
    // fourth row
    device->add_edge(15, 16);
    device->add_edge(16, 17);
    device->add_edge(17, 18);
    device->add_edge(18, 19);
    // first col
    device->add_edge(0, 5);
    device->add_edge(5, 10);
    device->add_edge(10, 15);
    // second col
    device->add_edge(1, 6);
    device->add_edge(6, 11);
    device->add_edge(11, 16);
    // third col
    device->add_edge(2, 7);
    device->add_edge(7, 12);
    device->add_edge(12, 17);
    // fourth col
    device->add_edge(3, 8);
    device->add_edge(8, 13);
    device->add_edge(13, 18);
    // fifth col
    device->add_edge(4, 9);
    device->add_edge(9, 14);
    device->add_edge(14, 19);
    // crossing in row 1
    device->add_edge(1, 7);
    device->add_edge(2, 6);
    device->add_edge(3, 9);
    device->add_edge(4, 8);
    // crossing in row 2
    device->add_edge(5, 11);
    device->add_edge(6, 10);
    device->add_edge(7, 13);
    device->add_edge(8, 12);
    // crossing in row 3
    device->add_edge(11, 17);
    device->add_edge(12, 16);
    device->add_edge(13, 19);
    device->add_edge(14, 18);

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
    std::vector<int> iter_list{1, 2, 3, 5};
    std::vector<double> W_list{0.0, 0.01, 0.1, 0.2, 0.5, 1.0};
    std::vector<bool> use_extensive_list{true, false};
    double min_sabre_cost = 100000;
    Graph best_graph = graph;
    for (const auto& iter_cnt : iter_list) {
        for (const auto& w_value : W_list) {
            for (const auto& use_extensive : use_extensive_list) {
                for (int repeat = 0; repeat < 5; ++repeat) {
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

    // sabre swap
    auto execution_history = sabre_swap(best_graph, device, true, 0.5);
    auto eh_status = check_execution_history(best_graph, device, execution_history);
    if (eh_status != ExecutionHistoryStatus::VALID) {
        cout << "Invalid execution history!" << endl;
    }
    int real_cost = execution_cost(execution_history);
    cout << "Sabre swap real cost is " << real_cost << endl;
};

