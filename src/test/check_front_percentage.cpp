/* This file contains sabre 1 pass layout. It can be used as a basic benchmark.
 * Experiment setting: sabre layout x 1 + sabre swap x 1.
 */

#include "quartz/sabre/sabre_swap.h"
#include "quartz/supported_devices/supported_devices.h"
#include "quartz/game/game_utils.h"
#include <filesystem>

namespace fs = std::filesystem;

#include <iostream>

using namespace quartz;
using namespace std;

const int LARGE_INT = 987654321;

void test_front_gate(const string &circuit_file_name) {
    BackendType device_type = BackendType::IBM_Q127_EAGLE;
    int num_experiments = 1;
    int sabre_layout_iterations = 1;
    bool sabre_use_extensive_heuristic = true;
    double sabre_w = 0.5f;

    // log parameter information
    // cout << "Circuit name: " << circuit_file_name << endl;
    // cout << "Device name: " << device_type << endl;
    // cout << "Number of experiments: " << num_experiments << endl;
    // cout << endl;

    // initialize circuit and device
    // circuit
    Context src_ctx({GateType::h, GateType::x, GateType::cx, GateType::input_qubit,
                     GateType::t, GateType::tdg, GateType::s, GateType::sdg, GateType::z});
    QASMParser qasm_parser(&src_ctx);
    DAG *dag = nullptr;
    if (!qasm_parser.load_qasm(circuit_file_name, dag)) {
        cout << "Parser failed" << endl;
        return;
    }
    Graph graph(&src_ctx, dag);
    // device
    shared_ptr<DeviceTopologyGraph> device = GetDevice(device_type);

    // perform sabre layout and sabre swap
    int best_implementation_cost = LARGE_INT;
    int best_execution_cost = LARGE_INT;
    int total_implementation_cost = 0;
    int total_execution_cost = 0;
    for (int experiment_id = 0; experiment_id < num_experiments; ++experiment_id) {
        // sabre layout for initial mapping
        graph.init_physical_mapping(InitialMappingType::SABRE, device, sabre_layout_iterations,
                                    sabre_use_extensive_heuristic, sabre_w);
        assert(graph.check_mapping_correctness() == MappingStatus::VALID);
        int cur_implementation_cost = static_cast<int>(graph.circuit_implementation_cost(device));
        best_implementation_cost = min(best_implementation_cost, cur_implementation_cost);
        total_implementation_cost += best_implementation_cost;

        // sabre swap for qubit routing
        vector<ExecutionHistory> execution_history = sabre_swap(graph, device,
                                                                sabre_use_extensive_heuristic, sabre_w);
        assert(check_execution_history(graph, device, execution_history) == ExecutionHistoryStatus::VALID);
        int cur_execution_cost = execution_cost(execution_history);
        best_execution_cost = min(best_execution_cost, cur_execution_cost);
        total_execution_cost += cur_execution_cost;

    }

    // log statistics
    // cout << "Gate count: " << graph.gate_count() << endl;
    // cout << "Best implementation cost after sabre layout: " << best_implementation_cost << endl;
    // cout << "Best execution cost using Sabre layout & Sabre swap: " << best_execution_cost << endl;
    // cout << "Avg. implementation cost after sabre layout: " << total_implementation_cost / num_experiments << endl;
    // cout << "Avg. execution cost using Sabre layout & Sabre swap: " << total_execution_cost / num_experiments << endl;

    // test topology
    auto num_qubits = graph.get_num_qubits();
    auto gate_count = graph.gate_count();
    auto single_qubit_gate_count = simplify_circuit(graph);
    auto cnot_count = graph.gate_count();
    auto front_set_size_7 =
            graph.get_front_layers(7, true).size() - num_qubits;  // This is because front set contains input qubits
    auto front_set_size_6 =
            graph.get_front_layers(6, true).size() - num_qubits;  // This is because front set contains input qubits
    assert(single_qubit_gate_count + cnot_count == gate_count);
    // name -- #qubits -- #gates -- #CNOTs -- #FrontSet Gates -- Percentage
    cout << circuit_file_name << " " << num_qubits << " "
         << gate_count << " " << cnot_count << " "
         << front_set_size_7 << " " << double(front_set_size_7) / cnot_count << " "
         << front_set_size_6 << " " << double(front_set_size_6) / cnot_count << endl;
}

int main() {
    // gather data for each circuit
    string path = "../circuit/nam-circuits/qasm_files/";
    for (const auto &entry: fs::directory_iterator(path)) {
        if (entry.path().string().find("E64") != string::npos ||
            entry.path().string().find("E128") != string::npos ||
            entry.path().string().find("E131") != string::npos ||
            entry.path().string().find("E163") != string::npos ||
            entry.path().string().find("csum_mux_9_before_corrected.qasm") != string::npos) continue;
        test_front_gate(entry.path().string());
    }
}
