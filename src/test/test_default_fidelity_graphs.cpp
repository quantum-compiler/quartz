#include "fidelity/known_fidelity_graphs.h"

using namespace std;
using namespace quartz;

int main() {
    // the fidelity graphs
    auto device_list = {BackendType::IBM_Q27_FALCON, BackendType::IBM_Q65_HUMMINGBIRD, BackendType::IBM_Q127_EAGLE};
    for (const auto &device_type: device_list) {
        // prepare
        auto device_graph = GetDevice(device_type);
        auto fidelity_graph = GetFidelityGraph(device_type);
        double sum_ln_fidelity = 0;

        // loop through each edge
        for (int reg0 = 0; reg0 < device_graph->get_num_qubits(); ++reg0) {
            auto neighbors = device_graph->get_input_neighbours(reg0);
            for (const auto &reg1: neighbors) {
                sum_ln_fidelity += fidelity_graph->query_cx_fidelity(reg0, reg1);
            }
        }
        cout << "Device: " << device_type << " , sum ln fidelity (two-way) is " << sum_ln_fidelity
             << ", fidelity is " << exp(sum_ln_fidelity) << endl;
    }
}
