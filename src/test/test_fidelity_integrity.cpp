#include "fidelity/fidelity.h"
#include "fidelity/known_fidelity_graphs.h"
#include "tasograph/tasograph.h"
#include "sabre/sabre_swap.h"
#include <fstream>
#include <string>


using namespace std;
using namespace quartz;

int main() {
    // build execution history
    std::ifstream infile("../rl_eh.txt");
    int guid, phy0, phy1, log0, log1;
    string tp;
    auto execution_history = vector<ExecutionHistory>();
    while (infile >> guid >> tp >> phy0 >> phy1 >> log0 >> log1) {
//        cout << guid << tp << phy0 << phy1 << log0 << log1 << endl;
        assert(tp == "cx" || tp == "swap");
        auto gate_type = tp == "cx" ? GateType::cx : GateType::swap;
        execution_history.emplace_back(guid, gate_type, log0, log1, phy0, phy1);
    }

    // calculate fidelity
    auto fidelity_graph = GetFidelityGraph(BackendType::IBM_Q27_FALCON);
    std::vector<bool> is_qubit_used = std::vector<bool>(27, false);
    double sum_ln_cx_fidelity = 0;
    for (const ExecutionHistory &eh_item: execution_history) {
        if (eh_item.gate_type == GateType::swap) {
            // only swaps with at least one logical input used have non-zero cost
            int _logical0 = eh_item.logical0;
            int _logical1 = eh_item.logical1;
            if (is_qubit_used[_logical0] || is_qubit_used[_logical1]) {
                // the swap has cost, it also marks the input qubits as used.
                // Note: we think swap2 below has cost.
                // 1          swap2
                // 2    swap1 swap2
                // 3 cx swap1
                // 4 cx
                sum_ln_cx_fidelity += 3 * fidelity_graph->query_cx_fidelity(eh_item.physical0,
                                                                            eh_item.physical1);
                is_qubit_used[_logical0] = true;
                is_qubit_used[_logical1] = true;
            }

            // check if virtual swaps indeed have cost 0
            if (eh_item.guid == -2) {
                Assert(!is_qubit_used[_logical0] && !is_qubit_used[_logical1],
                       "Virtual Swaps in phase one must have cost 0!");
            }
        } else {
            // set target logical qubit as used
            int _logical0 = eh_item.logical0;
            int _logical1 = eh_item.logical1;
            is_qubit_used[_logical0] = true;
            is_qubit_used[_logical1] = true;

            // add fidelity of this gate (it must be a CNOT)
            Assert(eh_item.gate_type==GateType::cx, "Fidelity calculation only supports cx gates!");
            sum_ln_cx_fidelity += fidelity_graph->query_cx_fidelity(eh_item.physical0, eh_item.physical1);
        }
    }
    cout << sum_ln_cx_fidelity << endl;
}
