#include <string>
#include "env/simple_physical_env.h"
#include "env/simple_hybrid_env.h"

using namespace std;
using namespace quartz;

int main() {
    // initialize the environment
    //  string circuit_file_name = "../circuit/nam-circuits/qasm_files/gf2^E5_mult_after_heavy.qasm";
    string circuit_file_name = "../mbq_test.qasm";
    SimpleHybridEnv env = SimpleHybridEnv(
            // basic settings
            circuit_file_name, BackendType::IBM_Q27_FALCON, "../test_mapping_file.txt",
            // GameBuffer
            0, 0.8, 5, 3,
            // GameHybrid
            5, true, -0.3);

    // apply fixed check
    // NOTE: This check is designed such that it can only pass when we disable all sanity checks!
    env.cur_game_ptr->graph.inEdges.clear();
    env.cur_game_ptr->graph.outEdges.clear();
    env.cur_game_ptr->graph.qubit_mapping_table.clear();
    env.cur_game_ptr->execution_history.emplace_back(1, GateType::swap,
                                                     0, 1, 0, 1, "");
    env.cur_game_ptr->execution_history.emplace_back(1, GateType::cx,
                                                     2, 0, 2, 1, "");
    env.cur_game_ptr->execution_history.emplace_back(1, GateType::swap,
                                                     2, 3, 2, 3, "");
    env.cur_game_ptr->execution_history.emplace_back(1, GateType::swap,
                                                     4, 5, 4, 5, "");
    double ref_value = env.cur_game_ptr->fidelity_graph->query_cx_fidelity(2, 1) +
                       3 * env.cur_game_ptr->fidelity_graph->query_cx_fidelity(2, 3);
    cout << "Sum fidelity = " << env.sum_ln_cx_fidelity() << " , ref value is " << ref_value << endl;

    // reset the environment to start a new game
    env.reset();
    cout << "Reset!" << endl;
    while (!env.is_finished()) {
        State state_before = env.get_state();
        vector<Action> action_space = env.get_action_space();
        int selected_action_id = rand() % action_space.size();
        Action selected_action = action_space[selected_action_id];
        Reward reward = env.step(selected_action);
        State state_after = env.get_state();
        bool is_finished = env.is_finished();
        cout << "Reward is " << reward << ", is finished = " << is_finished << endl;
    }
    cout << "Sum fidelity = " << env.sum_ln_cx_fidelity() << endl;
}

