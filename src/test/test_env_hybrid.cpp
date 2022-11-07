#include <string>
#include "env/simple_physical_env.h"
#include "env/simple_hybrid_env.h"

using namespace std;
using namespace quartz;

int main() {
    // initialize the environment
    //  string circuit_file_name = "../circuit/nam-circuits/qasm_files/gf2^E5_mult_after_heavy.qasm";
    string circuit_file_name = "../tof_3_after_heavy.qasm";
    SimpleHybridEnv env = SimpleHybridEnv(
            // basic settings
            circuit_file_name, BackendType::IBM_Q27_FALCON, "../test_mapping_file.txt",
            // randomness
            0, 0.8,
            // GameHybrid
            5, true, -0.3);

    // apply one step on the environment
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
}
