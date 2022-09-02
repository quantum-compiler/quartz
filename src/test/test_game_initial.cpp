#include "quartz/game/game_initial.h"
#include "quartz/env/simple_initial_env.h"

#include <string>
#include <cstdlib>

using namespace std;
using namespace quartz;

int main() {
    // initialize the environment
    string circuit_file_name = "../circuit/nam-circuits/qasm_files/barenco_tof_10_before.qasm";
    SimpleInitialEnv env = SimpleInitialEnv(circuit_file_name, BackendType::IBM_Q20_TOKYO);
    int step_count = 0;
    double total_reward = 0;
    double max_reward = -1000;

    while (step_count < 100) {
        // get state and action space
        auto cur_state = env.get_state();
        auto action_space = env.get_action_space();

        // apply action
        int selected_action_id = rand() % action_space.size();
        auto selected_action = action_space[selected_action_id];
        double reward = env.step(selected_action);

        // check finished
        step_count += 1;
        total_reward += reward;
        max_reward = max_reward > reward ? max_reward : reward;

        // log
        if (step_count % 100 == 0) {
            env.cur_game_ptr->save_execution_history_to_file("./test.eh", "./test.qasm", false);
            cout << "Total reward: " << total_reward << endl;
            cout << "Step count: " << step_count << endl;
            cout << "Avg reward: " << total_reward / step_count << endl;
            cout << "Max reward: " << max_reward << endl;
        }
    }
}



