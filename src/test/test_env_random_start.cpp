#include <string>
#include <cstdlib>
#include <cmath>
#include "env/simple_physical_env.h"

using namespace std;
using namespace quartz;

int main() {
    // initialize the environment
    string circuit_file_name = "../circuit/nam-circuits/qasm_files/gf2^E5_mult_after_heavy.qasm";
    SimplePhysicalEnv env = SimplePhysicalEnv(circuit_file_name, BackendType::IBM_Q27_FALCON,
                                              0, 0.8);
    int step_count = 0;
    double total_reward = 0;
    double max_reward = -1000;
    bool is_finished = false;

    for (int i = 0; i < 10; ++i) {
        // get state and action space
        auto cur_state = env.get_state();
        auto action_space = env.get_action_space();

        // apply action
        int selected_action_id = rand() % action_space.size();
        auto selected_action = action_space[selected_action_id];
        double reward = env.step(selected_action);

        // check finished
        is_finished = env.is_finished();
        step_count += 1;
        total_reward += reward;
        max_reward = max_reward > reward ? max_reward : reward;

        // log
//        if (is_finished) {
//            int total_cost = env.total_cost();
//            env.cur_game_ptr->save_execution_history_to_file("./test.eh", "./test.qasm", false);
//            cout << "Total reward: " << total_reward << endl;
//            cout << "Step count: " << step_count << endl;
//            cout << "Avg reward: " << total_reward / step_count << endl;
//            cout << "Max reward: " << max_reward << endl;
//            cout << "Final cost to implement circuit: " << total_cost << endl;
//        }
    }

    // test reset
    for (int i = 0; i < 100; ++i) {
        env.reset();
    }
}




