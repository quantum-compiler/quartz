#include "quartz/game/game_search.h"
#include "quartz/env/simple_search_env.h"
#include "quartz/supported_devices/supported_devices.h"

#include <string>
#include <cstdlib>

using namespace std;
using namespace quartz;

int main() {
    // initialize the environment
    string circuit_file_name = "../circuit/nam-circuits/qasm_files/gf2^E5_mult_after_heavy.qasm";
    SimpleSearchEnv env = SimpleSearchEnv(circuit_file_name, BackendType::IBM_Q27_FALCON, 0, 0.8);
    std::shared_ptr<SimpleSearchEnv> env_copy = env.copy();

    // make a few moves
    int step_count = 0;
    while (step_count < 10) {
        // get state and action space
        auto cur_state = env.get_state();
        auto action_space = env.get_action_space();

        // apply action
        auto selected_action = *action_space.begin();
        double reward = env.step(selected_action);
        cout << reward << endl;

        // check finished
        step_count += 1;
    }
}
