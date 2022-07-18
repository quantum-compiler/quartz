#include <string>
#include "env/simple_physical_env.h"

using namespace std;
using namespace quartz;

int main() {
    // initialize the environment
    string circuit_file_name = "../sabre.qasm";
    SimplePhysicalEnv env = SimplePhysicalEnv(circuit_file_name, BackendType::IBM_Q20_TOKYO);

    // apply one step on the environment
    State state_before = env.get_state();
    vector<Action> action_space = env.get_action_space();
    Action selected_action = Action(ActionType::PhysicalFront,
                                    env.cur_game_ptr->logical2physical[0],
                                    env.cur_game_ptr->logical2physical[0]);
    Reward reward = env.step(selected_action);
    State state_after = env.get_state();
    bool is_finished = env.is_finished();
    cout << "Reward 1 is " << reward << ", is finished = " << is_finished << endl;

    // reset the environment to start a new game
    env.reset();
    State state_before2 = env.get_state();
    vector<Action> action_space2 = env.get_action_space();
    Reward reward2 = env.step(action_space2[0]);
    State state_after2 = env.get_state();
    bool is_finished2 = env.is_finished();
    cout << "Reward 2 is " << reward2 << ", is finished = " << is_finished2 << endl;
}
