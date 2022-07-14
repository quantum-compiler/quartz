#include <string>
#include "env/simple_physical_env.h"

using namespace std;
using namespace quartz;

int main() {
    string circuit_file_name = "../sabre.qasm";
    SimplePhysicalEnv env = SimplePhysicalEnv(circuit_file_name, BackendType::IBM_Q20_TOKYO);
    auto action_space = env.get_action_space();
    Reward reward = env.step(Action(ActionType::PhysicalFront,
                                    env.cur_game_ptr->logical2physical[0],
                                    env.cur_game_ptr->logical2physical[0]));
    bool is_finished = env.is_finished();
    cout << "Reward 1 is " << reward << ", is finished = " << is_finished << endl;
    env.reset();
    auto action_space2 = env.get_action_space();
    Reward reward2 = env.step(action_space2[0]);
    bool is_finished2 = env.is_finished();
    cout << "Reward 2 is " << reward2 << ", is finished = " << is_finished2 << endl;
}
