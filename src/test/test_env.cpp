#include <string>
#include "../quartz/env/env.h"

using namespace std;
using namespace quartz;

int main() {
    string circuit_file_name = "../sabre.qasm";
    SimplePhysicalEnv env = SimplePhysicalEnv(circuit_file_name, BackendType::IBM_Q20_TOKYO);
    auto action_space = env.get_action_space();
    Reward reward = env.step(Action(ActionType::PhysicalFront, 0, 0));
    bool is_finished = env.is_finished();
    env.reset();
    auto action_space2 = env.get_action_space();
    Reward reward2 = env.step(Action(ActionType::PhysicalFront, 0, 0));
    bool is_finished2 = env.is_finished();
}
