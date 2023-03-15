#include <string>
#include <filesystem>
#include "env/simple_hybrid_env.h"

using namespace std;
using namespace quartz;
namespace fs = std::filesystem;

int main() {
    // check each circuit in the benchmark for compatibility
    std::string path = "../circuit/quartzphysical/";

    for (const auto & entry : fs::directory_iterator(path)) {
        // initialize the env
        auto circuit_file_path = entry.path().string();
        std::cout << "Testing file:" << circuit_file_path << std::endl;
        SimpleHybridEnv env = SimpleHybridEnv(
                // basic settings
                circuit_file_path, BackendType::IBM_Q127_EAGLE, "../mapping127.txt",
                // GameBuffer
                0, 0.8, 5, 3,
                // GameHybrid
                5, true, -0.3);
//        while (!env.is_finished()) {
//            State state_before = env.get_state();
//            vector<Action> action_space = env.get_action_space();
//            int selected_action_id = rand() % action_space.size();
//            Action selected_action = action_space[selected_action_id];
//            Reward reward = env.step(selected_action);
//            State state_after = env.get_state();
//            bool is_finished = env.is_finished();
//        }
//        env.save_context_to_file("../eh1.txt", "../single1.txt");
//        env.generate_mapped_qasm("../final1.qasm", true);
    }
}
