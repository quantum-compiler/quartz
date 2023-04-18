#include <string>
#include <algorithm>
#include <fstream>
#include "env/simple_physical_env.h"
#include "env/simple_hybrid_env.h"

using namespace std;
using namespace quartz;

int main() {
    // test target
    string circuit_file_name = "../asplos_mini_circuit.qasm";
    BackendType backend_type = BackendType::Q5_TEST;
    int num_regs = 5;
    string mapping_file_name = "../asplos_mini_mapping.txt";

    // init mapping
    string mapping;
    for (int i = 0; i < num_regs; ++i) {
        mapping += to_string(i);
    }
    // init result array
    vector<int> results = vector(10, 0);

    // run BFS for each possible mapping to get the distribution
    do {
        // write current mapping into file
        ofstream out_file;
        out_file.open(mapping_file_name);
        string formatted_mapping;
        for (auto s: mapping) {
            formatted_mapping += string(1, s) + " ";
        }
        out_file << formatted_mapping;
        out_file.close();

        // init environment and run BFS search
        SimpleHybridEnv env = SimpleHybridEnv(
                // basic settings
                circuit_file_name, backend_type, mapping_file_name,
                // GameBuffer
                0, 0.8, 5, 3,
                // GameHybrid
                0, true, -0.3);
        queue<SimpleHybridEnv> query_list;
        query_list.emplace(env);
        while (!query_list.empty()) {
            // get cur env
            auto cur_env = query_list.front();
            query_list.pop();

            // expand all possible actions
            bool found = false;
            auto found_env = cur_env;
            for (auto action : cur_env.get_action_space()) {
                auto new_env = cur_env;
                new_env.step(action);
                if (new_env.is_finished()) {
                    found = true;
                    found_env = new_env;
                    break;
                } else {
                    query_list.emplace(new_env);
                }
            }

            // if found, break
            if (found) {
                int swap_count = found_env.cur_game_ptr->swaps_inserted;
                results[swap_count] += 1;
                cout << "Mapping: " << mapping << ", swap count " << swap_count << ".\n";
                break;
            }
        }

    } while (std::next_permutation(mapping.begin(), mapping.end()));

    // output results
    for (int i: results) std::cout << i << ' ';
}
