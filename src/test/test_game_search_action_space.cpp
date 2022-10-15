#include "quartz/game/game_search.h"
#include "quartz/supported_devices/supported_devices.h"

#include <string>
#include <cstdlib>

using namespace std;
using namespace quartz;

int main() {
    // initialize the environment
    string circuit_file_name = "../circuit/nam-circuits/qasm_files/tof_3_after_heavy.qasm";
    // initialize context, graph and device
    auto context = std::make_shared<Context>(Context({GateType::h, GateType::cx, GateType::t,
                                                      GateType::tdg, GateType::input_qubit, GateType::s,
                                                      GateType::sdg}));
    QASMParser qasm_parser(&(*context));
    DAG *dag = nullptr;
    if (!qasm_parser.load_qasm(circuit_file_name, dag)) {
        std::cout << "Parser failed" << std::endl;
        assert(false);
    }
    auto graph = std::make_shared<Graph>(Graph(&(*context), dag));
    set_initial_mapping(*graph, 0, "../test_mapping_file.txt", 27);
    auto device = GetDevice(BackendType::IBM_Q27_FALCON);

    // initialize game
    GameSearch current_game = GameSearch(*graph, device);

    // make a few moves
    int step_count = 0;
    while (step_count < 10) {
        // get state and action space
        auto cur_state = current_game.state();
        auto action_space = current_game.action_space(ActionType::SearchFull);

        // apply action
        auto selected_action = *action_space.begin();
        double reward = current_game.apply_action(selected_action);
        cout << reward << endl;

        // check finished
        step_count += 1;
    }
}
