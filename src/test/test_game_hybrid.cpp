#include "quartz/game/game_hybrid.h"
#include "quartz/supported_devices/supported_devices.h"

#include <string>
#include <cstdlib>

using namespace std;
using namespace quartz;

int main() {
    // initialize the environment
    // "../circuit/nam-circuits/qasm_files/gf2^E5_mult_after_heavy.qasm"
    string circuit_file_name = "../tof_3_after_heavy.qasm";
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
    auto fidelity_graph = GetFidelityGraph(BackendType::IBM_Q27_FALCON);

    // initialize game
    GameHybrid current_game = GameHybrid(*graph, device, fidelity_graph, 5, true, -0.3);

    // make a few moves
    int step_count = 0;
    while (true) {
        if (is_circuit_finished(current_game.graph)) break;
        // get state and action space
        auto cur_state = current_game.state();
        auto action_space = current_game.action_space();

        // apply action
        int selected_action_id = rand() % action_space.size();
        auto selected_action = *next(action_space.begin(), selected_action_id);
        double reward = current_game.apply_action(selected_action);
        cout << reward << " " << action_space.size() << " " << selected_action_id << endl;

        // check finished
        step_count += 1;
    }
    cout << current_game.total_cost() << endl;
}
