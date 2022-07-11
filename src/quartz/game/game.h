#pragma once

#include "../tasograph/tasograph.h"
#include "game_utils.h"

namespace quartz {
    class State {

    };

    enum class ActionType {
        Physical = 0, logical = 1, unknown = 2
    };

    class Action {
    public:
        Action() : type(ActionType::unknown), qubit_idx_0(-1), qubit_idx_1(-1) {}

        Action(ActionType _type, int _qubit_idx_0, int _qubit_idx_1) : type(_type), qubit_idx_0(_qubit_idx_0),
                                                                       qubit_idx_1(_qubit_idx_1) {}

    public:
        ActionType type;
        int qubit_idx_0;
        int qubit_idx_1;
    };

    class Game {
    public:
        Game() = delete;

        Game(const Graph &_graph, std::shared_ptr<DeviceTopologyGraph> &_device) : graph(_graph), device(_device) {
            /// Game expects that the input graph has been initialized !!!
            single_qubit_gate_count = simplify_circuit(graph);
            imp_cost = graph.circuit_implementation_cost(device);
        }

    public:
        // state related
        Graph graph;
        std::shared_ptr<DeviceTopologyGraph> device;

        // action related

        // reward related
        int single_qubit_gate_count;
        double imp_cost;
    };
}