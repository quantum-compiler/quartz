#pragma once

#include <utility>

#include "../tasograph/tasograph.h"
#include "game_utils.h"

namespace quartz {
    class State {

    };

    enum class ActionType {
        Physical = 0, Logical = 1, Unknown = 2
    };

    class Action {
    public:
        Action() : type(ActionType::Unknown), qubit_idx_0(-1), qubit_idx_1(-1) {}

        Action(ActionType _type, int _qubit_idx_0, int _qubit_idx_1) : type(_type), qubit_idx_0(_qubit_idx_0),
                                                                       qubit_idx_1(_qubit_idx_1) {}

    public:
        ActionType type;
        int qubit_idx_0;
        int qubit_idx_1;
    };

    struct ActionCompare {
        bool operator()(const Action &a, const Action &b) const {
            if (a.type != b.type) return a.type < b.type;
            if (a.qubit_idx_0 != b.qubit_idx_0) return a.qubit_idx_0 < b.qubit_idx_0;
            return a.qubit_idx_1 < b.qubit_idx_1;
        };
    };

    class Game {
    public:
        Game() = delete;

        Game(const Graph &_graph, std::shared_ptr<DeviceTopologyGraph> _device) : graph(_graph),
                                                                                  device(std::move(_device)) {
            /// Game expects that the input graph has been initialized !!!
            single_qubit_gate_count = simplify_circuit(graph);
            imp_cost = graph.circuit_implementation_cost(device);
        }

        State state() {
            // TODO: implement this
            return {};
        }

        std::set<Action, ActionCompare> action_space(ActionType action_type) {
            if (action_type == ActionType::Physical) {
                // Physical action space
                std::set<Action, ActionCompare> physical_action_space;
                for (const auto& qubit_pair : graph.qubit_mapping_table) {
                    int physical_idx = qubit_pair.second.second;
                    auto neighbor_list = device->get_input_neighbours(physical_idx);
                    for (int neighbor: neighbor_list) {
                        physical_action_space.insert(Action(ActionType::Physical,
                                                            std::min(neighbor, physical_idx),
                                                            std::max(neighbor, physical_idx)));
                    }
                }
                return std::move(physical_action_space);
            } else if (action_type == ActionType::Logical) {
                // Logical action space
                std::set<Action, ActionCompare> logical_action_space;
                for (const auto& qubit_pair_1 : graph.qubit_mapping_table) {
                    for (const auto& qubit_pair_2: graph.qubit_mapping_table) {
                        int logical_1 = qubit_pair_1.second.first;
                        int logical_2 = qubit_pair_2.second.first;
                        logical_action_space.insert(Action(ActionType::Logical,
                                                           std::min(logical_1, logical_2),
                                                           std::max(logical_1, logical_2)));
                    }
                }
                return std::move(logical_action_space);
            } else {
                std::cout << "Unknown action space type." << std::endl;
                assert(false);
                return {};
            }
        }

    public:
        // state related
        Graph graph;
        std::shared_ptr<DeviceTopologyGraph> device;

        // reward related
        int single_qubit_gate_count;
        double imp_cost;
    };
}