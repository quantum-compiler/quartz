#pragma once

#include <utility>

#include "../tasograph/tasograph.h"
#include "game_utils.h"

namespace quartz {
    class Game {
    public:
        Game() = delete;

        Game(const Graph &_graph, std::shared_ptr<DeviceTopologyGraph> _device) : graph(_graph),
                                                                                  device(std::move(_device)) {
            /// Game expects that the input graph has been initialized !!!
            // reward related
            single_qubit_gate_count = simplify_circuit(graph);
            imp_cost = graph.circuit_implementation_cost(device);

            // state related (mapping table)
            // initialize
            logical_qubit_num = static_cast<int>(graph.qubit_mapping_table.size());
            physical_qubit_num = device->get_num_qubits();
            for (int i = 0; i < physical_qubit_num; ++i) {
                logical2physical.emplace_back(-1);
                physical2logical.emplace_back(-1);
            }
            // copy partial one from graph
            std::set<int> occupied_physical_idx_set;
            std::vector<int> free_physical_idx_list;
            for (const auto &qubit_pair: graph.qubit_mapping_table) {
                int tmp_logical_idx = qubit_pair.second.first;
                int tmp_physical_idx = qubit_pair.second.second;
                logical2physical[tmp_logical_idx] = tmp_physical_idx;
                physical2logical[tmp_physical_idx] = tmp_logical_idx;
                occupied_physical_idx_set.insert(tmp_physical_idx);
            }
            for (int i = 0; i < physical_qubit_num; ++i) {
                if (occupied_physical_idx_set.find(i) == occupied_physical_idx_set.end()) {
                    free_physical_idx_list.emplace_back(i);
                }
            }
            // finish whole mapping table
            for (int logical_idx = 0; logical_idx < physical_qubit_num; ++logical_idx) {
                if (logical2physical[logical_idx] == -1) {
                    // get free physical idx from list
                    int free_physical_idx = free_physical_idx_list.front();
                    free_physical_idx_list.erase(free_physical_idx_list.begin());
                    assert(physical2logical[free_physical_idx] == -1);
                    // assign
                    logical2physical[logical_idx] = free_physical_idx;
                    physical2logical[free_physical_idx] = logical_idx;
                }
            }
        }

        State state() {
            // TODO: implement this
            return {};
        }

        std::set<Action, ActionCompare> action_space(ActionType action_type) {
            if (action_type == ActionType::PhysicalFull) {
                // Physical Full: swaps between physical neighbors of all used logical qubits
                std::set<Action, ActionCompare> physical_action_space;
                for (const auto &qubit_pair: graph.qubit_mapping_table) {
                    int physical_idx = qubit_pair.second.second;
                    auto neighbor_list = device->get_input_neighbours(physical_idx);
                    for (int neighbor: neighbor_list) {
                        physical_action_space.insert(Action(ActionType::PhysicalFull,
                                                            std::min(neighbor, physical_idx),
                                                            std::max(neighbor, physical_idx)));
                    }
                }
                return std::move(physical_action_space);
            } else if (action_type == ActionType::PhysicalFront) {
                // Physical Front: only swaps between neighbors of inputs to front gates
                // get gates with at least one input input_qubit
                std::set<Op, OpCompare> tmp_front_gate_set;
                for (const auto &initial_qubit_mapping: graph.qubit_mapping_table) {
                    auto initial_qubit = initial_qubit_mapping.first;
                    assert(graph.outEdges.find(initial_qubit) != graph.outEdges.end()
                           && !graph.outEdges[initial_qubit].empty());
                    for (auto edge: graph.outEdges[initial_qubit]) {
                        tmp_front_gate_set.insert(edge.dstOp);
                    }
                }

                // only retain those real front gates
                std::set<Op, OpCompare> front_gate_set;
                for (const auto &tmp_front_gate: tmp_front_gate_set) {
                    // check all inputs
                    bool is_front_gate = true;
                    for (const auto &in_edge: graph.inEdges[tmp_front_gate]) {
                        if (in_edge.srcOp.ptr->tp != GateType::input_qubit) {
                            is_front_gate = false;
                        }
                    }
                    // append
                    if (is_front_gate) front_gate_set.insert(tmp_front_gate);
                }

                // put their neighbors into action space
                std::set<Action, ActionCompare> physical_action_space;
                for (const auto &op: front_gate_set) {
                    for (const Edge &in_edge: graph.inEdges[op]) {
                        int input_physical_idx = graph.qubit_mapping_table[in_edge.srcOp].second;
                        auto neighbor_list = device->get_input_neighbours(input_physical_idx);
                        for (int neighbor: neighbor_list) {
                            physical_action_space.insert(Action(ActionType::PhysicalFront,
                                                                std::min(neighbor, input_physical_idx),
                                                                std::max(neighbor, input_physical_idx)));
                        }
                    }
                }
                return physical_action_space;
            } else if (action_type == ActionType::Logical) {
                // Logical action space
                std::set<Action, ActionCompare> logical_action_space;
                for (int logical_1 = 0; logical_1 < logical_qubit_num; ++logical_1) {
                    for (int logical_2 = 0; logical_2 < physical_qubit_num; ++logical_2) {
                        if (logical_1 != logical_2) {
                            logical_action_space.insert(Action(ActionType::Logical,
                                                               std::min(logical_1, logical_2),
                                                               std::max(logical_1, logical_2)));
                        }
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
        /// state related
        // mapping table + remaining part of circuit
        Graph graph;
        // device
        std::shared_ptr<DeviceTopologyGraph> device;
        // full mapping table
        // Note that the first #logical_qubit_num elements are the same as the mapping table in graph
        int logical_qubit_num;
        int physical_qubit_num;
        std::vector<int> logical2physical;
        std::vector<int> physical2logical;

        /// reward related
        int single_qubit_gate_count;
        double imp_cost;
    };
}