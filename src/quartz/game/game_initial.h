#pragma once

#include <utility>

#include "../tasograph/tasograph.h"
#include "../sabre/sabre_swap.h"
#include "game_utils.h"

namespace quartz {
    class GameInitial {
    public:
        GameInitial() = delete;

        GameInitial(const Graph &_graph, std::shared_ptr<DeviceTopologyGraph> _device) : graph(_graph),
                                                                                         device(std::move(_device)) {
            /// GameInitial expects that the input graph has been initialized !!!
            // simplify circuit
            original_gate_count = graph.gate_count();

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
            // save into initial cache
            initial_logical2physical = logical2physical;
            initial_physical2logical = physical2logical;

            // save device edges
            for (int i = 0; i < device->get_num_qubits(); ++i) {
                auto neighbor_list = device->get_input_neighbours(i);
                for (int j: neighbor_list) {
                    device_edges.emplace_back(i, j);
                }
            }
        }

        [[nodiscard]] State state() {
            return {device_edges, logical2physical, physical2logical,
                    graph.convert_circuit_to_state(7, true)};
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
            } else {
                // the game of initial mapping only supports physical full action space
                std::cout << "Unsupported action space type." << std::endl;
                assert(false);
                return {};
            }
        }

        Reward apply_action(const Action &action) {
            if (action.type == ActionType::PhysicalFull) {
                // STEP 1: put swap into history & change mapping tables
                // put action into execution history
                int physical_0 = action.qubit_idx_0;
                int physical_1 = action.qubit_idx_1;
                int logical_0 = physical2logical[physical_0];
                int logical_1 = physical2logical[physical_1];
                execution_history.emplace_back(-1, GateType::swap, logical_0, logical_1, physical_0, physical_1);
                // change full mapping table
                logical2physical[logical_0] = physical_1;
                logical2physical[logical_1] = physical_0;
                physical2logical[physical_0] = logical_1;
                physical2logical[physical_1] = logical_0;
                // change mapping table in graph and propagate
                int hit_count = 0;
                for (auto &input_mapping_pair: graph.qubit_mapping_table) {
                    int cur_logical_idx = input_mapping_pair.second.first;
                    if (cur_logical_idx == logical_0) {
                        input_mapping_pair.second.second = physical_1;
                        hit_count += 1;
                    }
                    if (cur_logical_idx == logical_1) {
                        input_mapping_pair.second.second = physical_0;
                        hit_count += 1;
                    }
                }
                assert(hit_count == 1 || hit_count == 2);
                graph.propagate_mapping();
                return -1;
            } else {
                // the game of initial mapping only supports physical full action space
                std::cout << "Unknown action type" << std::endl;
                assert(false);
                return NAN;
            }
        }

        [[nodiscard]] static int total_cost() {
            std::cout << "Invalid call to total cost in GameInitial" << std::endl;
            assert(false);
            return NAN;
        }

        void save_execution_history_to_file(const std::string &eh_file_name,
                                            const std::string &qasm_file_name,
                                            bool include_swap = true) const {
            // initialize file
            std::ofstream eh_file;
            std::ofstream qasm_file;
            eh_file.open(eh_file_name);
            qasm_file.open(qasm_file_name);

            // output initial mapping table
            eh_file << logical2physical.size() << " " << logical_qubit_num << "\n";
            for (int idx: initial_physical2logical) {
                eh_file << idx << " ";
            }
            eh_file << "\n";
            for (int idx: initial_logical2physical) {
                eh_file << idx << " ";
            }
            eh_file << "\n";

            // output execution history
            eh_file << execution_history.size() << "\n";
            for (const ExecutionHistory &eh: execution_history) {
                eh_file << eh.guid << " "
                        << eh.gate_type << " "
                        << eh.physical0 << " "
                        << eh.physical1 << " "
                        << eh.logical0 << " "
                        << eh.logical1 << "\n";
            }

            // output qasm file
            qasm_file << "OPENQASM 2.0;" << "\n";
            qasm_file << "include \"qelib1.inc\";" << "\n";
            qasm_file << "qreg q[" << logical_qubit_num << "];\n";
            for (const ExecutionHistory &eh: execution_history) {
                if (!include_swap && eh.gate_type == GateType::swap) continue;
                qasm_file << eh.gate_type << " q[" << eh.logical0 << "], q[" << eh.logical1 << "];\n";
            }

            // clean up
            eh_file.close();
            qasm_file.close();
        }

    public:
        // graph & device
        Graph graph;
        std::shared_ptr<DeviceTopologyGraph> device;
        std::vector<std::pair<int, int>> device_edges;

        // full mapping table
        // Note that the first #logical_qubit_num elements are the same as the mapping table in graph
        int logical_qubit_num;
        int physical_qubit_num;
        std::vector<int> logical2physical;
        std::vector<int> physical2logical;

        // reward related
        int original_gate_count;  // number of gates in the graph at beginning

        // execution history & initial mapping table
        std::vector<int> initial_logical2physical;
        std::vector<int> initial_physical2logical;
        std::vector<ExecutionHistory> execution_history;
    };
}