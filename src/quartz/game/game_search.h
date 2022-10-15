#pragma once

#include <utility>

#include "../tasograph/tasograph.h"
#include "../sabre/sabre_swap.h"
#include "game_utils.h"

namespace quartz {
    class GameSearch {
    public:
        GameSearch() = delete;

        GameSearch(const Graph &_graph, std::shared_ptr<DeviceTopologyGraph> _device)
                : graph(_graph), device(std::move(_device)) {
            /// GameSearch expects that the input graph has been initialized !!!
            /// Game for initial mapping search
            // simplify circuit
            original_gate_count = graph.gate_count();
            single_qubit_gate_count = simplify_circuit(graph);

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

            // track the number of swaps inserted (distance from root)
            swaps_inserted = 0;
        }

        GameSearch(const GameSearch &game) : graph(game.graph) {
            // graph & device
            // graph is directly copied
            device = game.device;
            device_edges = game.device_edges;

            // full mapping table
            logical_qubit_num = game.logical_qubit_num;
            physical_qubit_num = game.physical_qubit_num;
            logical2physical = game.logical2physical;
            physical2logical = game.physical2logical;

            // reward related
            original_gate_count = game.original_gate_count;
            single_qubit_gate_count = game.single_qubit_gate_count;
            swaps_inserted = game.swaps_inserted;

            // execution history
            initial_logical2physical = game.initial_logical2physical;
            initial_physical2logical = game.initial_physical2logical;
            execution_history = game.execution_history;
        }

        [[nodiscard]] State state() {
            return {device_edges, logical2physical, physical2logical,
                    graph.convert_circuit_to_state(7, true)};
        }

        std::set<Action, ActionCompare> action_space(ActionType action_type) {
            if (action_type == ActionType::SearchFull) {
                // Physical Full: The Physical
                std::set<Action, ActionCompare> physical_action_space;
                for (const auto &qubit_pair: graph.qubit_mapping_table) {
                    int physical_idx = qubit_pair.second.second;
                    for (int other = 0; other < physical_qubit_num; ++other) {
                        if (other == physical_idx) continue;
                        physical_action_space.insert(Action(ActionType::SearchFull,
                                                            std::min(other, physical_idx),
                                                            std::max(other, physical_idx)));
                    }
                }
                return std::move(physical_action_space);
            } else {
                std::cout << "GameSearch only supports SearchFull" << std::endl;
                assert(false);
                return {};
            }
        }

        Reward apply_action(const Action &action) {
            if (action.type == ActionType::SearchFull) {
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
                swaps_inserted += 1;

                // STEP 2: return 0 as reward (since search uses value network's result as reward)
                return 0;
            } else {
                // unknown
                std::cout << "GameSearch only supports PhysicalFull" << std::endl;
                assert(false);
                return NAN;
            }
        }

        void save_execution_history_to_file(const std::string &eh_file_name,
                                            const std::string &qasm_file_name,
                                            bool include_swap = true) const {
            assert(is_circuit_finished(graph));

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
        int single_qubit_gate_count;
        int swaps_inserted;

        // execution history & initial mapping table
        std::vector<int> initial_logical2physical;
        std::vector<int> initial_physical2logical;
        std::vector<ExecutionHistory> execution_history;
    };
}