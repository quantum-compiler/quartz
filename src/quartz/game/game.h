#pragma once

#include <utility>

#include "../tasograph/tasograph.h"
#include "../sabre/sabre_swap.h"
#include "game_utils.h"

namespace quartz {
    class Game {
    public:
        Game() = delete;

        Game(const Graph &_graph, std::shared_ptr<DeviceTopologyGraph> _device) : graph(_graph),
                                                                                  device(std::move(_device)) {
            /// Game expects that the input graph has been initialized !!!
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

            // reward related: execute all currently executable gates and set imp cost
            executed_logical_gate_count = 0;
            swaps_inserted = 0;
            while (true) {
                auto executable_gate_list = find_executable_front_gates(graph, device);
                for (const auto &executable_gate: executable_gate_list) {
                    assert(graph.inEdges[executable_gate].size() == 2);
                    Edge in_edge_0 = *(graph.inEdges[executable_gate].begin());
                    Edge in_edge_1 = *(std::next(graph.inEdges[executable_gate].begin()));
                    if (in_edge_0.dstIdx == 1) std::swap(in_edge_0, in_edge_1);
                    int input_logical_0 = in_edge_0.logical_qubit_idx;
                    int input_logical_1 = in_edge_1.logical_qubit_idx;
                    int input_physical_0 = in_edge_0.physical_qubit_idx;
                    int input_physical_1 = in_edge_1.physical_qubit_idx;
                    assert(input_physical_0 == initial_logical2physical[input_logical_0]);
                    assert(input_physical_1 == initial_logical2physical[input_logical_1]);
                    assert(input_logical_0 == initial_physical2logical[input_physical_0]);
                    assert(input_logical_1 == initial_physical2logical[input_physical_1]);
                    execution_history.emplace_back(executable_gate.guid, executable_gate.ptr->tp,
                                                   input_logical_0, input_logical_1,
                                                   input_physical_0, input_physical_1);
                    execute_front_gate(graph, executable_gate);
                    executed_logical_gate_count += 1;
                }
                if (executable_gate_list.empty()) break;
            }
            imp_cost = graph.circuit_implementation_cost(device);
        }

        [[nodiscard]] State state() {
            return {device_edges, logical2physical, physical2logical, graph.convert_circuit_to_state()};
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

        Reward apply_action(const Action &action) {
            if (action.type == ActionType::PhysicalFront || action.type == ActionType::PhysicalFull) {
                // physical action

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

                // STEP 2: execute all gates that are enabled by this swap and record them
                int executed_gate_count = 0;
                while (true) {
                    auto executable_gate_list = find_executable_front_gates(graph, device);
                    for (const auto &executable_gate: executable_gate_list) {
                        assert(graph.inEdges[executable_gate].size() == 2);
                        Edge in_edge_0 = *(graph.inEdges[executable_gate].begin());
                        Edge in_edge_1 = *(std::next(graph.inEdges[executable_gate].begin()));
                        if (in_edge_0.dstIdx == 1) std::swap(in_edge_0, in_edge_1);
                        int input_logical_0 = in_edge_0.logical_qubit_idx;
                        int input_logical_1 = in_edge_1.logical_qubit_idx;
                        int input_physical_0 = in_edge_0.physical_qubit_idx;
                        int input_physical_1 = in_edge_1.physical_qubit_idx;
                        assert(input_physical_0 == logical2physical[input_logical_0]);
                        assert(input_physical_1 == logical2physical[input_logical_1]);
                        assert(input_logical_0 == physical2logical[input_physical_0]);
                        assert(input_logical_1 == physical2logical[input_physical_1]);
                        execution_history.emplace_back(executable_gate.guid, executable_gate.ptr->tp,
                                                       input_logical_0, input_logical_1,
                                                       input_physical_0, input_physical_1);
                        execute_front_gate(graph, executable_gate);
                        executed_gate_count += 1;
                    }
                    if (executable_gate_list.empty()) break;
                }
                executed_logical_gate_count += executed_gate_count;
                swaps_inserted += 1;

                // STEP 3: calculate reward
                double original_circuit_cost = imp_cost;
                imp_cost = graph.circuit_implementation_cost(device);
                double new_circuit_cost = imp_cost + executed_gate_count + SWAPCOST;
                Reward reward = original_circuit_cost - new_circuit_cost;
                return reward;
            } else if (action.type == ActionType::Logical) {
                // logical action
                // TODO: implement this
                std::cout << "Logical action not implemented" << std::endl;
                assert(false);
                return NAN;
            } else {
                // unknown
                std::cout << "Unknown action type" << std::endl;
                assert(false);
                return NAN;
            }
        }

        [[nodiscard]] int total_cost() const {
            // this function can only be called at the end of a game
            // some quick sanity checks
            assert(is_circuit_finished(graph));
            assert(original_gate_count == single_qubit_gate_count + executed_logical_gate_count);
            assert(execution_history.size() == swaps_inserted + executed_logical_gate_count);
            assert(check_execution_history(graph, device, execution_history) == ExecutionHistoryStatus::VALID);
            return original_gate_count + int(SWAPCOST) * swaps_inserted;
        }

        void save_execution_history_to_file(const std::string& eh_file_name,
                                            const std::string& qasm_file_name,
                                            bool include_swap = true) const {
            assert(is_circuit_finished(graph));

            // initialize file
            std::ofstream eh_file;
            std::ofstream qasm_file;
            eh_file.open(eh_file_name);
            qasm_file.open(qasm_file_name);

            // output initial mapping table
            eh_file << logical2physical.size() << " " << logical_qubit_num << "\n";
            for (int idx : initial_physical2logical) {
                eh_file << idx << " ";
            }
            eh_file << "\n";
            for (int idx : initial_logical2physical) {
                eh_file << idx << " ";
            }
            eh_file << "\n";

            // output execution history
            eh_file << execution_history.size() << "\n";
            for (const ExecutionHistory& eh : execution_history) {
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
            for (const ExecutionHistory& eh : execution_history) {
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
        int executed_logical_gate_count;  // we do not consider swaps here
        int swaps_inserted;
        double imp_cost;

        // execution history & initial mapping table
        std::vector<int> initial_logical2physical;
        std::vector<int> initial_physical2logical;
        std::vector<ExecutionHistory> execution_history;
    };
}