#pragma once

#include <utility>

#include "../tasograph/tasograph.h"
#include "../sabre/sabre_swap.h"
#include "game_utils.h"

namespace quartz {
    /// GameHybrid implements the underlying environment for joint RL agent
    class GameHybrid {
    public:
        GameHybrid() = delete;

        GameHybrid(const Graph &_graph, std::shared_ptr<DeviceTopologyGraph> _device,
                   int _initial_phase_len, bool _allow_nop_in_initial, double _initial_phase_reward)
                : graph(_graph), device(std::move(_device)) {
            /// GameHybrid expects that the input graph has been initialized !!!

            // STEP 1: record phase info
            Assert(_initial_phase_len > 0, "Initial phase must have length > 0!");
            initial_phase_len = _initial_phase_len;
            allow_nop_in_initial = _allow_nop_in_initial;
            initial_phase_reward = _initial_phase_reward;
            is_initial_phase = true;
            initial_phase_action_type = ActionType::PhysicalFull;   // Type of initial phase action

            // STEP 2: simplify circuit
            original_gate_count = graph.gate_count();
            single_qubit_gate_count = simplify_circuit(graph);

            // STEP 3: state related (mapping table)
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

            // STEP 4: save device edges
            for (int i = 0; i < device->get_num_qubits(); ++i) {
                auto neighbor_list = device->get_input_neighbours(i);
                for (int j: neighbor_list) {
                    device_edges.emplace_back(i, j);
                }
            }

            // STEP 5: record some extra statistics (we do not execute gates at the beginning)
            executed_logical_gate_count = 0;
            swaps_inserted = 0;
            virtual_swaps_inserted = 0;
            imp_cost = graph.circuit_implementation_cost(device);
        }

        GameHybrid(const GameHybrid &game) : graph(game.graph) {
            // graph & device (graph is directly copied)
            device = game.device;
            device_edges = game.device_edges;

            // phase info
            initial_phase_len = game.initial_phase_len;
            allow_nop_in_initial = game.allow_nop_in_initial;
            is_initial_phase = game.is_initial_phase;
            initial_phase_reward = game.initial_phase_reward;
            initial_phase_action_type = game.initial_phase_action_type;

            // full mapping table
            logical_qubit_num = game.logical_qubit_num;
            physical_qubit_num = game.physical_qubit_num;
            logical2physical = game.logical2physical;
            physical2logical = game.physical2logical;

            // reward related
            original_gate_count = game.original_gate_count;
            single_qubit_gate_count = game.single_qubit_gate_count;
            executed_logical_gate_count = game.executed_logical_gate_count;
            swaps_inserted = game.swaps_inserted;
            virtual_swaps_inserted = game.virtual_swaps_inserted;
            imp_cost = game.imp_cost;

            // execution history
            initial_logical2physical = game.initial_logical2physical;
            initial_physical2logical = game.initial_physical2logical;
            execution_history = game.execution_history;
        }

        [[nodiscard]] State state() {
            return {device_edges, logical2physical, physical2logical,
                    graph.convert_circuit_to_state(7, true),
                    is_initial_phase};
        }

        std::set<Action, ActionCompare> action_space() {
            if (is_initial_phase) {
                // We are in stage 1 (add free swaps to change mapping), we use Physical Full instead of
                // Search Full to avoid bumping into bad mappings on large devices.
                // Physical Full: swaps between one used physical qubit and any of its physical neighbors
                std::set<Action, ActionCompare> physical_action_space;
                for (const auto &qubit_pair: graph.qubit_mapping_table) {
                    int physical_idx = qubit_pair.second.second;
                    auto neighbor_list = device->get_input_neighbours(physical_idx);
                    for (int neighbor: neighbor_list) {
                        physical_action_space.insert(Action(initial_phase_action_type,
                                                            std::min(neighbor, physical_idx),
                                                            std::max(neighbor, physical_idx)));
                    }
                }
                // Add nop into action space if allowed
                if (allow_nop_in_initial) {
                    physical_action_space.insert(Action(initial_phase_action_type, 0, 0));
                }
                return std::move(physical_action_space);
            } else {
                // We are in stage 2 (insert swaps into circuit), use PhysicalFront
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
            }
        }

        Reward apply_action(const Action &action) {
            if (is_initial_phase) {
                // In stage 1, we should have SearchFull action space
                Assert(action.type == initial_phase_action_type, "Action should be the same as initial phase mapping type in phase 1!");
                Assert(virtual_swaps_inserted < initial_phase_len, "Phase 1 termination error!");
                virtual_swaps_inserted += 1;
                if (virtual_swaps_inserted == initial_phase_len) is_initial_phase = false;

                // check if this action is nop (nop terminates phase 1 immediately and has reward 0)
                if (action.qubit_idx_0 == 0 && action.qubit_idx_1 == 0) {
                    Assert(allow_nop_in_initial, "Found NOP in phase 1, while allow_nop = False.");
                    execution_history.emplace_back(-2, GateType::swap, 0, 0, 0, 0);
                    is_initial_phase = false;
                    return 0;
                }

                // STEP 1: put swap into history & change mapping tables
                // put action into execution history
                int physical_0 = action.qubit_idx_0;
                int physical_1 = action.qubit_idx_1;
                int logical_0 = physical2logical[physical_0];
                int logical_1 = physical2logical[physical_1];
                execution_history.emplace_back(-2, GateType::swap, logical_0, logical_1, physical_0, physical_1);
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

                // STEP 2: return reward
                imp_cost = graph.circuit_implementation_cost(device);
                return initial_phase_reward;
            } else {
                // In stage 2, we should have PhysicalFront action space
                Assert(action.type == ActionType::PhysicalFront, "Action should be PhysicalFront in phase 2!");

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

                // STEP 3: calculate imp cost (not used)
                // We use -3 reward here instead. Recover the following four lines if we want to use imp reward
//                 double original_circuit_cost = imp_cost;
//                 imp_cost = graph.circuit_implementation_cost(device);
//                 double new_circuit_cost = imp_cost + executed_gate_count + SWAPCOST;
//                 Reward imp_reward = original_circuit_cost - new_circuit_cost;
                return -SWAPCOST + executed_gate_count;
            }
        }

        [[nodiscard]] int total_cost() const {
            // this function can only be called at the end of a game
            // some quick sanity checks
            Assert(is_circuit_finished(graph),
                   "Circuit should be finished when call total cost!");
            Assert(original_gate_count == single_qubit_gate_count + executed_logical_gate_count,
                   "original gate count != single qubit gate count + executed logical gate count!");
            Assert(execution_history.size() == swaps_inserted + virtual_swaps_inserted + executed_logical_gate_count,
                   "Execution history size mismatch with gates executed!");
            Assert(check_execution_history(graph, device, execution_history, false) == ExecutionHistoryStatus::VALID,
                   "Execution history should be valid!");

            // scan through the execution history to determine cost of each swap
            std::vector<bool> is_qubit_used = std::vector<bool>(physical_qubit_num, false);
            int swap_with_cost = 0;
            for (ExecutionHistory eh_item: execution_history) {
                if (eh_item.gate_type == GateType::swap) {
                    // only swaps with at least one logical input used have non-zero cost
                    int _logical0 = eh_item.logical0;
                    int _logical1 = eh_item.logical1;
                    if (is_qubit_used[_logical0] || is_qubit_used[_logical1]) swap_with_cost += 1;

                    // check if virtual swaps indeed have cost 0
                    if (eh_item.guid == -2) {
                        Assert(!is_qubit_used[_logical0] && !is_qubit_used[_logical1],
                               "Virtual Swaps in phase one must have cost 0!");
                    }
                } else {
                    // set target logical qubit as used
                    int _logical0 = eh_item.logical0;
                    int _logical1 = eh_item.logical1;
                    is_qubit_used[_logical0] = true;
                    is_qubit_used[_logical1] = true;
                }
            }
            Assert(swap_with_cost <= swaps_inserted,
                   "Swaps with cost must be fewer than swaps inserted!");

            return original_gate_count + int(SWAPCOST) * swap_with_cost;
        }

        void save_execution_history_to_file(const std::string &eh_file_name,
                                            const std::string &qasm_file_name,
                                            bool include_swap = true) const {
            Assert(is_circuit_finished(graph), "Circuit must be finished!");

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
            // TODO: the qasm output is not well defined, fix this if necessary
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

        // phase info
        int initial_phase_len;
        bool allow_nop_in_initial;
        bool is_initial_phase;
        double initial_phase_reward;
        ActionType initial_phase_action_type;

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
        int virtual_swaps_inserted;  // swaps in phase 1 are virtual swaps
        double imp_cost;

        // execution history & initial mapping table
        std::vector<int> initial_logical2physical;
        std::vector<int> initial_physical2logical;
        std::vector<ExecutionHistory> execution_history;
    };
}