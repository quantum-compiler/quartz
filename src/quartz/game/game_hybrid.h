#pragma once

#include <utility>
#include <sstream>

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
            swaps_inserted = 0;             // This is for real swaps (i.e. phase 2 swaps)
            virtual_swaps_inserted = 0;     // This is for virtual swaps (i.e. phase 1 swaps)
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
                Assert(action.type == initial_phase_action_type,
                       "Action should be the same as initial phase mapping type in phase 1!");
                Assert(virtual_swaps_inserted < initial_phase_len, "Phase 1 termination error!");
                virtual_swaps_inserted += 1;
                if (virtual_swaps_inserted == initial_phase_len) is_initial_phase = false;

                // check if this action is nop (nop terminates phase 1 immediately and has reward 0 + # executed gates)
                if (action.qubit_idx_0 == 0 && action.qubit_idx_1 == 0) {
                    Assert(allow_nop_in_initial, "Found NOP in phase 1, while allow_nop = False.");

                    // nop terminates phase 1
                    execution_history.emplace_back(-2, GateType::swap, 0, 0, 0, 0);
                    is_initial_phase = false;

                    // execute all executable gates
                    int executed_gate_count = _execute_all_executable_gates();
                    executed_logical_gate_count += executed_gate_count;
                    return 0 + executed_gate_count;
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

                // STEP 2: return reward (condition on whether this is the last step in phase 1)
                imp_cost = graph.circuit_implementation_cost(device);
                if (is_initial_phase) {
                    // Not last step
                    return initial_phase_reward;
                } else {
                    // Last step, need to execute gates
                    int executed_gate_count = _execute_all_executable_gates();
                    executed_logical_gate_count += executed_gate_count;
                    return initial_phase_reward + executed_gate_count;
                }
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
                int executed_gate_count = _execute_all_executable_gates();
                executed_logical_gate_count += executed_gate_count;
                swaps_inserted += 1;

                // STEP 3: calculate imp cost (not used)
                // We use -3 reward here instead. Recover the following four lines if we want to use imp reward
                // ------------------------------------------------------------------------- //
                // double original_circuit_cost = imp_cost;
                // imp_cost = graph.circuit_implementation_cost(device);
                // double new_circuit_cost = imp_cost + executed_gate_count + SWAPCOST;
                // Reward imp_reward = original_circuit_cost - new_circuit_cost;
                // ------------------------------------------------------------------------- //
                return -SWAPCOST + executed_gate_count;
            }
        }

        int _execute_all_executable_gates() {
            // Execute all executable gates and return the number
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
            return executed_gate_count;
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
                    if (is_qubit_used[_logical0] || is_qubit_used[_logical1]) {
                        // the swap has cost, it also marks the input qubits as used.
                        // Note: we think swap2 below has cost.
                        // 1          swap2
                        // 2    swap1 swap2
                        // 3 cx swap1
                        // 4 cx
                        swap_with_cost += 1;
                        is_qubit_used[_logical0] = true;
                        is_qubit_used[_logical1] = true;
                    }

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

        void save_execution_history_to_file(const std::string &eh_file_name) const {
            /* Execution History File Format:
             *
             *      [register count] [logical qubit count]
             *      [initial physical->logical mapping (a list of numbers: p2l[p0], p2l[p1], ...)]
             *      [initial logical->physical mapping (a list of numbers: l2p[l0], l2p[l1], ...)]
             *      [number of entries (i.e. two-qubit gate count + all swap count)]
             *      [guid] [gate type] [physical idx 0] [physical idx 1] [logical idx 0] [logical idx 1]
             *      [guid] [gate type] [physical idx 0] [physical idx 1] [logical idx 0] [logical idx 1]
             *      ...
             *
             * Entry Format Examples:
             *
             *      1. CNOT:             16400 cx 8 3 3 4
             *      2. SWAP:             -1 swap 3 9 4 9
             *      3. Virtual SWAP:     -2 swap 3 9 4 9
             *      4. Phase Transition: -2 swap 0 0 0 0
             *
             * Notice that during RL mapping we only consider two-qubit gates, so the execution history file
             * also only contains such information. To reconstruct a valid mapping plan for the original qasm
             * file, call generated_mapping_plan().
            */

            Assert(is_circuit_finished(graph), "Circuit must be finished!");

            // initialize file
            std::ofstream eh_file;
            eh_file.open(eh_file_name);

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

            // clean up
            eh_file.close();
        }

        void generated_mapping_plan(const std::string &output_qasm_file_path,
                                    const std::string &original_qasm_file_path) const {
            // circuit must be finished in order to generate the plan
            Assert(is_circuit_finished(graph), "Circuit must be finished!");

            // STEP 1: read original qasm file and parse it into a queue of gates (original_qasm_gates)
            std::ifstream original_qasm_file(original_qasm_file_path);
            std::string tmp_file_line;
            std::vector<OutputGateRepresentation> original_qasm_gates;
            int original_qasm_qubit_count = -1;
            while (std::getline(original_qasm_file, tmp_file_line)) {
                // ignore headings and annotations
                if (tmp_file_line.rfind("//", 0) == 0) continue;
                if (tmp_file_line.rfind("OPENQASM", 0) == 0) continue;
                if (tmp_file_line.rfind("include", 0) == 0) continue;
                if (tmp_file_line.rfind("qreg", 0) == 0) {
                    size_t offset = tmp_file_line.find('[') + 1;
                    size_t count = tmp_file_line.find(']') - offset;
                    std::string qubit_count_str = tmp_file_line.substr(offset, count);
                    original_qasm_qubit_count = std::stoi(qubit_count_str);
                    continue;
                }

                // parse each line to get the gates
                // gate type: first non-empty substring split by space
                // qubit idx: numbers between []

                // parse gate type
                // we use string stream here to avoid parsing spaces and tabs
                std::istringstream _tmp_iss(tmp_file_line);
                std::string gate_type_str;
                _tmp_iss >> gate_type_str;
                GateType gate_type = to_gate_type(gate_type_str);

                // parse the input of gates
                auto left_bracket_occurrences = find_all_occurrences(tmp_file_line, '[');
                auto right_bracket_occurrences = find_all_occurrences(tmp_file_line, ']');
                Assert(left_bracket_occurrences.size() == right_bracket_occurrences.size(),
                       "Bracket mismatch!");
                Assert(left_bracket_occurrences.size() == 1 || left_bracket_occurrences.size() == 2,
                       "We only support 1 / 2 qubit gates, found " + std::to_string(left_bracket_occurrences.size()));
                std::vector<int> gate_inputs;
                for (int _idx = 0; _idx < left_bracket_occurrences.size(); ++_idx) {
                    size_t offset = left_bracket_occurrences[_idx] + 1;
                    size_t count = right_bracket_occurrences[_idx] - offset;
                    int input_idx = std::stoi(tmp_file_line.substr(offset, count));
                    gate_inputs.emplace_back(input_idx);
                }

                // assemble them into OutputGateRepresentation
                if (gate_inputs.size() == 1) {
                    original_qasm_gates.emplace_back(true, gate_type, gate_inputs[0], -1);
                } else {
                    original_qasm_gates.emplace_back(false, gate_type, gate_inputs[0], gate_inputs[1]);
                }
            }
            Assert(original_qasm_qubit_count != -1, "Missing qreg line in original qasm file!");

            // STEP 2: parse the execution history into a queue and exclude all virtual and free swaps
            // initial_qubit_coverage: used to determine free swaps.
            // real_initial_l2p_mapping: the l2p mapping table after all virtual and free swaps are executed.
            // output_execution_history: the execution history without virtual and free swaps.
            std::vector<bool> initial_qubit_coverage = std::vector<bool>(physical_qubit_num, false);
            std::vector<int> real_initial_l2p_mapping = initial_logical2physical;
            std::deque<ExecutionHistory> output_execution_history;
            int _swap_with_cost = 0, _two_qubit_gates = 0;
            for (const auto &eh_entry: execution_history) {
                if (eh_entry.gate_type == GateType::swap) {
                    // swap gate, check if it is virtual / free
                    int l0 = eh_entry.logical0;
                    int l1 = eh_entry.logical1;
                    bool is_swap_free = !initial_qubit_coverage[l0] && !initial_qubit_coverage[l1];

                    // check that virtual swaps are indeed free
                    if (eh_entry.guid == -2) Assert(is_swap_free, "Virtual swaps must be free!");

                    // parse swap according to whether it is free
                    if (is_swap_free) {
                        // free swaps only changes the mapping table and will not be recorded
                        // we can check mapping table here, they should match
                        int p0 = eh_entry.physical0;
                        int p1 = eh_entry.physical1;
                        Assert(real_initial_l2p_mapping[l0] == p0 && real_initial_l2p_mapping[l1] == p1,
                               "Free swap mapping mismatch!");

                        // change the mapping table (real_initial_l2p_mapping)
                        real_initial_l2p_mapping[l0] = p1;
                        real_initial_l2p_mapping[l1] = p0;
                    } else {
                        // swaps with cost will be recorded, and they will also mark the inputs as used
                        // mark the corresponding entry as used
                        initial_qubit_coverage[l0] = true;
                        initial_qubit_coverage[l1] = true;

                        // record the swap
                        output_execution_history.emplace_back(eh_entry);
                        _swap_with_cost += 1;
                    }
                } else {
                    // non-swap two-qubit gates will cover the input qubits
                    int l0 = eh_entry.logical0;
                    int l1 = eh_entry.logical1;
                    initial_qubit_coverage[l0] = true;
                    initial_qubit_coverage[l1] = true;
                    output_execution_history.emplace_back(eh_entry);
                    _two_qubit_gates += 1;
                }
            }
            Assert(_swap_with_cost * SWAPCOST + _two_qubit_gates + single_qubit_gate_count == total_cost(),
                   "Cost function mismatch!");

            // STEP 3: generate the output qasm file (with integrity checks)
            // Note that in the output qasm file, the indices are physical qubit indices.

            // open output file
            std::ofstream output_qasm_file;
            output_qasm_file.open(output_qasm_file_path);

            // some structures for qasm generation and integrity check
            // cur_l2p_mapping: the current l2p mapping table, used to determine the physical index and integrity check.
            // real_final_l2p_mapping: the l2p mapping table at the end of execution.
            // pending_cnot_coverage: used in CNOT integrity check.
            // pending_cnot_queue: used in CNOT integrity check and single qubit gate position determination.
            std::vector<int> cur_l2p_mapping = initial_logical2physical;

            std::vector<int> real_final_l2p_mapping;
            std::vector<bool> pending_cnot_coverage = std::vector<bool>(physical_qubit_num, false);
            std::deque<OutputGateRepresentation> pending_cnot_queue;

            // walk through the history
            while (!output_execution_history.empty()) {
                // pop the first eh entry
                auto cur_eh_entry = output_execution_history.front();
                output_execution_history.pop_front();

                if (cur_eh_entry.gate_type == GateType::swap) {
                    // the gate is a swap gate, first check if this swap is free
                    int swap_logical_idx_0 = cur_eh_entry.logical0;
                    int swap_logical_idx_1 = cur_eh_entry.logical1;
                    Assert(cur_l2p_mapping[swap_logical_idx_0] == cur_eh_entry.physical0, "Mapping error!");
                    Assert(cur_l2p_mapping[swap_logical_idx_1] == cur_eh_entry.physical1, "Mapping error!");
                    bool is_cur_swap_free =
                            !initial_qubit_coverage[swap_logical_idx_0] && !initial_qubit_coverage[swap_logical_idx_1];

                    // virtual swaps must be free
                    if (cur_eh_entry.guid == -2) Assert(is_cur_swap_free, "Virtual swaps must be free!");

                    // apply the swap
                    if (is_cur_swap_free) {
                        // free swaps only change cur_l2p_mapping (and real_initial_l2p_mapping)
                        // it will not be written into the final qasm file
                    } else {

                    }
                } else {
                    // the gate is cnot or other two qubit gates
                }
            }

            // clean up
            output_qasm_file.close();
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