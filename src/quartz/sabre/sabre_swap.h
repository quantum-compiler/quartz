#pragma once

#include <utility>

#include "../sabre/sabre.h"
#include "../tasograph/tasograph.h"

namespace quartz {
    using QubitMappingTable = std::unordered_map<Op, std::pair<int, int>, OpHash>;

    struct ExecutionHistory {
    public:
        int guid;
        GateType gate_type;
        int logical0;
        int logical1;
        int physical0;
        int physical1;
        std::string parameter_string;
    public:
        ExecutionHistory() : guid{-10}, logical0{-10}, logical1{-10},
                             physical0{-10}, physical1{-10}, gate_type{GateType::input_param} {}

        ExecutionHistory(int _guid, GateType _gate_type, int _logical0, int _logical1,
                         int _physical0, int _physical1, std::string _parameter_string="")
                : guid(_guid), gate_type(_gate_type), logical0(_logical0), logical1(_logical1),
                  physical0(_physical0), physical1(_physical1), parameter_string(std::move(_parameter_string)) {}
    };

    std::string gate_type_to_string(GateType t) {
        const std::string name_list[] = {"h", "x", "y", "rx", "ry", "rz", "cx", "ccx", "add",
                                         "neg", "z", "s", "sdg", "t", "tdg", "ch", "swap", "p",
                                         "pdg", "rx1", "rx3", "u1", "u2", "u3", "ccz", "cz",
                                         "input_qubit", "input_param"
        };
        return name_list[int(t)];
    }

    std::vector<ExecutionHistory> _sabre_swap_loop(Graph graph, const std::shared_ptr<DeviceTopologyGraph> &device,
                                                   bool use_extensive, double w_value) {
        // returns the logical mapping at the end after sabre pass
        // initialize execution history list
        std::vector<ExecutionHistory> execution_history_list;
        // initialize mapping
        std::vector<int> logical2physical;
        std::vector<int> physical2logical;
        std::vector<double> decay_list;
        QubitMappingTable initial_qubit_mapping = graph.qubit_mapping_table;
        logical2physical.reserve(initial_qubit_mapping.size());
        physical2logical.reserve(device->get_num_qubits());
        for (int i = 0; i < device->get_num_qubits(); i++) {
            physical2logical.emplace_back(-1);
            decay_list.emplace_back(1.0);
        }
        for (int i = 0; i < initial_qubit_mapping.size(); i++) {
            logical2physical.emplace_back(-1);
        }
        for (const auto &qubit_mapping: graph.qubit_mapping_table) {
            int logical = qubit_mapping.second.first;
            int physical = qubit_mapping.second.second;
            logical2physical[logical] = physical;
            physical2logical[physical] = logical;
        }
        // initialize front set f
        std::unordered_set<Op, OpHash> front_set;
        std::unordered_set<Op, OpHash> executed_set;
        for (const auto &op_edge: graph.outEdges) {
            if (op_edge.first.ptr->tp == GateType::input_qubit) {
                front_set.insert(op_edge.first);
            }
        }
        // sabre loop 1
        while (!front_set.empty()) {
            // line 2 - 7, find executable gates
            std::vector<Op> executable_gate_list;
            for (const auto &front_gate: front_set) {
                // one qubit gate / input gate is always executable
                if (front_gate.ptr->tp == GateType::input_qubit || graph.inEdges[front_gate].size() == 1) {
                    executable_gate_list.emplace_back(front_gate);
                } else if (graph.inEdges[front_gate].size() == 2) {
                    Edge first_input = *graph.inEdges[front_gate].begin();
                    Edge second_input = *std::next(graph.inEdges[front_gate].begin());
                    int physical1 = logical2physical[first_input.logical_qubit_idx];
                    int physical2 = logical2physical[second_input.logical_qubit_idx];
                    std::vector<int> first_neighbour = device->get_input_neighbours(physical1);
                    for (int neighbour: first_neighbour) {
                        if (neighbour == physical2) {
                            executable_gate_list.emplace_back(front_gate);
                            break;
                        }
                    }
                } else {
                    // we do not support gates with more than 2 inputs in sabre
                    std::cout << "Find a gate with more than 2 inputs in sabre" << std::endl;
                    assert(false);
                }
            }
            // line 8 - 27
            if (!executable_gate_list.empty()) {
                // heuristic: reset decay to one if we found an executable gate
                for (double &decay: decay_list) { decay = 1.0; }
                // line 9 - 16
                for (const Op &gate: executable_gate_list) {
                    // line 10, execute
                    size_t _erase = front_set.erase(gate);
                    assert(_erase == 1);
                    executed_set.insert(gate);
                    // sabre swap: add gate to execution history
                    if (gate.ptr->tp != GateType::input_qubit) {
                        size_t qubit_num = graph.inEdges[gate].size();
                        assert(qubit_num == 1 || qubit_num == 2);
                        ExecutionHistory execution_history;
                        execution_history.guid = int(gate.guid);
                        execution_history.gate_type = gate.ptr->tp;
                        execution_history.logical0 = graph.inEdges[gate].begin()->logical_qubit_idx;
                        execution_history.physical0 = logical2physical[execution_history.logical0];
                        if (qubit_num == 2) {
                            execution_history.logical1 = std::next(graph.inEdges[gate].begin())->logical_qubit_idx;
                            execution_history.physical1 = logical2physical[execution_history.logical1];
                        }
                        if (graph.inEdges[gate].begin()->dstIdx == 1) {
                            // this means that the gate has two inputs and the order is reversed
                            std::swap(execution_history.logical0, execution_history.logical1);
                            std::swap(execution_history.physical0, execution_history.physical1);
                        }
                        execution_history_list.emplace_back(execution_history);
                    }
                    // apply swap to the mapping
                    if (gate.ptr->tp == GateType::swap) {
                        // need to update mapping if we have executed a swap
                        int first_logical = graph.inEdges[gate].begin()->logical_qubit_idx;
                        int second_logical = std::next(graph.inEdges[gate].begin())->logical_qubit_idx;
                        // swap
                        int ori_first_physical = logical2physical[first_logical];
                        int ori_second_physical = logical2physical[second_logical];
                        logical2physical[first_logical] = ori_second_physical;
                        logical2physical[second_logical] = ori_first_physical;
                        physical2logical[ori_first_physical] = second_logical;
                        physical2logical[ori_second_physical] = first_logical;
                    }
                    // line 11, obtain successor
                    if (graph.outEdges[gate].empty()) {
                        // this means that it is a final gate
                        continue;
                    }
                    std::vector<Op> successor_list;
                    for (const auto &edge: graph.outEdges[gate]) {
                        successor_list.emplace_back(edge.dstOp);
                    }
                    // line 12 - 14, check whether successor's dependency has been resolved
                    for (const auto &successor: successor_list) {
                        int resolved_count = 0;
                        for (const auto &successor_edge: graph.inEdges[successor]) {
                            if (executed_set.find(successor_edge.srcOp) != executed_set.end()) {
                                // found in executed
                                resolved_count += 1;
                            }
                        }
                        if (resolved_count == graph.inEdges[successor].size()) {
                            // resolved
                            front_set.emplace(successor);
                        }
                    }
                }
                continue;
            } else {
                // line 18 -19, obtain swaps
                std::vector<std::pair<int, int>> swap_candidate_list;
                for (const auto &gate: front_set) {
                    // all gates in F must be two qubit gates
                    auto edge_set = graph.inEdges[gate];
                    assert(edge_set.size() == 2);
                    // swaps are between input qubits and all physical neighbours
                    for (const auto &edge: edge_set) {
                        int logical_idx = edge.logical_qubit_idx;
                        int physical_idx = logical2physical[logical_idx];
                        auto neighbour_list = device->get_input_neighbours(physical_idx);
                        for (int neighbour: neighbour_list) {
                            swap_candidate_list.emplace_back(std::pair<int, int>{physical_idx, neighbour});
                        }
                    }
                }
                // heuristic: get extended set from front set
                std::unordered_set<Op, OpHash> extensive_set;
                for (const auto &gate: front_set) {
                    if (graph.outEdges[gate].empty()) {
                        // this means that the current gate in front set a final gate
                        continue;
                    }
                    for (const auto &edge: graph.outEdges[gate]) {
                        // we only consider two qubit gates in extensive set
                        if (graph.inEdges[edge.dstOp].size() == 2) {
                            extensive_set.emplace(edge.dstOp);
                        }
                    }
                }
                // line 20 - 24, find swap with minimal score
                double min_swap_cost = 10000000;
                std::pair<int, int> optimal_swap{-1, -1};
                for (const auto &swap: swap_candidate_list) {
                    // line 21, generate \pi_tmp
                    std::vector<int> tmp_logical2physical = logical2physical;
                    std::vector<int> tmp_physical2logical = physical2logical;
                    int physical_1 = swap.first;
                    int physical_2 = swap.second;
                    int logical_1 = tmp_physical2logical[physical_1];
                    int logical_2 = tmp_physical2logical[physical_2];
                    // swap physical
                    tmp_physical2logical[physical_1] = logical_2;
                    tmp_physical2logical[physical_2] = logical_1;
                    // swap logical, there must be one with logical qubit
                    assert(logical_1 != -1 || logical_2 != -1);
                    if (logical_1 != -1) tmp_logical2physical[logical_1] = physical_2;
                    if (logical_2 != -1) tmp_logical2physical[logical_2] = physical_1;
                    // line 22, calculate heuristic score
                    std::vector<std::pair<int, int>> front_mapping;
                    for (const auto &gate: front_set) {
                        int f_logical_1 = graph.inEdges[gate].begin()->logical_qubit_idx;
                        int f_logical_2 = std::next(graph.inEdges[gate].begin())->logical_qubit_idx;
                        int f_physical_1 = tmp_logical2physical[f_logical_1];
                        int f_physical_2 = tmp_logical2physical[f_logical_2];
                        front_mapping.emplace_back(std::pair<int, int>{f_physical_1, f_physical_2});
                    }
                    // heuristic: get extensive set mapping
                    std::vector<std::pair<int, int>> extensive_set_mapping;
                    for (const auto &gate: extensive_set) {
                        assert(graph.inEdges[gate].size() == 2);
                        int e_logical1 = graph.inEdges[gate].begin()->logical_qubit_idx;
                        int e_logical2 = std::next(graph.inEdges[gate].begin())->logical_qubit_idx;
                        int e_physical1 = tmp_logical2physical[e_logical1];
                        int e_physical2 = tmp_logical2physical[e_logical2];
                        extensive_set_mapping.emplace_back(std::pair<int, int>{e_physical1, e_physical2});
                    }
                    double cur_swap_score;
                    if (use_extensive) {
                        cur_swap_score = extended_sabre_heuristic(front_mapping, extensive_set_mapping,
                                                                  decay_list, swap, device, w_value);
                    } else {
                        cur_swap_score = basic_sabre_heuristic(front_mapping, device);
                    }
                    if (cur_swap_score <= min_swap_cost) {
                        min_swap_cost = cur_swap_score;
                        optimal_swap = swap;
                    }
                }
                // line 25, apply swap
                int physical_1 = optimal_swap.first;
                int physical_2 = optimal_swap.second;
                assert(physical_1 != -1 && physical_2 != -1);
                int logical_1 = physical2logical[physical_1];
                int logical_2 = physical2logical[physical_2];
                assert(logical_1 != -1 || logical_2 != -1);
                physical2logical[physical_1] = logical_2;
                physical2logical[physical_2] = logical_1;
                if (logical_1 != -1) logical2physical[logical_1] = physical_2;
                if (logical_2 != -1) logical2physical[logical_2] = physical_1;
                // heuristic: increase decay when applying swap
                decay_list[physical_1] += 0.001;
                decay_list[physical_2] += 0.001;
                // sabre swap: add swap to execution history
                ExecutionHistory execution_history;
                execution_history.guid = -1;
                execution_history.gate_type = GateType::swap;
                execution_history.logical0 = logical_1;
                execution_history.logical1 = logical_2;
                execution_history.physical0 = physical_1;
                execution_history.physical1 = physical_2;
                execution_history_list.emplace_back(execution_history);
            }
        }
        return execution_history_list;
    }

    std::vector<ExecutionHistory> sabre_swap(Graph initial_graph, const std::shared_ptr<DeviceTopologyGraph> &device,
                                             bool use_extensive, double w_value) {
        // applies sabre swap on an initialized circuit

        // STEP1: Generate initial, final qubit mapping table
        // Note that we suppose an initial mapping is already provided.
        // <logical, physical>
        QubitMappingTable initial_qubit_mapping = initial_graph.qubit_mapping_table;
        QubitMappingTable final_qubit_mapping;
        auto tmp_inEdges = initial_graph.inEdges;
        for (const auto &op_edge: tmp_inEdges) {
            if (initial_graph.outEdges.find(op_edge.first) == initial_graph.outEdges.end()) {
                // Case 1: the gate has no output
                // initialize output edges for this gate
                for (const auto &in_edge: op_edge.second) {
                    // generate final op and corresp. edge
                    Op final_op = Op(initial_graph.context->next_global_unique_id(),
                                     initial_graph.context->get_gate(GateType::input_qubit));
                    Edge edge_to_final = Edge(op_edge.first, final_op, in_edge.dstIdx, 0,
                                              in_edge.logical_qubit_idx, in_edge.physical_qubit_idx);
                    // put into graph's edge list
                    initial_graph.outEdges[op_edge.first].insert(edge_to_final);
                    initial_graph.inEdges[final_op].insert(edge_to_final);
                    // put into final qubit mapping
                    final_qubit_mapping.insert({final_op, std::pair<int, int>(edge_to_final.logical_qubit_idx,
                                                                              edge_to_final.physical_qubit_idx)});
                }
            } else if (initial_graph.outEdges[op_edge.first].size() < op_edge.second.size()) {
                // Case 2: the gate has fewer outputs than inputs
                for (const auto &in_edge: op_edge.second) {
                    // check whether this input edge has corresp. output
                    bool has_output = false;
                    for (const auto &out_edge: initial_graph.outEdges[op_edge.first]) {
                        if (out_edge.srcIdx == in_edge.dstIdx) has_output = true;
                    }
                    if (has_output) continue;
                    // generate final op and corresp. edge
                    Op final_op = Op(initial_graph.context->next_global_unique_id(),
                                     initial_graph.context->get_gate(GateType::input_qubit));
                    Edge edge_to_final = Edge(op_edge.first, final_op, in_edge.dstIdx, 0,
                                              in_edge.logical_qubit_idx, in_edge.physical_qubit_idx);
                    // put into graph's edge list
                    initial_graph.outEdges[op_edge.first].insert(edge_to_final);
                    initial_graph.inEdges[final_op].insert(edge_to_final);
                    // put into final qubit mapping
                    final_qubit_mapping.insert({final_op, std::pair<int, int>(edge_to_final.logical_qubit_idx,
                                                                              edge_to_final.physical_qubit_idx)});
                }
            }
        }

        // STEP2: SWAP-based heuristic search
        std::vector<ExecutionHistory> execution_history = _sabre_swap_loop(initial_graph, device,
                                                                           use_extensive, w_value);

        return execution_history;
    }

    int execution_cost(const std::vector<ExecutionHistory> &execution_history_list) {
        int cost = 0;
        for (const auto &execution_history: execution_history_list) {
            assert (execution_history.gate_type != GateType::input_qubit
                    && execution_history.gate_type != GateType::input_param);
            if (execution_history.gate_type == GateType::swap) {
                cost += SWAPCOST;
            } else {
                cost += 1;
            }
        }
        return cost;
    }

    enum class ExecutionHistoryStatus {
        VALID,
        UNINITIALIZED_EH,
        INVALID_EH
    };

    ExecutionHistoryStatus check_execution_history(const Graph &graph,
                                                   const std::shared_ptr<DeviceTopologyGraph> &device,
                                                   const std::vector<ExecutionHistory> &execution_history_list,
                                                   bool check_gate_count = true) {
        // check whether an execution history is valid on device
        int executed_gate_count = 0;
        for (const auto &execution_history: execution_history_list) {
            // check initialization
            if (execution_history.guid == -10 || execution_history.logical0 == -10 ||
                execution_history.physical0 == -10) {
                return ExecutionHistoryStatus::UNINITIALIZED_EH;
            }
            // check gate count
            if (!(execution_history.gate_type == GateType::swap && execution_history.guid == -1) &&
                !(execution_history.gate_type == GateType::swap && execution_history.guid == -2)) {
                executed_gate_count += 1;
            }
            // check whether the gate is valid on device
            if (execution_history.gate_type == GateType::swap && execution_history.guid == -2) {
                // virtual swaps in phase 1 of GameHybrid are always executable
                continue;
            } else {
                // execution of other swaps and logical gates
                int physical_0 = execution_history.physical0;
                int physical_1 = execution_history.physical1;
                if (physical_1 != -10) {
                    auto neighbours = device->get_input_neighbours(physical_0);
                    auto iterator = std::find(neighbours.begin(), neighbours.end(), physical_1);
                    if (iterator == neighbours.end()) {
                        return ExecutionHistoryStatus::INVALID_EH;
                    }
                }
            }
        }
        if (check_gate_count) assert(executed_gate_count == graph.gate_count());
        return ExecutionHistoryStatus::VALID;
    }

}
