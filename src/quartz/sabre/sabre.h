#pragma once
#include "../tasograph/tasograph.h"

namespace quartz {
    using QubitMappingTable = std::unordered_map<Op, std::pair<int, int>, OpHash>;

    double basic_sabre_heuristic(const std::vector<std::pair<int, int>>& front_set,
                                 const std::shared_ptr<DeviceTopologyGraph>& device) {
        double total_cost = 0;
        for (const auto& qubit_pair : front_set) {
            double min_swap_cost = device->unconnected_swap_penalty;
            auto input_neighbours = device->get_input_neighbours(qubit_pair.second);
            for (const auto& neighbour : input_neighbours) {
                double swap_cost = device->cal_swap_cost(qubit_pair.first, neighbour);
                min_swap_cost = std::min(min_swap_cost, swap_cost);
            }
            total_cost += min_swap_cost;
        }
        return total_cost;
    }

    std::vector<int> calculate_sabre_mapping(Graph initial_graph, const std::shared_ptr<DeviceTopologyGraph>& device) {
        // STEP1: Generate a trivial mapping and generate initial, final qubit mapping table
        // <logical, physical>
        initial_graph.init_physical_mapping(InitialMappingType::TRIVIAL);
        QubitMappingTable initial_qubit_mapping = initial_graph.qubit_mapping_table;
        QubitMappingTable final_qubit_mapping;
        auto tmp_inEdges = initial_graph.inEdges;
        for (const auto& op_edge : tmp_inEdges) {
            if (initial_graph.outEdges.find(op_edge.first) == initial_graph.outEdges.end()) {
                // Case 1: the gate has no output
                // initialize output edges for this gate
                for (const auto& in_edge : op_edge.second) {
                    // generate final op and corresp. edge
                    Op final_op = Op(initial_graph.context->next_global_unique_id(),
                                      initial_graph.context->get_gate(GateType::input_qubit));
                    Edge edge_to_final = Edge(op_edge.first, final_op, in_edge.dstIdx, 0,
                                              in_edge.logical_qubit_idx, in_edge.physical_qubit_idx);
                    // put into graph's edge list
                    initial_graph.outEdges[op_edge.first].insert(edge_to_final);
                    initial_graph.inEdges[final_op].insert(edge_to_final);
                    // put into final qubit mapping
                    final_qubit_mapping.insert({final_op,std::pair<int, int>(edge_to_final.logical_qubit_idx,
                                                                             edge_to_final.physical_qubit_idx)});
                }
            } else if (initial_graph.outEdges[op_edge.first].size() < op_edge.second.size()) {
                // Case 2: the gate has fewer outputs than inputs
                for (const auto& in_edge : op_edge.second) {
                    // check whether this input edge has corresp. output
                    bool has_output = false;
                    for (const auto& out_edge : initial_graph.outEdges[op_edge.first]) {
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
                    final_qubit_mapping.insert({final_op,std::pair<int, int>(edge_to_final.logical_qubit_idx,
                                                                             edge_to_final.physical_qubit_idx)});
                }
            }
        }

        // STEP2: SWAP-based heuristic search
        std::vector<int> logical2physical;
        std::vector<int> physical2logical;
        logical2physical.reserve(initial_qubit_mapping.size());
        physical2logical.reserve(device->get_num_qubits());
        for (int i = 0; i < device->get_num_qubits(); i++) {
            physical2logical.emplace_back(-1);
        }
        for (int i = 0; i < initial_qubit_mapping.size(); i++) {
            physical2logical[i] = i;
            logical2physical.emplace_back(i);
        }

        // STEP3: reverse the graph

        // STEP4: SWAP-based heuristic search

        // return final mapping
        return {};
    }
}
