#pragma once
#include "../tasograph/tasograph.h"

namespace quartz {
    using QubitMappingTable = std::unordered_map<Op, std::pair<int, int>, OpHash>;

    std::vector<int> calculate_sabre_mapping(Graph initial_graph) {
        // STEP1: Generate a trivial mapping and generate initial, final qubit mapping table
        // <logical, physical>
        initial_graph.init_physical_mapping(InitialMappingType::TRIVIAL);
        QubitMappingTable initial_qubit_mapping = initial_graph.qubit_mapping_table;
        QubitMappingTable final_qubit_mapping;
        for (const auto& op_edge : initial_graph.inEdges) {
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

        return {};
    }
}