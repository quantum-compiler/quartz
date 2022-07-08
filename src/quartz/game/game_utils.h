#pragma once

#include <iostream>
#include "../tasograph/tasograph.h"

namespace quartz {
    std::ostream &operator<<(std::ostream &stream, GateType t) {
        const std::string name_list[] = {"h", "x", "y", "rx", "ry", "rz", "cx", "ccx", "add",
                                         "neg", "z", "s", "sdg", "t", "tdg", "ch", "swap", "p",
                                         "pdg", "rx1", "rx3", "u1", "u2", "u3", "ccz", "cz",
                                         "input_qubit", "input_param"
        };
        return stream << name_list[int(t)];
    }

    void execute_gate(Graph &graph, Op op) {
        /// execute an op in a graph, the op must have all inputs as input qubit gate
        /// Note that this function assumes that the graph has already been initialized
        /// by SabreSwap (calculate_sabre_mapping) so that each non-input op in the graph
        /// has #in_deg == #out_deg

        // check if the op is a valid op to execute
        if (op.ptr->tp == GateType::swap || op.ptr->tp == GateType::input_qubit) {
            std::cout << "Try to execute invalid gate with type " << op.ptr->tp << std::endl;
            assert(false);
        }
        // check if the op belongs to the graph
        if (graph.inEdges.find(op) == graph.inEdges.end() ||
            graph.outEdges.find(op) == graph.outEdges.end()) {
            std::cout << "Gate not in circuit." << std::endl;
            assert(false);
        }
        // check if all inputs are input qubit gates
        auto op_in_edge_set = graph.inEdges[op];
        auto op_out_edge_set = graph.outEdges[op];
        for (const auto &edge: op_in_edge_set) {
            if (edge.srcOp.ptr->tp != GateType::input_qubit) {
                std::cout << "Current gate is not in DAG first layer." << std::endl;
                assert(false);
            }
        }

        // change edge connection
        for (const auto &in_edge: op_in_edge_set) {
            for (const auto &out_edge: op_out_edge_set) {
                if (in_edge.dstIdx == out_edge.srcIdx) {
                    // this means that the two edges should merge into one
                    auto previous_op = in_edge.srcOp;
                    auto previous_port = in_edge.srcIdx;
                    auto successive_op = out_edge.dstOp;
                    auto successive_port = out_edge.dstIdx;
                    // change out edge of previous op
                    for (auto &prev_out_edge : graph.outEdges[previous_op]) {
                        if (prev_out_edge.srcIdx == in_edge.srcIdx) {
                            prev_out_edge.dstOp = successive_op;
                            prev_out_edge.dstIdx = successive_port;
                        }
                    }
                    // change in edge of successive op
                    for (auto &suc_in_edge : graph.inEdges[successive_op]) {
                        if (suc_in_edge.dstIdx == out_edge.dstIdx) {
                            suc_in_edge.srcOp = previous_op;
                            suc_in_edge.srcIdx = previous_port;
                        }
                    }
                }
            }
        }
        // delete gate from circuit
        graph.inEdges.erase(op);
        graph.outEdges.erase(op);
    }
}