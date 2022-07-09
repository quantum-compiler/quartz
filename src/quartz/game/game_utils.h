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

    void execute_front_gate(Graph &graph, Op op) {
        /// execute an op in a graph, the op must have all inputs as input qubit gate

        // check if the op is a valid op to execute
        if (op.ptr->tp == GateType::swap || op.ptr->tp == GateType::input_qubit) {
            std::cout << "Try to execute invalid gate with type " << op.ptr->tp << std::endl;
            assert(false);
        }
        // check if the op belongs to the graph
        if (graph.inEdges.find(op) == graph.inEdges.end()) {
            std::cout << "Gate not in circuit." << std::endl;
            assert(false);
        }
        // check if all inputs are input qubit gates
        auto op_in_edge_set = graph.inEdges[op];
        std::set<Edge, EdgeCompare> op_out_edge_set;
        if (graph.outEdges.find(op) != graph.outEdges.end()) {
            op_out_edge_set = graph.outEdges[op];
        }
        for (const auto &edge: op_in_edge_set) {
            if (edge.srcOp.ptr->tp != GateType::input_qubit) {
                std::cout << "Current gate is not in DAG first layer." << std::endl;
                assert(false);
            }
        }

        // change edge connection
        for (const auto &in_edge: op_in_edge_set) {
            bool found_out_edge = false;
            for (const auto &out_edge: op_out_edge_set) {
                if (in_edge.dstIdx == out_edge.srcIdx) {
                    found_out_edge = true;
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
            // if we can not find corresponding out edge, it means that
            // the op is a final op on some qubits, and we need to delete
            // corresponding out edge from previous op
            if (!found_out_edge) {
                // remove edge
                auto previous_op = in_edge.srcOp;
                graph.outEdges[previous_op].erase(in_edge);
                // if this makes outEdges empty, remove it too
                if (graph.outEdges[previous_op].empty()) {
                    graph.outEdges.erase(previous_op);
                    // if this finishes all gates on a qubit, erase it from
                    // qubit mapping table
                    if (previous_op.ptr->tp == GateType::input_qubit) {
                        graph.qubit_mapping_table.erase(previous_op);
                    }
                }
            }
        }
        // delete gate from circuit
        graph.inEdges.erase(op);
        graph.outEdges.erase(op);
    }

    bool is_circuit_finished(Graph &graph) {
        return (graph.inEdges.empty() && graph.outEdges.empty() && graph.qubit_mapping_table.empty());
    }

}