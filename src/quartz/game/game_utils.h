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
                    for (auto &prev_out_edge: graph.outEdges[previous_op]) {
                        if (prev_out_edge.srcIdx == in_edge.srcIdx) {
                            prev_out_edge.dstOp = successive_op;
                            prev_out_edge.dstIdx = successive_port;
                        }
                    }
                    // change in edge of successive op
                    for (auto &suc_in_edge: graph.inEdges[successive_op]) {
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

    std::set<Op, OpCompare> find_executable_front_gates(Graph &graph,
                                                        const std::shared_ptr<DeviceTopologyGraph> &device) {
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

        // check whether they are executable on device
        std::set<Op, OpCompare> executable_gate_set;
        for (auto &front_op: front_gate_set) {
            if (graph.inEdges[front_op].size() == 1) {
                // single qubit gate is always executable
                executable_gate_set.insert(front_op);
            } else if (graph.inEdges[front_op].size() == 2) {
                // check two qubit gates
                // get input physical idx
                Edge in_edge_0 = *(graph.inEdges[front_op].begin());
                Edge in_edge_1 = *(std::next(graph.inEdges[front_op].begin()));
                int physical_idx_0 = graph.qubit_mapping_table[in_edge_0.srcOp].second;
                int physical_idx_1 = graph.qubit_mapping_table[in_edge_1.srcOp].second;
                assert(physical_idx_0 == in_edge_0.physical_qubit_idx);
                assert(physical_idx_1 == in_edge_1.physical_qubit_idx);
                // check whether they are neighbors
                auto neighbor_list = device->get_input_neighbours(physical_idx_1);
                bool is_neighbor = false;
                for (int neighbor_qubit_idx: neighbor_list) {
                    if (physical_idx_0 == neighbor_qubit_idx) is_neighbor = true;
                }
                if (is_neighbor) {
                    executable_gate_set.insert(front_op);
                }
            } else {
                // we do not support gates with more than 3 inputs
                std::cout << "Found gate with >= 2 inputs" << std::endl;
                assert(false);
            }
        }

        return executable_gate_set;
    }

    bool is_circuit_finished(Graph &graph) {
        return (graph.inEdges.empty() && graph.outEdges.empty() && graph.qubit_mapping_table.empty());
    }

}