#pragma once

#include <iostream>
#include <utility>
#include "../tasograph/tasograph.h"
#include "../sabre/sabre_swap.h"
#include "../utils/physical_mapping_utils.h"

namespace quartz {
    class State {
    public:
        State() = delete;

        State(std::vector<std::pair<int, int>> _device_edges,
              std::vector<int> _logical2physical,
              std::vector<int> _physical2logical,
              GraphState _graph_state,
              bool _is_initial_phase = false) : device_edges(std::move(_device_edges)),
                                                logical2physical(std::move(_logical2physical)),
                                                physical2logical(std::move(_physical2logical)),
                                                graph_state(std::move(_graph_state)),
                                                is_initial_phase(_is_initial_phase) {}

    public:
        std::vector<std::pair<int, int>> device_edges;
        std::vector<int> logical2physical;
        std::vector<int> physical2logical;
        GraphState graph_state;
        bool is_initial_phase;
    };

    enum class ActionType {
        PhysicalFull = 0,   // swaps between physical neighbors of all used logical qubits
        PhysicalFront = 1,  // swaps between physical neighbors of inputs to front gates
        Logical = 2,        // swaps between logical qubits, at least one must be used
        SearchFull = 3,     // swaps between one used physical qubit and any other physical qubits
        Unknown = 4
    };

    class Action {
    public:
        Action() : type(ActionType::Unknown), qubit_idx_0(-1), qubit_idx_1(-1) {}

        Action(ActionType _type, int _qubit_idx_0, int _qubit_idx_1) : type(_type), qubit_idx_0(_qubit_idx_0),
                                                                       qubit_idx_1(_qubit_idx_1) {}

    public:
        ActionType type;
        int qubit_idx_0;
        int qubit_idx_1;
    };

    struct ActionCompare {
        bool operator()(const Action &a, const Action &b) const {
            if (a.type != b.type) return a.type < b.type;
            if (a.qubit_idx_0 != b.qubit_idx_0) return a.qubit_idx_0 < b.qubit_idx_0;
            return a.qubit_idx_1 < b.qubit_idx_1;
        };
    };

    using Reward = double;

    std::ostream &operator<<(std::ostream &stream, GateType t) {
        const std::string name_list[] = {"h", "x", "y", "rx", "ry", "rz", "cx", "ccx", "add",
                                         "neg", "z", "s", "sdg", "t", "tdg", "ch", "swap", "p",
                                         "pdg", "rx1", "rx3", "u1", "u2", "u3", "ccz", "cz",
                                         "input_qubit", "input_param", "sx"
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
                auto edge_to_erase = graph.outEdges[previous_op].end();
                for (auto it = graph.outEdges[previous_op].cbegin(); it != graph.outEdges[previous_op].cend(); ++it) {
                    if (it->srcOp.guid == in_edge.srcOp.guid && it->dstOp.guid == in_edge.dstOp.guid
                        && it->srcIdx == in_edge.srcIdx && it->dstIdx == in_edge.dstIdx) {
                        edge_to_erase = it;
                    }
                }
                graph.outEdges[previous_op].erase(edge_to_erase);
                // if this makes outEdges empty, remove it too
                if (graph.outEdges[previous_op].empty()) {
                    size_t _erase = graph.outEdges.erase(previous_op);
                    assert(_erase == 1);
                    // if this finishes all gates on a qubit, erase it from
                    // qubit mapping table
                    if (previous_op.ptr->tp == GateType::input_qubit) {
                        size_t _erase_2 = graph.qubit_mapping_table.erase(previous_op);
                        assert(_erase_2 == 1);
                    }
                }
            }
        }
        // delete gate from circuit
        size_t _erase = graph.inEdges.erase(op);
        assert(_erase == 1);
        graph.outEdges.erase(op);
    }

    std::set<Op, OpCompare> find_executable_front_gates(Graph &graph,
                                                        const std::shared_ptr<DeviceTopologyGraph> &device) {
        /// This functions assumes that a valid mapping has been given to the input graph.
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

    std::set<Op, OpCompare> find_front_gates(Graph &graph) {
        /// This functions assumes that a valid mapping has been given to the input graph.
        // get gates with at least one input input_qubit
        std::set<Op, OpCompare> tmp_front_gate_set;
        for (const auto &initial_qubit_mapping: graph.qubit_mapping_table) {
            auto initial_qubit = initial_qubit_mapping.first;
            assert(graph.outEdges.find(initial_qubit) != graph.outEdges.end()
                   && graph.outEdges[initial_qubit].size() == 1);
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

        return std::move(front_gate_set);
    }

    bool is_circuit_finished(const Graph &graph) {
        return (graph.inEdges.empty() && graph.outEdges.empty() && graph.qubit_mapping_table.empty());
    }

    void remove_gate(Graph &graph, Op op) {
        /// remove a gate from given circuit
        /// This is identical to execute_front_gate but does not check input.

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
            assert(!op_out_edge_set.empty());
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
                auto edge_to_erase = graph.outEdges[previous_op].end();
                for (auto it = graph.outEdges[previous_op].cbegin(); it != graph.outEdges[previous_op].cend(); ++it) {
                    if (it->srcOp.guid == in_edge.srcOp.guid && it->dstOp.guid == in_edge.dstOp.guid
                        && it->srcIdx == in_edge.srcIdx && it->dstIdx == in_edge.dstIdx) {
                        edge_to_erase = it;
                    }
                }
                graph.outEdges[previous_op].erase(edge_to_erase);
                // if this makes outEdges empty, remove it too
                if (graph.outEdges[previous_op].empty()) {
                    size_t _erase = graph.outEdges.erase(previous_op);
                    assert(_erase == 1);
                    // if this finishes all gates on a qubit, erase it from
                    // qubit mapping table
                    if (previous_op.ptr->tp == GateType::input_qubit) {
                        size_t _erase2 = graph.qubit_mapping_table.erase(previous_op);
                        assert(_erase2 == 1);
                    }
                }
            }
        }
        // delete gate from circuit
        size_t _erase = graph.inEdges.erase(op);
        assert(_erase == 1);
        graph.outEdges.erase(op);
    }

    [[nodiscard]] int simplify_circuit(Graph &graph) {
        /// This function simplifies given graph by removing all single qubit gates.
        /// return: number of single qubit gates removed
        auto tmp_in_edges = graph.inEdges;  // we need a copy here since delete invalidates iterators
        int removed_gate_cnt = 0;
        for (const auto &op_pair: tmp_in_edges) {
            if (op_pair.second.size() == 1) {
                // STEP 1: collapse this gate forward and record it (and the gates it carries) in
                // simplified_gates_after_op (this is necessary for final restoration of the circuit)
                Op cur_op = op_pair.first;
                Op op_before = graph.inEdges[cur_op].begin()->srcOp;  // must use this, since only this maintains the correct graph structure!

                // put current gate into the list
                if (graph.simplified_gates_after_op.find(op_before) == graph.simplified_gates_after_op.end())
                    graph.simplified_gates_after_op[op_before] = std::deque<OutputGateRepresentation>();
                graph.simplified_gates_after_op[op_before].emplace_back(true, cur_op.ptr->tp,
                                                                        graph.inEdges[cur_op].begin()->logical_qubit_idx, -1,
                                                                        cur_op.parameter_string);

                // put gates that the current gate carries into the list (and remove corresponding entry
                // from simplified_gates_after_op)
                if (graph.simplified_gates_after_op.find(cur_op) != graph.simplified_gates_after_op.end()) {
                    // move the contents
                    std::deque<OutputGateRepresentation> &cur_op_gate_queue = graph.simplified_gates_after_op[cur_op];
                    std::deque<OutputGateRepresentation> &op_before_gate_queue = graph.simplified_gates_after_op[op_before];
                    while (!cur_op_gate_queue.empty()) {
                        op_before_gate_queue.emplace_back(cur_op_gate_queue.front());
                        cur_op_gate_queue.pop_front();
                    }

                    // delete the entry
                    size_t removed_cnt = graph.simplified_gates_after_op.erase(cur_op);
                    Assert(removed_cnt == 1, "Remove from simplified_gates_after_op failed!");
                }

                // STEP 2: remove the gate
                removed_gate_cnt += 1;
                remove_gate(graph, op_pair.first);
            }
        }

        // some final integrity tests
        size_t _gates_simplified = 0;
        for (const auto& op_dequeue_pair : graph.simplified_gates_after_op) {
            _gates_simplified += op_dequeue_pair.second.size();
            Assert(op_dequeue_pair.first.ptr->tp == GateType::input_qubit || op_dequeue_pair.first.ptr->num_qubits == 2,
                   "Gates remain in simplified_gates_after_op must be input qubits / two-qubit gates!");
        }
        Assert(_gates_simplified == removed_gate_cnt, "Number of gates removed mismatch!");

        return removed_gate_cnt;
    }

    template<typename Iter, typename RandomGenerator>
    Iter select_randomly(Iter start, Iter end, RandomGenerator &g) {
        std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
        std::advance(start, dis(g));
        return start;
    }

    template<typename Iter>
    Iter select_randomly(Iter start, Iter end, int seed) {
        static std::mt19937 gen(seed);
        return select_randomly(start, end, gen);
    }

    void find_initial_mapping(Graph &graph, const std::shared_ptr<DeviceTopologyGraph> &device,
                              int quota) {
        // STEP 1: find a good initial mapping
        QubitMappingTable best_mapping;
        int best_mapping_cost = 1000000;
        for (int i = 0; i < quota; ++i) {
            // initialize mapping
            graph.init_physical_mapping(InitialMappingType::SABRE, device, 5, true, 0.5);
            auto execution_history = sabre_swap(graph, device, true, 0.5);
            int current_cost = execution_cost(execution_history);
            // update best
            if (current_cost < best_mapping_cost) {
                best_mapping = graph.qubit_mapping_table;
                best_mapping_cost = current_cost;
            }
        }

        // STEP 2: propagate the mapping
        graph.qubit_mapping_table = best_mapping;
        graph.propagate_mapping();
    }

    std::vector<std::vector<int>> read_logical2physical_file(const std::string &file_name, int reg_count) {
        // initialize file
        std::ifstream input_file;
        input_file.open(file_name);

        // read file, each line contains one logical -> physical mapping
        std::vector<std::vector<int>> logical2physical_list;
        std::string raw_line;
        while (std::getline(input_file, raw_line)) {
            // for each line, we split it into an array of numbers
            std::istringstream cur_line(raw_line);
            std::vector<int> cur_logical2physical;
            int tmp_int, num = 0;
            while (cur_line >> tmp_int) {
                cur_logical2physical.emplace_back(tmp_int);
                num++;
            }

            // check mapping validity and save to
            Assert(num == reg_count, "Incompatible Mapping!");
            logical2physical_list.emplace_back(std::move(cur_logical2physical));
        }

        // clean up and return
        input_file.close();
        return std::move(logical2physical_list);
    }

    void set_initial_mapping(Graph &graph, int random_number, const std::string &file_name, int reg_count) {
        // read candidate logical2physical mappings from disk
        std::vector<std::vector<int>> logical2physical = read_logical2physical_file(file_name, reg_count);

        // choose one randomly and set as mapping
        std::vector<int> chosen_mapping = *select_randomly(logical2physical.begin(), logical2physical.end(),
                                                           random_number);
        graph.set_physical_mapping(chosen_mapping);
    }

    void set_search_initial_mapping(Graph &graph, int random_number, const std::string &file_name, int reg_count) {
        // read candidate logical2physical mappings from disk
        std::vector<std::vector<int>> logical2physical = read_logical2physical_file(file_name, reg_count);

        // choose one randomly and set as mapping
        std::vector<int> chosen_mapping = *select_randomly(logical2physical.begin(), logical2physical.end(),
                                                           random_number);
        graph.set_physical_mapping(chosen_mapping);
    }

    std::vector<size_t> find_all_occurrences(const std::string &str, char target) {
        // find all occurrences of target in str
        std::vector<size_t> positions;
        size_t pos = str.find(target, 0);
        while (pos != std::string::npos) {
            positions.push_back(pos);
            pos = str.find(target, pos + 1);
        }
        return std::move(positions);
    }
}