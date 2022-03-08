#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <memory>

namespace quartz {
    struct DeviceEdge {
    public:
        DeviceEdge() = default;

        DeviceEdge(int _src, int _dst, double _swap_cost) : src_idx(_src), dst_idx(_dst), swap_cost(_swap_cost) {}

        void print() const {
            std::cout << "src: " << src_idx << ", dst: " << dst_idx << ", swap cost: " << swap_cost << '\n';
        }

    public:
        int src_idx, dst_idx;
        double swap_cost;
    };

    struct DeviceQubit {
    public:
        DeviceQubit() = default;

        explicit DeviceQubit(int _idx) : idx(_idx) {}

        void print() const {
            std::cout << "qubit idx: " << idx << '\n';
            std::cout << "out edges: " << '\n';
            for (const auto &out_edge: out_edges) out_edge->print();
            std::cout << "in edges: " << '\n';
            for (const auto &in_edge: in_edges) in_edge->print();
            std::cout << '\n';
        }

    public:
        int idx{-1};
        std::vector<std::shared_ptr<DeviceEdge>> out_edges;
        std::vector<std::shared_ptr<DeviceEdge>> in_edges;
    };

    class DeviceTopologyGraph {
    public:
        DeviceTopologyGraph() = default;

        explicit DeviceTopologyGraph(int _num_qubits) : num_qubits(_num_qubits) {
            // initialize qubits
            for (auto idx = 0; idx < _num_qubits; ++idx) {
                device_qubits.emplace_back(std::make_shared<DeviceQubit>(idx));
            }
        }

        void print() const {
            for (const auto &device_qubit: device_qubits) device_qubit->print();
        }

        void add_qubit() {
            device_qubits.emplace_back(std::make_shared<DeviceQubit>(num_qubits++));
        }

        void add_edge(int src_idx, int dst_idx, double swap_cost, bool bidirectional = true) {
            auto new_edge = std::make_shared<DeviceEdge>(src_idx, dst_idx, swap_cost);
            device_qubits[src_idx]->out_edges.emplace_back(new_edge);
            device_qubits[dst_idx]->in_edges.emplace_back(new_edge);
            if (bidirectional) {
                auto reverse_edge = std::make_shared<DeviceEdge>(dst_idx, src_idx, swap_cost);
                device_qubits[dst_idx]->out_edges.emplace_back(reverse_edge);
                device_qubits[src_idx]->in_edges.emplace_back(reverse_edge);
            }
        }

        double cal_swap_cost(int src_idx, int dst_idx) const {
            // implemented using dijkstra with heap optimization
            // time complexity is nlog(n)

            // a query structure used locally
            struct Query {
                Query() = delete;

                Query(int _target_idx, double _cost) : target_idx(_target_idx), cost(_cost) {}

                int target_idx;
                double cost;
            };
            struct QueryCompare {
                bool operator()(const Query &a, const Query &b) const {
                    return a.cost > b.cost;
                };
            };

            // initialize
            auto swap_cost = std::vector<double>();
            auto closed = std::vector<bool>();
            auto search_queue = std::priority_queue<Query, std::vector<Query>, QueryCompare>();
            for (int i = 0; i < num_qubits; i++) {
                swap_cost.emplace_back(unconnected_swap_penalty);
                closed.emplace_back(false);
            }
            search_queue.emplace(Query(src_idx, 0));

            // shortest path
            while (!search_queue.empty()) {
                // get query from priority queue
                auto query = search_queue.top();
                search_queue.pop();

                // update distance
                if (closed[query.target_idx]) continue;
                closed[query.target_idx] = true;
                swap_cost[query.target_idx] = query.cost;

                // update other distances
                for (const auto &out_edge: device_qubits[query.target_idx]->out_edges)
                    if (!closed[out_edge->dst_idx] && query.cost + out_edge->swap_cost < swap_cost[out_edge->dst_idx]) {
                        swap_cost[out_edge->dst_idx] = query.cost + out_edge->swap_cost;
                        search_queue.emplace(Query(out_edge->dst_idx, swap_cost[out_edge->dst_idx]));
                    }
            }

            // return
            return swap_cost[dst_idx];
        }

    private:
        int num_qubits{0};
        constexpr static double unconnected_swap_penalty = 10000;
        std::vector<std::shared_ptr<DeviceQubit>> device_qubits;
    };
}

