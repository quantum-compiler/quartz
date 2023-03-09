#pragma once

#include <string>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <cassert>

#include "../utils/physical_mapping_utils.h"

namespace quartz {
    class FidelityGraph {
    public:
        FidelityGraph() : num_regs(0) {};

        explicit FidelityGraph(int _num_regs) : num_regs(_num_regs) {}

        FidelityGraph(const FidelityGraph &old_graph) {
            num_regs = old_graph.num_regs;
            cx_error_rate_map = old_graph.cx_error_rate_map;
        }

        void insert_cx_error_rate(int reg0, int reg1, double error_rate) {
            // insert cx error rate between reg0 and reg1
            // some checks on input parameters
            Assert(reg0 != reg1, "reg 0 should not be equal to reg1!");
            Assert(0 <= reg0 && reg0 < num_regs && 0 <= reg1 && reg1 < num_regs,
                   "reg 0 & reg 1 should be in range 0 - #regs - 1!");
            Assert(0 <= error_rate && error_rate <= 1, "error rate should be in [0, 1]!");

            // insert the error_rate message into the map
            std::string f_edge_name = std::to_string(reg0) + "->" + std::to_string(reg1);
            Assert(cx_error_rate_map.find(f_edge_name) == cx_error_rate_map.end(), "Duplicate edge!");
            cx_error_rate_map[f_edge_name] = error_rate;
        }

        double query_cx_fidelity(int reg0, int reg1) {
            // returns the log fidelity (base e) of cx between reg 0 and reg 1, i.e. returns ln(1 - error)
            // when we calculate circuit fidelity, we sum ln(1 - error) and then exp it
            std::string f_edge_name = std::to_string(reg0) + "->" + std::to_string(reg1);
            Assert(cx_error_rate_map.find(f_edge_name) != cx_error_rate_map.end(), "Edge not found!");
            double error_rate = cx_error_rate_map[f_edge_name];
            return log(1 - error_rate);
        }

    public:
        int num_regs;
        std::unordered_map<std::string, double> cx_error_rate_map;
    };
}
