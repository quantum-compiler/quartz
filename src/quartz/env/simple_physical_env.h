#pragma once

#include "basic_env.h"
#include "../parser/qasm_parser.h"
#include "../supported_devices/supported_devices.h"

namespace quartz {
    class SimplePhysicalEnv : public BasicEnv {
    public:
        SimplePhysicalEnv() = delete;

        SimplePhysicalEnv(const std::string &qasm_file_path, BackendType backend_type) {
            /// A simple physical environment that does not use curriculum
            /// Graph    qasm            fixed
            ///          mapping type    1 x sabre with random init
            /// Device   topology        fixed
            // initialize context, graph and device
            context = std::make_shared<Context>(Context({GateType::h, GateType::cx, GateType::t,
                                                         GateType::tdg, GateType::input_qubit}));
            QASMParser qasm_parser(&(*context));
            DAG *dag = nullptr;
            if (!qasm_parser.load_qasm(qasm_file_path, dag)) {
                std::cout << "Parser failed" << std::endl;
                assert(false);
            }
            graph = std::make_shared<Graph>(Graph(&(*context), dag));
            device = GetDevice(backend_type);

            // initialize mapping for graph and create game
            graph->init_physical_mapping(InitialMappingType::SABRE, device, 4, true, 0.5);
            assert(graph->check_mapping_correctness() == MappingStatus::VALID);
            cur_game_ptr = std::make_shared<Game>(Game(*graph, device));
        }

        void reset() override {
            // re-initialize mapping and game
            graph->init_physical_mapping(InitialMappingType::SABRE, device, 3, true, 0.5);
            assert(graph->check_mapping_correctness() == MappingStatus::VALID);
            cur_game_ptr = std::make_shared<Game>(Game(*graph, device));
        }

        Reward step(Action action) override {
            // check whether action is valid
            assert(action.type == ActionType::PhysicalFront);
            assert(action.qubit_idx_0 < cur_game_ptr->physical_qubit_num);
            assert(action.qubit_idx_1 < cur_game_ptr->physical_qubit_num);
            // apply action
            Reward reward = cur_game_ptr->apply_action(action);
            return reward;
        }

        bool is_finished() override {
            return is_circuit_finished(cur_game_ptr->graph);
        }

        State get_state() override {
            // TODO: implement this
            return {};
        }

        std::vector<Action> get_action_space() override {
            // ActionType: Physical Front
            std::set<Action, ActionCompare> available_action_set = cur_game_ptr->action_space(
                    ActionType::PhysicalFront);
            std::vector<Action> action_space;
            action_space.reserve(available_action_set.size());
            for (const auto &available_action: available_action_set) {
                action_space.emplace_back(available_action);
            }
            return std::move(action_space);
        }

    public:
        std::shared_ptr<DeviceTopologyGraph> device;
        std::shared_ptr<Context> context;
        std::shared_ptr<Graph> graph;
    };
}
