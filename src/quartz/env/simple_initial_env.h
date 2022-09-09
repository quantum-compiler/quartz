#pragma once

#include "../game/game_initial.h"
#include "../parser/qasm_parser.h"
#include "../supported_devices/supported_devices.h"

namespace quartz {
    class SimpleInitialEnv {
    public:
        SimpleInitialEnv() = delete;

        SimpleInitialEnv(const std::string &qasm_file_path, BackendType backend_type) {
            /// A simple initial mapping environment that does not use curriculum
            /// This is only a partial environment, reward calculation is in Python side (using Qiskit)
            /// Graph    qasm            fixed
            ///          mapping type    64 x sabre with random init
            /// Device   topology        fixed
            // initialize context, graph and device
            context = std::make_shared<Context>(Context({GateType::h, GateType::cx, GateType::t,
                                                         GateType::tdg, GateType::input_qubit, GateType::s,
                                                         GateType::sdg}));
            QASMParser qasm_parser(&(*context));
            DAG *dag = nullptr;
            if (!qasm_parser.load_qasm(qasm_file_path, dag)) {
                std::cout << "Parser failed" << std::endl;
                assert(false);
            }
            graph = std::make_shared<Graph>(Graph(&(*context), dag));
            device = GetDevice(backend_type);

            // initialize mapping for graph and create game
            find_initial_mapping(*graph, device, 8);
            assert(graph->check_mapping_correctness() == MappingStatus::VALID);
            cur_game_ptr = std::make_shared<GameInitial>(GameInitial(*graph, device));
        }

        void reset() {
            // re-initialize mapping and game
            find_initial_mapping(*graph, device, 8);
            assert(graph->check_mapping_correctness() == MappingStatus::VALID);
            cur_game_ptr = std::make_shared<GameInitial>(GameInitial(*graph, device));
        }

        [[nodiscard]] Reward step(Action action) const {
            // check whether action is valid
            assert(action.type == ActionType::PhysicalFull);
            assert(action.qubit_idx_0 < cur_game_ptr->physical_qubit_num);
            assert(action.qubit_idx_1 < cur_game_ptr->physical_qubit_num);
            // apply action
            Reward reward = cur_game_ptr->apply_action(action);
            return reward;
        }

        [[nodiscard]] State get_state() const {
            return cur_game_ptr->state();
        }

        [[nodiscard]] std::vector<Action> get_action_space() const {
            // ActionType: Physical Front
            std::set<Action, ActionCompare> available_action_set = cur_game_ptr->action_space(
                    ActionType::PhysicalFull);
            std::vector<Action> action_space;
            action_space.reserve(available_action_set.size());
            for (const auto &available_action: available_action_set) {
                action_space.emplace_back(available_action);
            }
            return std::move(action_space);
        }

    public:
        std::shared_ptr<GameInitial> cur_game_ptr;
        std::shared_ptr<DeviceTopologyGraph> device;
        std::shared_ptr<Context> context;
        std::shared_ptr<Graph> graph;
    };
}
