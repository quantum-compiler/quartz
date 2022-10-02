#pragma once

#include "basic_env.h"
#include "../parser/qasm_parser.h"
#include "../supported_devices/supported_devices.h"
#include "../game/game_buffer.h"

namespace quartz {
    class SimplePhysicalEnv : public BasicEnv {
    public:
        SimplePhysicalEnv() = delete;

        SimplePhysicalEnv(const std::string &qasm_file_path, BackendType backend_type,
                          int _seed, double _start_from_internal_prob) : game_buffer(_seed), random_generator(_seed) {
            /// A simple physical environment that does not use curriculum
            /// Graph    qasm            fixed
            ///          mapping type    1 x sabre with random init
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

            // set randomness (random_generator is initialized above)
            seed = _seed;
            uniform01dist = std::uniform_real_distribution<double>(0.0, 1.0);

            // initialize mapping for graph and create game
            set_initial_mapping(*graph);
            assert(graph->check_mapping_correctness() == MappingStatus::VALID);
            cur_game_ptr = std::make_shared<Game>(Game(*graph, device));
            while (is_circuit_finished(cur_game_ptr->graph)) {
                set_initial_mapping(*graph);
                assert(graph->check_mapping_correctness() == MappingStatus::VALID);
                cur_game_ptr = std::make_shared<Game>(Game(*graph, device));
            }

            // initialize game buffer (game_buffer is initialize above)
            start_from_internal_prob = _start_from_internal_prob;
            game_buffer.save(*cur_game_ptr);
        }

        void reset() override {
            // generate a random number
            double _random_val = uniform01dist(random_generator);

            // set new game
            if (_random_val < start_from_internal_prob) {
                // start from internal states
                Game sampled_game = game_buffer.sample();
                cur_game_ptr = std::make_shared<Game>(sampled_game);
            } else {
                // start from initial states
                set_initial_mapping(*graph);
                assert(graph->check_mapping_correctness() == MappingStatus::VALID);
                cur_game_ptr = std::make_shared<Game>(Game(*graph, device));
                while (is_finished()) {
                    set_initial_mapping(*graph);
                    assert(graph->check_mapping_correctness() == MappingStatus::VALID);
                    cur_game_ptr = std::make_shared<Game>(Game(*graph, device));
                }
            }
        }

        Reward step(Action action) override {
            // check whether action is valid
            assert(action.type == ActionType::PhysicalFront);
            assert(action.qubit_idx_0 < cur_game_ptr->physical_qubit_num);
            assert(action.qubit_idx_1 < cur_game_ptr->physical_qubit_num);

            // apply action
            Reward reward = cur_game_ptr->apply_action(action);

            // save game to buffer if not finished && w.p. 1 / 10 to avoid buffer overflow
            bool _pass_roll_dice = (uniform01dist(random_generator) < 0.1);
            if (!is_finished() && _pass_roll_dice) game_buffer.save(*cur_game_ptr);

            // return reward
            return reward;
        }

        bool is_finished() override {
            return is_circuit_finished(cur_game_ptr->graph);
        }

        int total_cost() override {
            // call this only when game ends !!!
            return cur_game_ptr->total_cost();
        }

        State get_state() override {
            return cur_game_ptr->state();
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
        // basic parameters
        std::shared_ptr<DeviceTopologyGraph> device;
        std::shared_ptr<Context> context;
        std::shared_ptr<Graph> graph;

        // game buffer
        double start_from_internal_prob;
        GameBuffer game_buffer;

        // randomness related
        int seed;
        std::mt19937 random_generator;
        std::uniform_real_distribution<double> uniform01dist;
    };
}
