#pragma once

#include "../game/game_search.h"
#include "../parser/qasm_parser.h"
#include "../supported_devices/supported_devices.h"
#include "../game/game_buffer.h"

namespace quartz {
    class SimpleSearchEnv {
    public:
        SimpleSearchEnv() : start_from_internal_prob(-1), seed(-1), game_buffer(0), random_generator(seed) {}

        SimpleSearchEnv(const std::string &qasm_file_path, BackendType backend_type,
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
            set_search_initial_mapping(*graph, static_cast<int>(uniform01dist(random_generator) * 1000000));
            assert(graph->check_mapping_correctness() == MappingStatus::VALID);
            cur_game_ptr = std::make_shared<GameSearch>(GameSearch(*graph, device));

            // initialize game buffer (game_buffer per se is initialized above)
            start_from_internal_prob = _start_from_internal_prob;
            game_buffer.save(*cur_game_ptr);
        }

        SimpleSearchEnv(const SimpleSearchEnv &old_env)
                : game_buffer(old_env.seed), random_generator(old_env.seed) {
            // basic parameters
            // TODO: context here refer to the same object when copy, consider changing it if necessary
            GameSearch game_copy = *old_env.cur_game_ptr;
            cur_game_ptr = std::make_shared<GameSearch>(game_copy);
            device = old_env.device;
            context = old_env.context;
            Graph graph_copy = *old_env.graph;
            graph = std::make_shared<Graph>(graph_copy);

            // game buffer
            start_from_internal_prob = old_env.start_from_internal_prob;
            game_buffer.random_generator = old_env.game_buffer.random_generator;
            game_buffer.buffer = old_env.game_buffer.buffer;

            // randomness related
            seed = old_env.seed;
            random_generator = old_env.random_generator;
            uniform01dist = old_env.uniform01dist;
        }

        void reset() {
            // generate a random number
            double _random_val = uniform01dist(random_generator);

            // set new game
            if (_random_val < start_from_internal_prob) {
                // start from internal states
                GameSearch sampled_game = game_buffer.sample();
                cur_game_ptr = std::make_shared<GameSearch>(sampled_game);
            } else {
                // start from initial states
                set_search_initial_mapping(*graph, static_cast<int>(uniform01dist(random_generator) * 1000000));
                assert(graph->check_mapping_correctness() == MappingStatus::VALID);
                cur_game_ptr = std::make_shared<GameSearch>(GameSearch(*graph, device));
            }
        }

        std::shared_ptr<SimpleSearchEnv> copy() {
            SimpleSearchEnv env_copy = *this;
            return std::make_shared<SimpleSearchEnv>(env_copy);
        }

        Reward step(Action action) {
            // check whether action is valid
            assert(action.type == ActionType::PhysicalFull);
            assert(action.qubit_idx_0 < cur_game_ptr->physical_qubit_num);
            assert(action.qubit_idx_1 < cur_game_ptr->physical_qubit_num);

            // apply action
            Reward reward = cur_game_ptr->apply_action(action);

            // save game to buffer
            game_buffer.save(*cur_game_ptr);

            // return reward
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
        // basic parameters
        std::shared_ptr<GameSearch> cur_game_ptr;
        std::shared_ptr<DeviceTopologyGraph> device;
        std::shared_ptr<Context> context;
        std::shared_ptr<Graph> graph;

        // game buffer
        double start_from_internal_prob;
        GameSearchBuffer game_buffer;

        // randomness related
        int seed;
        std::mt19937 random_generator;
        std::uniform_real_distribution<double> uniform01dist;
    };
}
