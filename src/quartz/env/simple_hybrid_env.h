#pragma once

#include "basic_env.h"
#include "../parser/qasm_parser.h"
#include "../supported_devices/supported_devices.h"
#include "../game/game_buffer.h"

namespace quartz {
    class SimpleHybridEnv {
    public:
        SimpleHybridEnv() = delete;

        SimpleHybridEnv(
                // basic environment settings
                const std::string &qasm_file_path, BackendType backend_type,
                const std::string &_initial_mapping_file,
                // Game buffer settings
                int _seed, double _start_from_internal_prob, int _game_buffer_size, int _save_interval,
                // GameHybrid settings
                int _initial_phase_len, bool _allow_nop_in_initial, double _initial_phase_reward
        )
                : game_buffer(_seed, _game_buffer_size), random_generator(_seed),
                  original_qasm_file_path(qasm_file_path) {
            /// A simple hybrid environment that does not use curriculum
            // initialize context, graph and device
            context = std::make_shared<Context>(Context({GateType::h, GateType::cx, GateType::t,
                                                         GateType::tdg, GateType::input_qubit, GateType::s,
                                                         GateType::sdg, GateType::x, GateType::z, GateType::rz,
                                                         GateType::sx, GateType::sxdg}));
            graph = Graph::from_qasm_file(&(*context), qasm_file_path);
            device = GetDevice(backend_type);

            // set randomness (random_generator is initialized above)
            seed = _seed;
            uniform01dist = std::uniform_real_distribution<double>(0.0, 1.0);

            // save GameHybrid settings
            initial_phase_len = _initial_phase_len;
            allow_nop_in_initial = _allow_nop_in_initial;
            initial_phase_reward = _initial_phase_reward;

            // initialize mapping for graph and create game
            device_reg_count = device->get_num_qubits();
            initial_mapping_file = _initial_mapping_file;
            set_initial_mapping(*graph, static_cast<int>(uniform01dist(random_generator) * 1000000),
                                initial_mapping_file, device_reg_count);
            Assert(graph->check_mapping_correctness() == MappingStatus::VALID, "Invalid mapping!");
            cur_game_ptr = std::make_shared<GameHybrid>(
                    GameHybrid(*graph, device, initial_phase_len, allow_nop_in_initial, initial_phase_reward));
            Assert(!is_circuit_finished(cur_game_ptr->graph), "Empty GameHybrid found!");

            // initialize game buffer (game_buffer is initialize above)
            game_buffer_size = _game_buffer_size;
            save_interval = _save_interval;
            cur_save_step = 0;
            start_from_internal_prob = _start_from_internal_prob;
            game_buffer.save(*cur_game_ptr);
        }

        SimpleHybridEnv(const SimpleHybridEnv &old_env)
                : game_buffer(old_env.seed, old_env.game_buffer_size), random_generator(old_env.seed) {
            // basic parameters
            // TODO: context here refer to the same object when copy, consider changing it if necessary
            device_reg_count = old_env.device_reg_count;
            initial_mapping_file = old_env.initial_mapping_file;
            GameHybrid game_copy = *old_env.cur_game_ptr;
            cur_game_ptr = std::make_shared<GameHybrid>(game_copy);
            device = old_env.device;
            context = old_env.context;
            Graph graph_copy = *old_env.graph;
            graph = std::make_shared<Graph>(graph_copy);

            // game buffer
            game_buffer_size = old_env.game_buffer_size;
            save_interval = old_env.save_interval;
            cur_save_step = old_env.cur_save_step;
            start_from_internal_prob = old_env.start_from_internal_prob;
            game_buffer.random_generator = old_env.game_buffer.random_generator;
            game_buffer.buffer = old_env.game_buffer.buffer;

            // GameHybrid settings
            initial_phase_len = old_env.initial_phase_len;
            allow_nop_in_initial = old_env.allow_nop_in_initial;
            initial_phase_reward = old_env.initial_phase_reward;

            // randomness related
            seed = old_env.seed;
            random_generator = old_env.random_generator;
            uniform01dist = old_env.uniform01dist;

            // original qasm file path
            original_qasm_file_path = old_env.original_qasm_file_path;
        }

        void reset() {
            // generate a random number
            double _random_val = uniform01dist(random_generator);

            // set new game
            if (_random_val < start_from_internal_prob) {
                // start from internal states
                GameHybrid sampled_game = game_buffer.sample();
                cur_game_ptr = std::make_shared<GameHybrid>(sampled_game);
            } else {
                // start from initial states
                set_initial_mapping(*graph, static_cast<int>(uniform01dist(random_generator) * 1000000),
                                    initial_mapping_file, device_reg_count);
                Assert(graph->check_mapping_correctness() == MappingStatus::VALID, "Invalid Mapping!");
                cur_game_ptr = std::make_shared<GameHybrid>(
                        GameHybrid(*graph, device, initial_phase_len, allow_nop_in_initial, initial_phase_reward));
                Assert(!is_finished(), "Empty GameHybrid found!");
            }
        }

        Reward step(Action action) {
            // check whether action is valid
            Assert(action.qubit_idx_0 < cur_game_ptr->physical_qubit_num, "step: id check failed!");
            Assert(action.qubit_idx_1 < cur_game_ptr->physical_qubit_num, "step: id check failed!");

            // apply action
            Reward reward = cur_game_ptr->apply_action(action);

            // save game to buffer every #save_interval steps
            cur_save_step += 1;
            if (cur_save_step % save_interval == 0) {
                if (!is_finished()) game_buffer.save(*cur_game_ptr);
            }

            // return reward
            return reward;
        }

        [[nodiscard]] bool is_finished() const {
            return is_circuit_finished(cur_game_ptr->graph);
        }

        [[nodiscard]] int total_cost() const {
            // call this only when game ends !!!
            return cur_game_ptr->total_cost();
        }

        [[nodiscard]] State get_state() const {
            return cur_game_ptr->state();
        }

        [[nodiscard]] std::vector<Action> get_action_space() const {
            // ActionType: Physical Front
            std::set<Action, ActionCompare> available_action_set = cur_game_ptr->action_space();
            std::vector<Action> action_space;
            action_space.reserve(available_action_set.size());
            for (const auto &available_action: available_action_set) {
                action_space.emplace_back(available_action);
            }
            return std::move(action_space);
        }

        void save_context_to_file(const std::string &execution_history_file_path,
                                  const std::string &single_qubit_gate_execution_plan_file_path) const {
            // save execution history and single qubit execution plan to file
            // assume the game is finished (otherwise an exception will be thrown)
            cur_game_ptr->save_context_to_file(execution_history_file_path, single_qubit_gate_execution_plan_file_path);
        }

        void generate_mapped_qasm(const std::string &mapped_qasm_file_path, bool debug_mode) const {
            // generate the final mapped qasm file
            // Note: 1. assume the game is finished (otherwise an exception will be thrown)
            //       2. debug mode will conduct some time-consuming checks, usually this is not necessary
            cur_game_ptr->generated_mapping_plan(mapped_qasm_file_path, original_qasm_file_path, debug_mode);
        }

    public:
        // basic parameters
        int device_reg_count;
        std::string initial_mapping_file;
        std::shared_ptr<DeviceTopologyGraph> device;
        std::shared_ptr<Context> context;
        std::shared_ptr<Graph> graph;
        std::shared_ptr<GameHybrid> cur_game_ptr;

        // game buffer
        int game_buffer_size;
        int save_interval;
        int cur_save_step;
        double start_from_internal_prob;
        GameHybridBuffer game_buffer;

        // GameHybrid settings
        int initial_phase_len;
        bool allow_nop_in_initial;
        double initial_phase_reward;

        // randomness related
        int seed;
        std::mt19937 random_generator;
        std::uniform_real_distribution<double> uniform01dist;

        // qasm file path
        std::string original_qasm_file_path;
    };
}
