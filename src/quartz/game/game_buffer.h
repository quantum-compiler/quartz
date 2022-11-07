#pragma once

#include "../tasograph/tasograph.h"
#include "../sabre/sabre_swap.h"
#include "game.h"
#include "game_search.h"
#include "game_hybrid.h"

namespace quartz {
    class GameBuffer {
    public:
        GameBuffer() = delete;

        explicit GameBuffer(int seed) : random_generator(seed) {}

        void save(const Game& game) {
            buffer.emplace_back(game);
            if (buffer.size() > 1000) {
                buffer.pop_front();
            }
        }

        Game sample() {
            // calculate index
            std::uniform_int_distribution<int> uni(0, static_cast<int>(buffer.size()) - 1);
            int index = uni(random_generator);

            // return sampled game
            return buffer[index];
        }

    public:
        std::mt19937 random_generator;
        std::deque<Game> buffer;
    };

    class GameSearchBuffer {
    public:
        GameSearchBuffer() = delete;

        explicit GameSearchBuffer(int seed) : random_generator(seed) {
            buffer.reserve(10000);
        }

        void save(const GameSearch& game) {
            buffer.emplace_back(game);
        }

        GameSearch sample() {
            // calculate index
            std::uniform_int_distribution<int> uni(0, static_cast<int>(buffer.size()) - 1);
            int index = uni(random_generator);

            // return sampled game
            return buffer[index];
        }

    public:
        std::mt19937 random_generator;
        std::vector<GameSearch> buffer;
    };

    class GameHybridBuffer {
    public:
        GameHybridBuffer() = delete;

        explicit GameHybridBuffer(int seed) : random_generator(seed) {}

        void save(const GameHybrid& game) {
            buffer.emplace_back(game);
            if (buffer.size() > 1000) {
                buffer.pop_front();
            }
        }

        GameHybrid sample() {
            // calculate index
            std::uniform_int_distribution<int> uni(0, static_cast<int>(buffer.size()) - 1);
            int index = uni(random_generator);

            // return sampled game
            return buffer[index];
        }

    public:
        std::mt19937 random_generator;
        std::deque<GameHybrid> buffer;
    };
}