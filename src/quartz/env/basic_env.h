#pragma once

#include "../game/game.h"

namespace quartz {
    class BasicEnv {
    public:
        // reset the environment to start a new game
        virtual void reset() = 0;

        // move one step forward in env
        virtual Reward step(Action action) = 0;

        // check whether current game is finished
        virtual bool is_finished() = 0;

        // state and action space
        virtual State get_state() = 0;

        // action space
        virtual std::vector<Action> get_action_space() = 0;

    public:
        std::shared_ptr<Game> cur_game_ptr;
    };
}
