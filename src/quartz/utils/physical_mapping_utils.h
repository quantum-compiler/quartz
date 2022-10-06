#pragma once

namespace quartz {
    void Assert(bool statement, const std::string& msg) {
        if (!statement) std::cout << "Error: " << msg << std::endl;
        assert(statement);
    }
}