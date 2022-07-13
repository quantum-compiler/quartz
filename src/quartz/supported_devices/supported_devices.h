#pragma once

#include "topologies/ibm_q20_tokyo.h"
#include "topologies/clique.h"

namespace quartz {
    enum class BackendType {
        Q20_CLIQUE = 0,
        IBM_Q20_TOKYO = 1,
    };

    std::shared_ptr<DeviceTopologyGraph> GetDevice(BackendType backend_type) {
        // return device according to input device_type
        if (backend_type == BackendType::Q20_CLIQUE) return Q20_Clique();
        else if (backend_type == BackendType::IBM_Q20_TOKYO) return IBM_Q20_Tokyo();
        else {
            assert(false);
            return {};
        }
    }

}
