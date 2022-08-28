#pragma once

#include "topologies/ibm_q20_tokyo.h"
#include "topologies/q20_clique.h"
#include "topologies/q5_test.h"
#include "topologies/ibm_q127_eagle.h"
#include "topologies/ibm_q27_falcon.h"

namespace quartz {
    enum class BackendType {
        Q20_CLIQUE = 0,
        IBM_Q20_TOKYO = 1,
        Q5_TEST = 2,
        IBM_Q127_EAGLE = 3,
        IBM_Q27_FALCON = 4,
    };

    std::ostream &operator<<(std::ostream &stream, BackendType t) {
        const std::string name_list[] = {"Q20_CLIQUE",
                                         "IBM_Q20_TOKYO",
                                         "Q5_TEST",
                                         "IBM_Q127_EAGLE",
                                         "IBM_Q27_FALCON"};
        return stream << name_list[int(t)];
    }

    std::shared_ptr<DeviceTopologyGraph> GetDevice(BackendType backend_type) {
        // return device according to input device_type
        if (backend_type == BackendType::Q20_CLIQUE) return Q20_Clique();
        else if (backend_type == BackendType::IBM_Q20_TOKYO) return IBM_Q20_Tokyo();
        else if (backend_type == BackendType::Q5_TEST) return Q5_Test();
        else if (backend_type == BackendType::IBM_Q127_EAGLE) return IBM_Q127_Eagle();
        else if (backend_type == BackendType::IBM_Q27_FALCON) return IBM_Q27_Falcon();
        else {
            assert(false);
            return {};
        }
    }

}
