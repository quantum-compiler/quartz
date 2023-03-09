#pragma once

#include <memory>
#include <cassert>

#include "../supported_devices/supported_devices.h"
#include "fidelity_graphs/ibm_q27_falcon_fake_auckland.h"
#include "fidelity_graphs/ibm_q65_hummingbird_fake_ithaca.h"
#include "fidelity_graphs/ibm_q127_eagle_fake_washington.h"

namespace quartz {

    std::shared_ptr<FidelityGraph> GetFidelityGraph(BackendType backend_type) {
        // return fidelity graph according to input device_type
        if (backend_type == BackendType::Q20_CLIQUE)
            Assert(false, "Q20_CLIQUE has no fidelity graph");
        else if (backend_type == BackendType::IBM_Q20_TOKYO)
            Assert(false, "IBM_Q20_TOKYO has no fidelity graph");
        else if (backend_type == BackendType::Q5_TEST)
            Assert(false, "Q5_TEST has no fidelity graph");
        else if (backend_type == BackendType::IBM_Q127_EAGLE)
            return IBM_Q127_Eagle_Washington();
        else if (backend_type == BackendType::IBM_Q27_FALCON)
            return IBM_Q27_Falcon_Auckland();
        else if (backend_type == BackendType::IBM_Q65_HUMMINGBIRD)
            return IBM_Q65_Hummingbird_Ithaca();
        else
            Assert(false, "Unknown device type!");
        return {};
    }

}
