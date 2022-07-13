#pragma once

#include "../../device/device.h"

namespace quartz {

    std::shared_ptr<DeviceTopologyGraph> Q20_Clique() {
        auto device = std::make_shared<quartz::SymmetricUniformDevice>(20);
        for (int i = 0; i < 20; ++i) {
            for (int j = i + 1; j < 20; ++j) {
                device->add_edge(i, j);
            }
        }
        return device;
    }


}