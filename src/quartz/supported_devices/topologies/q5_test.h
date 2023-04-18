#pragma once

#include "../../device/device.h"

namespace quartz {

    std::shared_ptr<DeviceTopologyGraph> Q5_Test() {
        auto device = std::make_shared<quartz::SymmetricUniformDevice>(5);
        device->add_edge(0, 1);
        device->add_edge(1, 2);
        device->add_edge(1, 3);
        device->add_edge(3, 4);
        return device;
    }


}