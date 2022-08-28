#pragma once

#include "../../device/device.h"

namespace quartz {

    std::shared_ptr<DeviceTopologyGraph> IBM_Q27_Falcon() {
        // initialize device IBM Q20 Tokyo
        auto device = std::make_shared<quartz::SymmetricUniformDevice>(27);
        // first row
        device->add_edge(0, 1);
        device->add_edge(1, 4);
        device->add_edge(4, 7);
        device->add_edge(7, 10);
        device->add_edge(10, 12);
        device->add_edge(12, 15);
        device->add_edge(15, 18);
        device->add_edge(18, 21);
        device->add_edge(21, 23);
        // second row
        device->add_edge(3, 5);
        device->add_edge(5, 8);
        device->add_edge(8, 11);
        device->add_edge(11, 14);
        device->add_edge(14, 16);
        device->add_edge(16, 19);
        device->add_edge(19, 22);
        device->add_edge(22, 25);
        device->add_edge(25, 26);
        // columns
        device->add_edge(6, 7);
        device->add_edge(17, 18);
        device->add_edge(1, 2);
        device->add_edge(2, 3);
        device->add_edge(12, 13);
        device->add_edge(13, 14);
        device->add_edge(23, 24);
        device->add_edge(24, 25);
        device->add_edge(8, 9);
        device->add_edge(19, 20);

        return device;
    }


}