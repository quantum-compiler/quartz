#pragma once

#include "../../device/device.h"

namespace quartz {

    std::shared_ptr<DeviceTopologyGraph> IBM_Q20_Tokyo() {
        // initialize device IBM Q20 Tokyo
        auto device = std::make_shared<quartz::SymmetricUniformDevice>(20);
        // first row
        device->add_edge(0, 1);
        device->add_edge(1, 2);
        device->add_edge(2, 3);
        device->add_edge(3, 4);
        // second row
        device->add_edge(5, 6);
        device->add_edge(6, 7);
        device->add_edge(7, 8);
        device->add_edge(8, 9);
        // third row
        device->add_edge(10, 11);
        device->add_edge(11, 12);
        device->add_edge(12, 13);
        device->add_edge(13, 14);
        // fourth row
        device->add_edge(15, 16);
        device->add_edge(16, 17);
        device->add_edge(17, 18);
        device->add_edge(18, 19);
        // first col
        device->add_edge(0, 5);
        device->add_edge(5, 10);
        device->add_edge(10, 15);
        // second col
        device->add_edge(1, 6);
        device->add_edge(6, 11);
        device->add_edge(11, 16);
        // third col
        device->add_edge(2, 7);
        device->add_edge(7, 12);
        device->add_edge(12, 17);
        // fourth col
        device->add_edge(3, 8);
        device->add_edge(8, 13);
        device->add_edge(13, 18);
        // fifth col
        device->add_edge(4, 9);
        device->add_edge(9, 14);
        device->add_edge(14, 19);
        // crossing in row 1
        device->add_edge(1, 7);
        device->add_edge(2, 6);
        device->add_edge(3, 9);
        device->add_edge(4, 8);
        // crossing in row 2
        device->add_edge(5, 11);
        device->add_edge(6, 10);
        device->add_edge(7, 13);
        device->add_edge(8, 12);
        // crossing in row 3
        device->add_edge(11, 17);
        device->add_edge(12, 16);
        device->add_edge(13, 19);
        device->add_edge(14, 18);

        return device;
    }


}