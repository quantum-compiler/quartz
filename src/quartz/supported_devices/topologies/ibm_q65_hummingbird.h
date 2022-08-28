#pragma once

#include "../../device/device.h"

namespace quartz {

    std::shared_ptr<DeviceTopologyGraph> IBM_Q65_Hummingbird() {
        // initialize device IBM Q65 HummingBird
        auto device = std::make_shared<quartz::SymmetricUniformDevice>(65);
        // first row
        device->add_edge(0, 1);
        device->add_edge(1, 2);
        device->add_edge(2, 3);
        device->add_edge(3, 4);
        device->add_edge(4, 5);
        device->add_edge(5, 6);
        device->add_edge(6, 7);
        device->add_edge(7, 8);
        device->add_edge(8, 9);
        // first layer columns
        device->add_edge(0, 10);
        device->add_edge(10, 13);
        device->add_edge(4, 11);
        device->add_edge(11, 17);
        device->add_edge(8, 12);
        device->add_edge(12, 21);
        // second row
        device->add_edge(13, 14);
        device->add_edge(14, 15);
        device->add_edge(15, 16);
        device->add_edge(16, 17);
        device->add_edge(17, 18);
        device->add_edge(18, 19);
        device->add_edge(19, 20);
        device->add_edge(20, 21);
        device->add_edge(21, 22);
        device->add_edge(22, 23);
        // second layer columns
        device->add_edge(15, 24);
        device->add_edge(24, 29);
        device->add_edge(19, 25);
        device->add_edge(25, 33);
        device->add_edge(23, 26);
        device->add_edge(26, 37);
        // third layer
        device->add_edge(27, 28);
        device->add_edge(28, 29);
        device->add_edge(29, 30);
        device->add_edge(30, 31);
        device->add_edge(31, 32);
        device->add_edge(32, 33);
        device->add_edge(33, 34);
        device->add_edge(34, 35);
        device->add_edge(35, 36);
        device->add_edge(36, 37);
        // third layer columns
        device->add_edge(27, 38);
        device->add_edge(38, 41);
        device->add_edge(31, 39);
        device->add_edge(39, 45);
        device->add_edge(35, 40);
        device->add_edge(40, 49);
        // fourth layer
        device->add_edge(41, 42);
        device->add_edge(42, 43);
        device->add_edge(43, 44);
        device->add_edge(44, 45);
        device->add_edge(45, 46);
        device->add_edge(46, 47);
        device->add_edge(47, 48);
        device->add_edge(48, 49);
        device->add_edge(49, 50);
        device->add_edge(50, 51);
        // fourth layer columns
        device->add_edge(43, 52);
        device->add_edge(52, 56);
        device->add_edge(47, 53);
        device->add_edge(53, 60);
        device->add_edge(51, 54);
        device->add_edge(54, 64);
        // fifth layer
        device->add_edge(55, 56);
        device->add_edge(56, 57);
        device->add_edge(57, 58);
        device->add_edge(58, 59);
        device->add_edge(59, 60);
        device->add_edge(60, 61);
        device->add_edge(61, 62);
        device->add_edge(62, 63);
        device->add_edge(63, 64);

        return device;
    }


}