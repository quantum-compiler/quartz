#pragma once

#include "../../device/device.h"

namespace quartz {

    std::shared_ptr<DeviceTopologyGraph> IBM_Q127_Eagle() {
        // initialize device IBM Q127 Eagle
        auto device = std::make_shared<quartz::SymmetricUniformDevice>(127);
        // first row
        device->add_edge(0, 1);
        device->add_edge(1, 2);
        device->add_edge(2, 3);
        device->add_edge(3, 4);
        device->add_edge(4, 5);
        device->add_edge(5, 6);
        device->add_edge(6, 7);
        device->add_edge(7, 8);
        // device->add_edge(8, 9); // This edge does not exist in real device!
        device->add_edge(9, 10);
        device->add_edge(10, 11);
        device->add_edge(11, 12);
        device->add_edge(12, 13);
        // first layer column
        device->add_edge(0, 14);
        device->add_edge(14, 18);
        device->add_edge(4, 15);
        device->add_edge(15, 22);
        device->add_edge(8, 16);
        device->add_edge(16, 26);
        device->add_edge(12, 17);
        device->add_edge(17, 30);
        // second row
        device->add_edge(18, 19);
        device->add_edge(19, 20);
        device->add_edge(20, 21);
        device->add_edge(21, 22);
        device->add_edge(22, 23);
        device->add_edge(23, 24);
        device->add_edge(24, 25);
        device->add_edge(25, 26);
        device->add_edge(26, 27);
        device->add_edge(27, 28);
        device->add_edge(28, 29);
        device->add_edge(29, 30);
        device->add_edge(30, 31);
        device->add_edge(31, 32);
        // second layer column
        device->add_edge(20, 33);
        device->add_edge(33, 39);
        device->add_edge(24, 34);
        device->add_edge(34, 43);
        device->add_edge(28, 35);
        device->add_edge(35, 47);
        device->add_edge(32, 36);
        device->add_edge(36, 51);
        // third row
        device->add_edge(37, 38);
        device->add_edge(38, 39);
        device->add_edge(39, 40);
        device->add_edge(40, 41);
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
        // third layer column
        device->add_edge(37, 52);
        device->add_edge(52, 56);
        device->add_edge(41, 53);
        device->add_edge(53, 60);
        device->add_edge(45, 54);
        device->add_edge(54, 64);
        device->add_edge(49, 55);
        device->add_edge(55, 68);
        // fourth row
        device->add_edge(56, 57);
        device->add_edge(57, 58);
        device->add_edge(58, 59);
        device->add_edge(59, 60);
        device->add_edge(60, 61);
        device->add_edge(61, 62);
        device->add_edge(62, 63);
        device->add_edge(63, 64);
        device->add_edge(64, 65);
        device->add_edge(65, 66);
        device->add_edge(66, 67);
        device->add_edge(67, 68);
        device->add_edge(68, 69);
        device->add_edge(69, 70);
        // fourth layer column
        device->add_edge(58, 71);
        device->add_edge(71, 77);
        device->add_edge(62, 72);
        device->add_edge(72, 81);
        device->add_edge(66, 73);
        device->add_edge(73, 85);
        device->add_edge(70, 74);
        device->add_edge(74, 89);
        // fifth row
        device->add_edge(75, 76);
        device->add_edge(76, 77);
        device->add_edge(77, 78);
        device->add_edge(78, 79);
        device->add_edge(79, 80);
        device->add_edge(80, 81);
        device->add_edge(81, 82);
        device->add_edge(82, 83);
        device->add_edge(83, 84);
        device->add_edge(84, 85);
        device->add_edge(85, 86);
        device->add_edge(86, 87);
        device->add_edge(87, 88);
        device->add_edge(88, 89);
        // fifth layer column
        device->add_edge(75, 90);
        device->add_edge(90, 94);
        device->add_edge(79, 91);
        device->add_edge(91, 98);
        device->add_edge(83, 92);
        device->add_edge(92, 102);
        device->add_edge(87, 93);
        device->add_edge(93, 106);
        // sixth row
        device->add_edge(94, 95);
        device->add_edge(95, 96);
        device->add_edge(96, 97);
        device->add_edge(97, 98);
        device->add_edge(98, 99);
        device->add_edge(99, 100);
        device->add_edge(100, 101);
        device->add_edge(101, 102);
        device->add_edge(102, 103);
        device->add_edge(103, 104);
        device->add_edge(104, 105);
        device->add_edge(105, 106);
        device->add_edge(106, 107);
        device->add_edge(107, 108);
        // seventh layer column
        device->add_edge(96, 109);
        device->add_edge(109, 114);
        device->add_edge(100, 110);
        device->add_edge(110, 118);
        device->add_edge(104, 111);
        device->add_edge(111, 122);
        device->add_edge(108, 112);
        device->add_edge(112, 126);
        // seventh row
        device->add_edge(113, 114);
        device->add_edge(114, 115);
        device->add_edge(115, 116);
        device->add_edge(116, 117);
        device->add_edge(117, 118);
        device->add_edge(118, 119);
        device->add_edge(119, 120);
        device->add_edge(120, 121);
        device->add_edge(121, 122);
        device->add_edge(122, 123);
        device->add_edge(123, 124);
        device->add_edge(124, 125);
        device->add_edge(125, 126);

        return device;
    }


}