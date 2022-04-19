#include "quartz/device/device.h"
#include <iostream>

int main() {
    // build a generic device
    auto generic_device = quartz::GenericDevice(9);
    generic_device.add_qubit();
    generic_device.add_edge(0, 1, 4, true);
    generic_device.add_edge(0, 7, 8, true);
    generic_device.add_edge(1, 2, 8, true);
    generic_device.add_edge(1, 7, 11, true);
    generic_device.add_edge(2, 3, 7, true);
    generic_device.add_qubit();
    generic_device.add_edge(2, 8, 2, true);
    generic_device.add_edge(2, 5, 4, true);
    generic_device.add_edge(3, 4, 9, true);
    generic_device.add_edge(3, 5, 14, true);
    generic_device.add_edge(4, 5, 10, true);
    generic_device.add_edge(5, 6, 2, true);
    generic_device.add_edge(6, 7, 1, true);
    generic_device.add_edge(6, 8, 6, true);
    generic_device.add_edge(7, 8, 7, true);
    generic_device.add_qubit();
    generic_device.cache_swap_cost();
    // generic_device.print();
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 12; ++j) {
            std::cout << "src: " << i << ", dst: " << j << ", cost: " << generic_device.cal_swap_cost(i, j) << '\n';
        }
    }
    std::cout << "Has 0->1 edge: " << generic_device.has_edge(0, 1) << '\n';
    std::cout << "Has 1->3 edge: " << generic_device.has_edge(1, 3) << '\n';

    // build a generic device
    auto symmetric_device = quartz::SymmetricUniformDevice(9);
    symmetric_device.add_qubit();
    symmetric_device.add_edge(0, 1);
    symmetric_device.add_edge(0, 7);
    symmetric_device.add_edge(1, 2);
    symmetric_device.add_edge(1, 7);
    symmetric_device.add_edge(2, 3);
    symmetric_device.add_qubit();
    symmetric_device.add_edge(2, 8);
    symmetric_device.add_edge(2, 5);
    symmetric_device.add_edge(3, 4);
    symmetric_device.add_edge(3, 5);
    symmetric_device.add_edge(4, 5);
    symmetric_device.add_edge(5, 6);
    symmetric_device.add_edge(6, 7);
    symmetric_device.add_edge(6, 8);
    symmetric_device.add_edge(7, 8);
    symmetric_device.add_qubit();
    symmetric_device.cache_swap_cost();
    // generic_device.print();
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 12; ++j) {
            std::cout << "src: " << i << ", dst: " << j << ", cost: " << symmetric_device.cal_swap_cost(i, j) << '\n';
        }
    }
    std::cout << "Has 0->1 edge: " << symmetric_device.has_edge(0, 1) << '\n';
    std::cout << "Has 1->3 edge: " << symmetric_device.has_edge(1, 3) << '\n';

    // get neighbours
    auto neighbours = symmetric_device.get_input_neighbours(7);
    std::cout << "Neighbours of qubit 7: ";
    for (auto neighbour: neighbours) { std::cout << neighbour << ' '; }
    std::cout << '\n';
};
