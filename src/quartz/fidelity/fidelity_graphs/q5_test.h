#pragma once

#include "../fidelity.h"

namespace quartz {

    std::shared_ptr<FidelityGraph> Q5_Test_Ideal() {
        auto fidelity_graph = std::make_shared<FidelityGraph>(5);
        fidelity_graph->insert_cx_error_rate(0, 1, 0);
        fidelity_graph->insert_cx_error_rate(1, 2, 0);
        fidelity_graph->insert_cx_error_rate(1, 3, 0);
        fidelity_graph->insert_cx_error_rate(3, 4, 0);
        return fidelity_graph;
    }


}