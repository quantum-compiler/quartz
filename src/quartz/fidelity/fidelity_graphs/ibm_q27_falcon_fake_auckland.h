#pragma once

#include "../fidelity.h"

namespace quartz {

    std::shared_ptr<FidelityGraph> IBM_Q27_Falcon_Auckland() {
        auto fidelity_graph = std::make_shared<FidelityGraph>(27);
        fidelity_graph->insert_cx_error_rate(0, 1, 0.01007158893096663);
        fidelity_graph->insert_cx_error_rate(1, 4, 0.009740807391002193);
        fidelity_graph->insert_cx_error_rate(1, 0, 0.01007158893096663);
        fidelity_graph->insert_cx_error_rate(1, 2, 0.006231198763846091);
        fidelity_graph->insert_cx_error_rate(2, 3, 0.006676496203251309);
        fidelity_graph->insert_cx_error_rate(2, 1, 0.006231198763846091);
        fidelity_graph->insert_cx_error_rate(3, 5, 0.013770758563644625);
        fidelity_graph->insert_cx_error_rate(3, 2, 0.006676496203251309);
        fidelity_graph->insert_cx_error_rate(4, 1, 0.009740807391002193);
        fidelity_graph->insert_cx_error_rate(4, 7, 0.072265626926617);
        fidelity_graph->insert_cx_error_rate(5, 8, 0.006465005134726587);
        fidelity_graph->insert_cx_error_rate(5, 3, 0.013770758563644625);
        fidelity_graph->insert_cx_error_rate(6, 7, 0.016174290843568057);
        fidelity_graph->insert_cx_error_rate(7, 6, 0.016174290843568057);
        fidelity_graph->insert_cx_error_rate(7, 10, 0.08525826881611757);
        fidelity_graph->insert_cx_error_rate(7, 4, 0.072265626926617);
        fidelity_graph->insert_cx_error_rate(8, 11, 0.00849883723770628);
        fidelity_graph->insert_cx_error_rate(8, 9, 0.013973627101974806);
        fidelity_graph->insert_cx_error_rate(8, 5, 0.006465005134726587);
        fidelity_graph->insert_cx_error_rate(9, 8, 0.013973627101974806);
        fidelity_graph->insert_cx_error_rate(10, 7, 0.08525826881611757);
        fidelity_graph->insert_cx_error_rate(10, 12, 0.06888206503554162);
        fidelity_graph->insert_cx_error_rate(11, 8, 0.00849883723770628);
        fidelity_graph->insert_cx_error_rate(11, 14, 0.004873932721507934);
        fidelity_graph->insert_cx_error_rate(12, 13, 0.04054692423303968);
        fidelity_graph->insert_cx_error_rate(12, 10, 0.06888206503554162);
        fidelity_graph->insert_cx_error_rate(12, 15, 0.02172872768); // replaced with avg
        fidelity_graph->insert_cx_error_rate(13, 12, 0.04054692423303968);
        fidelity_graph->insert_cx_error_rate(13, 14, 0.005566986218926456);
        fidelity_graph->insert_cx_error_rate(14, 16, 0.005821049751668417);
        fidelity_graph->insert_cx_error_rate(14, 11, 0.004873932721507934);
        fidelity_graph->insert_cx_error_rate(14, 13, 0.005566986218926456);
        fidelity_graph->insert_cx_error_rate(15, 12, 0.02172872768); // replaced with avg
        fidelity_graph->insert_cx_error_rate(15, 18, 0.02172872768); // replaced with avg
        fidelity_graph->insert_cx_error_rate(16, 14, 0.005821049751668417);
        fidelity_graph->insert_cx_error_rate(16, 19, 0.011679741716557168);
        fidelity_graph->insert_cx_error_rate(17, 18, 0.008655288238753);
        fidelity_graph->insert_cx_error_rate(18, 21, 0.007025027293971864);
        fidelity_graph->insert_cx_error_rate(18, 17, 0.008655288238753);
        fidelity_graph->insert_cx_error_rate(18, 15, 0.02172872768); // replaced with avg
        fidelity_graph->insert_cx_error_rate(19, 22, 0.013004610026883578);
        fidelity_graph->insert_cx_error_rate(19, 16, 0.011679741716557168);
        fidelity_graph->insert_cx_error_rate(19, 20, 0.09048533497624489);
        fidelity_graph->insert_cx_error_rate(20, 19, 0.09048533497624489);
        fidelity_graph->insert_cx_error_rate(21, 23, 0.01151967665578138);
        fidelity_graph->insert_cx_error_rate(21, 18, 0.007025027293971864);
        fidelity_graph->insert_cx_error_rate(22, 19, 0.013004610026883578);
        fidelity_graph->insert_cx_error_rate(22, 25, 0.007198262216018325);
        fidelity_graph->insert_cx_error_rate(23, 24, 0.012106666534920735);
        fidelity_graph->insert_cx_error_rate(23, 21, 0.01151967665578138);
        fidelity_graph->insert_cx_error_rate(24, 23, 0.012106666534920735);
        fidelity_graph->insert_cx_error_rate(24, 25, 0.02172872768); // replaced with avg
        fidelity_graph->insert_cx_error_rate(25, 22, 0.007198262216018325);
        fidelity_graph->insert_cx_error_rate(25, 26, 0.006726120382336276);
        fidelity_graph->insert_cx_error_rate(25, 24, 0.02172872768); // replaced with avg
        fidelity_graph->insert_cx_error_rate(26, 25, 0.006726120382336276);
        return fidelity_graph;
    }


}