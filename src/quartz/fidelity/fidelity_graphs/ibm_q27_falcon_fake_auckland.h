#pragma once

#include "../fidelity.h"

namespace quartz {

    std::shared_ptr<FidelityGraph> IBM_Q27_Falcon_Auckland() {
        auto fidelity_graph = std::make_shared<FidelityGraph>(27);
        fidelity_graph->insert_cx_error_rate(0, 1, 0.005627107607318493);
        fidelity_graph->insert_cx_error_rate(1, 4, 0.006083832295232822);
        fidelity_graph->insert_cx_error_rate(1, 0, 0.005627107607318493);
        fidelity_graph->insert_cx_error_rate(1, 2, 0.008753486575494296);
        fidelity_graph->insert_cx_error_rate(2, 3, 0.012686563583275995);
        fidelity_graph->insert_cx_error_rate(2, 1, 0.008753486575494296);
        fidelity_graph->insert_cx_error_rate(3, 2, 0.012686563583275995);
        fidelity_graph->insert_cx_error_rate(3, 5, 0.009928171436340755);
        fidelity_graph->insert_cx_error_rate(4, 1, 0.006083832295232822);
        fidelity_graph->insert_cx_error_rate(4, 7, 0.009519176410124675);
        fidelity_graph->insert_cx_error_rate(5, 3, 0.009928171436340755);
        fidelity_graph->insert_cx_error_rate(5, 8, 0.009921422875); // replaced with avg
        fidelity_graph->insert_cx_error_rate(6, 7, 0.011051936086061781);
        fidelity_graph->insert_cx_error_rate(7, 10, 0.009537667266830863);
        fidelity_graph->insert_cx_error_rate(7, 6, 0.011051936086061781);
        fidelity_graph->insert_cx_error_rate(7, 4, 0.009519176410124675);
        fidelity_graph->insert_cx_error_rate(8, 11, 0.011469369285632997);
        fidelity_graph->insert_cx_error_rate(8, 9, 0.009921422875); // replaced with avg
        fidelity_graph->insert_cx_error_rate(8, 5, 0.009921422875); // replaced with avg
        fidelity_graph->insert_cx_error_rate(9, 8, 0.009921422875); // replaced with avg
        fidelity_graph->insert_cx_error_rate(10, 7, 0.009537667266830863);
        fidelity_graph->insert_cx_error_rate(10, 12, 0.009677031085944338);
        fidelity_graph->insert_cx_error_rate(11, 8, 0.011469369285632997);
        fidelity_graph->insert_cx_error_rate(11, 14, 0.009921422875); // replaced with avg
        fidelity_graph->insert_cx_error_rate(12, 13, 0.00970278244374742);
        fidelity_graph->insert_cx_error_rate(12, 10, 0.009677031085944338);
        fidelity_graph->insert_cx_error_rate(12, 15, 0.007260771068301031);
        fidelity_graph->insert_cx_error_rate(13, 12, 0.00970278244374742);
        fidelity_graph->insert_cx_error_rate(13, 14, 0.006010443553518957);
        fidelity_graph->insert_cx_error_rate(14, 16, 0.009042962208406474);
        fidelity_graph->insert_cx_error_rate(14, 13, 0.006010443553518957);
        fidelity_graph->insert_cx_error_rate(14, 11, 0.009921422875); // replaced with avg
        fidelity_graph->insert_cx_error_rate(15, 18, 0.012964723838537334);
        fidelity_graph->insert_cx_error_rate(15, 12, 0.007260771068301031);
        fidelity_graph->insert_cx_error_rate(16, 14, 0.009042962208406474);
        fidelity_graph->insert_cx_error_rate(16, 19, 0.010819489686150568);
        fidelity_graph->insert_cx_error_rate(17, 18, 0.012793805001470343);
        fidelity_graph->insert_cx_error_rate(18, 17, 0.012793805001470343);
        fidelity_graph->insert_cx_error_rate(18, 15, 0.012964723838537334);
        fidelity_graph->insert_cx_error_rate(18, 21, 0.019645921452834336);
        fidelity_graph->insert_cx_error_rate(19, 20, 0.011960903786420785);
        fidelity_graph->insert_cx_error_rate(19, 22, 0.009401566688506613);
        fidelity_graph->insert_cx_error_rate(19, 16, 0.010819489686150568);
        fidelity_graph->insert_cx_error_rate(20, 19, 0.011960903786420785);
        fidelity_graph->insert_cx_error_rate(21, 23, 0.009164472547543462);
        fidelity_graph->insert_cx_error_rate(21, 18, 0.019645921452834336);
        fidelity_graph->insert_cx_error_rate(22, 25, 0.008705657095895769);
        fidelity_graph->insert_cx_error_rate(22, 19, 0.009401566688506613);
        fidelity_graph->insert_cx_error_rate(23, 21, 0.009164472547543462);
        fidelity_graph->insert_cx_error_rate(23, 24, 0.006466994129351394);
        fidelity_graph->insert_cx_error_rate(24, 23, 0.006466994129351394);
        fidelity_graph->insert_cx_error_rate(24, 25, 0.009921422875); // replaced with avg
        fidelity_graph->insert_cx_error_rate(25, 26, 0.009839313876210704);
        fidelity_graph->insert_cx_error_rate(25, 22, 0.008705657095895769);
        fidelity_graph->insert_cx_error_rate(25, 24, 0.009921422875); // replaced with avg
        fidelity_graph->insert_cx_error_rate(26, 25, 0.009839313876210704);
        return fidelity_graph;
    }


}