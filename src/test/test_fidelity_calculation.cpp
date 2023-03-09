#include "fidelity/fidelity.h"

using namespace std;
using namespace quartz;

int main() {
    FidelityGraph graph = FidelityGraph(5);
    graph.insert_cx_error_rate(0, 1, 0.001);
    graph.insert_cx_error_rate(1, 2, 0.002);
    graph.insert_cx_error_rate(2, 3, 0.003);
    double fidelity1 = graph.query_cx_fidelity(0, 1);
    double fidelity2 = graph.query_cx_fidelity(1, 2);
    double fidelity3 = graph.query_cx_fidelity(2, 3);
    std::cout << "Graph returned fidelity: " << exp(fidelity1 + fidelity2 + fidelity3) << std::endl;
    std::cout << "Real fidelity: " << 0.999 * 0.998 * 0.997 << std::endl;
}
