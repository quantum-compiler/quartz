#pragma once

#include "gate.h"

namespace quartz {
    class SXDGGate : public Gate {
    public:
        SXDGGate()
                : Gate(GateType::sxdg, 1 /*num_qubits*/, 0 /*num_parameters*/),
                  mat({{ComplexType(0.5, -0.5), ComplexType(0.5, 0.5)},
                       {ComplexType(0.5, 0.5), ComplexType(0.5, -0.5)}}) {}

        MatrixBase *get_matrix() override { return &mat; }
        Matrix< 2 > mat;
    };

} // namespace quartz