#pragma once

#include "gate.h"

namespace quartz {
    class SXGate : public Gate {
    public:
        SXGate()
                : Gate(GateType::sx, 1 /*num_qubits*/, 0 /*num_parameters*/),
                  mat({{ComplexType(0.5, 0.5), ComplexType(0.5, -0.5)},
                       {ComplexType(0.5, -0.5), ComplexType(0.5, 0.5)}}) {}

        MatrixBase *get_matrix() override { return &mat; }
        Matrix< 2 > mat;
    };

} // namespace quartz