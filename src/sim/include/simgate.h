#ifndef _SIMGATE_H_
#define _SIMGATE_H_

#include <algorithm>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

#include "const.h"

namespace sim {

// gates: SimGateType, num_target, num_control, vector: target, vector: control,
// matrix special gate: shuffle: vector:target => target permutation

// DT: DataType: FP32/FP64
template <typename DT, typename Mat = std::vector<std::complex<DT>>>
struct Gate {
  SimGateType gtype;

  unsigned num_target;
  unsigned num_control;
  std::vector<int> target;
  std::vector<int> control;
  std::vector<int> control_value;

  std::vector<Mat> matrix;

};

struct FusedGate {
  SimGateType gtype;

  unsigned num_target;
  unsigned num_control;
  // int target[MAX_KERNEL_SIZE];
  // int control[MAX_KERNEL_SIZE];
  qindex target_physical = 0;
  qindex control_physical = 0;

  qComplex matrix[MAX_DEVICES*(1<<MAX_KERNEL_SIZE)];

  FusedGate(const Gate<qreal> &gate) {
    num_target = gate.num_target;
    num_control = gate.num_control;
    for (int i = 0; i < gate.target.size(); i++) {
      target_physical |= qindex(1) << gate.target[i];
    }
    for (int i = 0; i < gate.control.size(); i++) {
      control_physical |= qindex(1) << gate.control[i];
    }
    for (int i = 0; i < gate.matrix.size(); i++) {
      for (int j = 0; j < gate.matrix[i].size(); j++) {
        matrix[i*gate.matrix[i].size()+j].x = gate.matrix[i][j].real();
        matrix[i*gate.matrix[i].size()+j].y = gate.matrix[i][j].imag();
      }    
    }
  }
    

};

struct KernelGate {
  int targetQubit;
  int controlQubit;
  int controlQubit2;
  KernelGateType type;
  char targetIsGlobal;   // 0-local 1-global
  char controlIsGlobal;  // 0-local 1-global 2-not control
  char control2IsGlobal; // 0-local 1-global 2-not control
  qreal r00, i00, r01, i01, r10, i10, r11, i11;

  KernelGate(KernelGateType type_, int controlQubit2_, char control2IsGlobal_,
             int controlQubit_, char controlIsGlobal_, int targetQubit_,
             char targetIsGlobal_, const qComplex mat[2][2])
      : targetQubit(targetQubit_), controlQubit(controlQubit_),
        controlQubit2(controlQubit2_), type(type_),
        targetIsGlobal(targetIsGlobal_), controlIsGlobal(controlIsGlobal_),
        control2IsGlobal(control2IsGlobal_), r00(mat[0][0].x), i00(mat[0][0].y),
        r01(mat[0][1].x), i01(mat[0][1].y), r10(mat[1][0].x), i10(mat[1][0].y),
        r11(mat[1][1].x), i11(mat[1][1].y) {}

  KernelGate(KernelGateType type_, int controlQubit_, char controlIsGlobal_,
             int targetQubit_, char targetIsGlobal_, const qComplex mat[2][2])
      : KernelGate(type_, 2, -1, controlQubit_, controlIsGlobal_, targetQubit_,
                   targetIsGlobal_, mat) {}

  KernelGate(KernelGateType type_, int targetQubit_, char targetIsGlobal_,
             const qComplex mat[2][2])
      : KernelGate(type_, 2, -1, 2, -1, targetQubit_, targetIsGlobal_, mat) {}

  KernelGate() = default;

  static KernelGate ID() {
    qComplex mat[2][2] = {1, 0, 0, 1};
    return KernelGate(KernelGateType::ID, 0, 0, mat);
  }
};

} // namespace sim

#endif // _SIMGATE_H_
