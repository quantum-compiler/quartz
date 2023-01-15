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

  // if using legion, should use LogicalRegion (similar to weights in FlexFlow),
  // currently using std::vector<DT> matrix is a flattened 2D array with size
  // 2^(num_target)x2^(num_target)
  Mat matrix;

  // TODO: add creat APIs for manual creation
};

} // namespace sim

#endif // _SIMGATE_H_
