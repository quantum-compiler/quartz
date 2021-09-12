#include "vector.h"
#include "../utils/utils.h"
#include <cassert>

bool Vector::apply_matrix(MatrixBase *mat,
                          const std::vector<int> &qubit_indices) {
  const int n0 = qubit_indices.size();
  assert(n0 <= 30);  // 1 << n0 does not overflow
  assert(mat->size() == (1 << n0));
  const int S = data_.size();
  assert(S >= (1 << n0));

  std::vector<ComplexType> buffer(1 << n0);

  for (int i = 0; i < S; i++) {
    bool already_applied = false;
    for (const auto &j : qubit_indices) {
      if (i & (1 << j)) {
        already_applied = true;
        break;
      }
    }
    if (already_applied)
      continue;

    for (auto &val : buffer)
      val = 0;

    // matrix * vector
    for (int j = 0; j < (1 << n0); j++) {
      for (int k = 0; k < (1 << n0); k++) {
        int index = i;
        for (int l = 0; l < n0; l++) {
          if (k & (1 << l)) {
            index ^= (1 << qubit_indices[l]);
          }
        }
        buffer[j] += (*mat)[j][k] * data_[index];
      }
    }

    for (int k = 0; k < (1 << n0); k++) {
      int index = i;
      for (int l = 0; l < n0; l++) {
        if (k & (1 << l)) {
          index ^= (1 << qubit_indices[l]);
        }
      }
      data_[index] = buffer[k];
    }
  }
  return true;
}
