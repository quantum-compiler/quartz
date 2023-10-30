#pragma once

#include "matrix.h"

#include <random>
#include <vector>

namespace quartz {
// An std::vector<ComplexType> to store the distributions.
class Vector {
 public:
  Vector() = default;
  explicit Vector(int sz) : data_(sz) {}
  explicit Vector(const std::vector<ComplexType> &data) : data_(data) {}
  explicit Vector(std::vector<ComplexType> &&data) : data_(data) {}
  ComplexType &operator[](int x) { return data_[x]; }
  const ComplexType &operator[](int x) const { return data_[x]; }
  [[nodiscard]] int size() const { return (int)data_.size(); }
  bool apply_matrix(MatrixBase *mat, const std::vector<int> &qubit_indices);
  [[nodiscard]] ComplexType dot(const Vector &other) const;  // dot product
  void print() const;

  // If |gen| is not nullptr, then use |gen| as the mt19937 generator.
  // Otherwise, use a static mt19937 generator for this function.
  static Vector random_generate(int num_qubits, std::mt19937 *gen = nullptr);

 private:
  std::vector<ComplexType> data_;
};

}  // namespace quartz
