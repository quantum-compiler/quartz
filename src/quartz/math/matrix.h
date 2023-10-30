#pragma once

#include "../utils/utils.h"

#include <cstring>
#include <iostream>
#include <vector>

namespace quartz {
class MatrixBase {
 public:
  virtual void clear() {}
  virtual ComplexType *operator[](int x) { return nullptr; }
  virtual const ComplexType *operator[](int x) const { return nullptr; }
  virtual MatrixBase operator*(const MatrixBase &other) const {
    return MatrixBase();
  }
  virtual MatrixBase &operator*=(const MatrixBase &b) { return *this; }
  virtual void print() const {}
  virtual int size() const { return -1; }
  virtual std::vector<ComplexType> flatten() {
    std::vector<ComplexType> flattened_mat;
    return flattened_mat;
  }
  virtual ~MatrixBase() = default;
};

template <int kSize> class Matrix : public MatrixBase {
 public:
  void clear() { memset(data_, 0, sizeof(data_)); }
  Matrix() = default;
  Matrix(ComplexType data[kSize][kSize]) : data_(data) {}
  Matrix(std::initializer_list<std::initializer_list<ComplexType>> data) {
    int counter = 0;
    for (auto &row : data) {
      std::copy(row.begin(), row.end(), data_[counter++]);
    }
  }
#ifdef USE_ARBLIB
  Matrix(
      std::initializer_list<std::initializer_list<std::complex<double>>> data) {
    int counter = 0;
    for (auto &row : data) {
      std::copy(row.begin(), row.end(), data_[counter++]);
    }
  }
#endif
  static Matrix identity() {
    Matrix I;
    for (int i = 0; i < kSize; i++)
      I[i][i] = 1;
    return I;
  }
  ComplexType *operator[](int x) { return data_[x]; }
  const ComplexType *operator[](int x) const { return data_[x]; }
  Matrix operator*(const Matrix &other) const {
    Matrix result;
    for (int i = 0; i < kSize; i++) {
      const auto *data_i = data_[i];
      auto *result_i = result[i];
      for (int j = 0; j < kSize; j++) {
        const auto *other_j = other[j];
        for (int k = 0; k < kSize; k++) {
          result_i[k] += data_i[j] * other_j[k];
        }
      }
    }
    return result;
  }
  Matrix &operator*=(const Matrix &b) { return *this = *this * b; }
  void print() const {
    for (int i = 0; i < kSize; i++) {
      for (int j = 0; j < kSize; j++) {
#ifdef USE_ARBLIB
        data_[i][j].print(/*digits=*/4);
#else
        std::cout << data_[i][j];
#endif
        std::cout << " ";
      }
      std::cout << std::endl;
    }
  }
  int size() const { return kSize; }

  std::vector<ComplexType> flatten() {
    std::vector<ComplexType> flattened_mat;
    flattened_mat.reserve(kSize * kSize);
    for (int i = 0; i < kSize; i++) {
      for (int j = 0; j < kSize; j++) {
        flattened_mat.push_back(data_[i][j]);
      }
    }

    return flattened_mat;
  }

 private:
  ComplexType data_[kSize][kSize];
};

}  // namespace quartz
