#pragma once

#include <cstring>
#include <complex>
#include <iostream>

class MatrixBase {
 public:
  virtual void clear() {}
  virtual std::complex<double> *operator[](int x) { return nullptr; }
  virtual const std::complex<double> *operator[](int x) const { return nullptr; }
  virtual MatrixBase operator*(const MatrixBase &other) const { return MatrixBase(); }
  virtual MatrixBase &operator*=(const MatrixBase &b) { return *this; }
  virtual void print() const {}
  virtual ~MatrixBase() = default;
};

template<int kSize>
class Matrix : public MatrixBase {
 public:
  void clear() { memset(data_, 0, sizeof(data_)); }
  Matrix() : data_(0) {}
  Matrix(std::complex<double> data[kSize][kSize]) : data_(data) {}
  Matrix(std::initializer_list<std::initializer_list<std::complex<double>>> data) {
    int counter = 0;
    for (auto &row : data) {
      std::copy(row.begin(), row.end(), data_[counter++]);
    }
  }
  static Matrix identity() {
    Matrix I;
    for (int i = 0; i < kSize; i++)
      I[i][i] = 1;
    return I;
  }
  std::complex<double> *operator[](int x) { return data_[x]; }
  const std::complex<double> *operator[](int x) const { return data_[x]; }
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
  Matrix &operator*=(const Matrix &b) {
    return *this = *this * b;
  }
  void print() const {
    for (int i = 0; i < kSize; i++) {
      for (int j = 0; j < kSize; j++)
        std::cout << data_[i][j] << " ";
      std::cout << std::endl;
    }
  }
 private:
  std::complex<double> data_[kSize][kSize];
};
