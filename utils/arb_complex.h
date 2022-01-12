#pragma once

#ifdef USE_ARBLIB
#include "arb.h"

#include <cmath>
#include <complex>
#include <iostream>

constexpr slong kArbPrec = 64;

class ArbComplex {
 public:
  using value_type = arb_t;
  ArbComplex() {
    arb_init(re);
    arb_init(im);
  }
  ArbComplex(const std::complex<double> &val) {
    arb_init(re);
    arb_init(im);
    arb_set_d(re, val.real());
    arb_set_d(im, val.imag());
  }
  ArbComplex(double re, double im) {
    arb_init(this->re);
    arb_init(this->im);
    arb_set_d(this->re, re);
    arb_set_d(this->im, im);
  }
  ArbComplex(arb_t re, arb_t im) {
    this->re[0] = re[0];
    this->im[0] = im[0];
  }
  ~ArbComplex() {
    arb_clear(re);
    arb_clear(im);
  }
  ArbComplex operator+(const ArbComplex &other) const {
    ArbComplex result;
    arb_add(result.re, re, other.re, kArbPrec);
    arb_add(result.im, im, other.im, kArbPrec);
    return result;
  }
  ArbComplex &operator+=(const ArbComplex &other) {
    return *this = *this + other;
  }
  ArbComplex operator*(const ArbComplex &other) const {
    ArbComplex result;
    arb_addmul(result.re, re, other.re, kArbPrec);
    arb_addmul(result.re, im, other.im, kArbPrec);
    arb_addmul(result.im, re, other.im, kArbPrec);
    arb_addmul(result.im, im, other.re, kArbPrec);
    return result;
  }
  [[nodiscard]] double real() const {
    return arf_get_d(&re->mid, ARF_RND_NEAR);
  }
  [[nodiscard]] double imag() const {
    return arf_get_d(&im->mid, ARF_RND_NEAR);
  }
  void real(arb_t new_re) {
    arb_set(re, new_re);
  }
  void imag(arb_t new_im) {
    arb_set(im, new_im);
  }
  void real(double new_re) {
    arb_set_d(re, new_re);
  }
  void imag(double new_im) {
    arb_set_d(im, new_im);
  }
  void print(int digits) const {
    std::cout << "(";
    arb_printd(re, digits);
    std::cout << ",";
    arb_printd(im, digits);
    std::cout << ")";
  }
 private:
  arb_t re, im;
};

namespace std {
double abs(const ArbComplex &val);
}

#endif
