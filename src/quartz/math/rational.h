// Adapted from Mingkuan Xu's template for competitive programming.

#include <iostream>
#include <string>
#include <vector>

namespace quartz {

class Complex {
 public:
  double re, im;
  Complex() {}
  Complex(double re, double im = 0) : re(re), im(im) {}
  Complex &operator+=(const Complex &b);
  Complex operator+(const Complex &b) const;
  const Complex &operator+() const;
  Complex &operator-=(const Complex &b);
  Complex operator-(const Complex &b) const;
  Complex operator-() const;
  Complex operator*(const Complex &b) const;
  Complex &operator*=(const Complex &b);
  Complex operator*(double b) const;
  Complex &operator*=(double b);
  [[nodiscard]] double len2() const;
  Complex operator/(const Complex &b) const;
  Complex &operator/=(const Complex &b);
  Complex operator/(double b) const;
  Complex &operator/=(double b);
  [[nodiscard]] double len() const;
  Complex conj() const;
  Complex &toconj();
  Complex muli() const;
};
class FFT {
 private:
  int *rev;
  Complex *wu1, *wu2;
  void fft_core(Complex *x, Complex *wu);

 public:
  const int n;
  explicit FFT(int logn);
  FFT(const FFT &b) = delete;
  FFT &operator=(const FFT &b) = delete;
  ~FFT();
  void fft(Complex *x);
  void fft(Complex *x, Complex *y);  // real numbers
  void ifft(Complex *x);
  void ifft(Complex *x, Complex *y);
};

class Unsigned {
 public:
  static std::vector<FFT *> ffts;

 private:
  static const int NUM = 1000000000;                                 // 1e9
  static const int NUM_DIGIT = 9;
  static const unsigned long long MAXNUM = 17000000000000000000ull;  // 1.7e19
  static const double FFT_MUL_COEFFICIENT;  // see below
  static const double NEWTON_DIVMOD_COEFFICIENT;
  static const double NEWTON_DIVMOD_POWER_2_RATIO;
  static const int FFT_NUM = 1000;
  static const int FFT_NUMD = 3;
  std::vector<int> a;
  void update();
  [[nodiscard]] double force_mul_estimate_time(const Unsigned &b) const;
  [[nodiscard]] Unsigned force_mul(const Unsigned &b) const;
  [[nodiscard]] double fft_mul_estimate_time(const Unsigned &b) const;
  [[nodiscard]] Unsigned fft_mul(const Unsigned &b) const;
  [[nodiscard]] double force_divmod_estimate_time(const Unsigned &b) const;
  Unsigned force_divmod(const Unsigned &b);
  [[nodiscard]] double newton_divmod_estimate_time(const Unsigned &b) const;
  [[nodiscard]] Unsigned rev_10(int n) const;  // 10^n / *this
  [[nodiscard]] bool is_power_of_10() const;
  Unsigned newton_divmod(const Unsigned &b);

 public:
  // comparison functions
  [[nodiscard]] int cmp(const Unsigned &b) const;
  bool operator<(const Unsigned &b) const { return cmp(b) < 0; }
  bool operator<=(const Unsigned &b) const { return cmp(b) <= 0; }
  bool operator>(const Unsigned &b) const { return cmp(b) > 0; }
  bool operator>=(const Unsigned &b) const { return cmp(b) >= 0; }
  bool operator==(const Unsigned &b) const { return cmp(b) == 0; }
  bool operator!=(const Unsigned &b) const { return cmp(b) != 0; }

  // IO functions
  void inputnum(unsigned long long s);
  void assign(const char *s, int len = -1);
  void assign(const std::string &s);
  [[nodiscard]] std::string to_string() const;
  friend std::istream &operator>>(std::istream &cin, Unsigned &b);
  friend std::ostream &operator<<(std::ostream &cout, const Unsigned &b);

  // constructor functions
  Unsigned() { a.push_back(0); }
  Unsigned(const Unsigned &b) : a(b.a) {}
  Unsigned(short s) { inputnum(s); }
  Unsigned(unsigned short s) { inputnum(s); }
  Unsigned(int s) { inputnum(s); }
  Unsigned(unsigned s) { inputnum(s); }
  Unsigned(long long s) { inputnum(s); }
  Unsigned(unsigned long long s) { inputnum(s); }
  Unsigned(const char *s) { assign(s); }
  Unsigned(const std::string &s) { assign(s); }

  // operators
  Unsigned &operator+=(const Unsigned &b);
  Unsigned operator+(const Unsigned &b) const;
  const Unsigned &operator+() const;
  Unsigned &operator-=(const Unsigned &b);
  Unsigned operator-(const Unsigned &b) const;
  Unsigned operator*(const Unsigned &b) const;
  Unsigned &operator*=(const Unsigned &b);
  Unsigned &operator/=(const Unsigned &b);
  Unsigned operator/(const Unsigned &b) const;
  Unsigned &operator%=(const Unsigned &b);
  Unsigned operator%(const Unsigned &b) const;
  bool operator!() const;
  Unsigned &operator++();
  const Unsigned operator++(int);
  Unsigned &operator--();
  const Unsigned operator--(int);

  // other operation functions
  Unsigned divmod(const Unsigned &b);
  Unsigned &shift10_eq(int x);                  //*=10^x
  [[nodiscard]] Unsigned shift10(int x) const;  //*10^x
  [[nodiscard]] Unsigned random() const;        // random number in [0, *this)
  friend Unsigned pow(const Unsigned &a, const Unsigned &b);
  friend Unsigned gcd(const Unsigned &a, const Unsigned &b);
  friend Unsigned abs(const Unsigned &a);

  // helper functions
  [[nodiscard]] bool is_zero() const;
  [[nodiscard]] bool is_odd() const;
  [[nodiscard]] bool is_neg() const;
  [[nodiscard]] int num_digit() const;
  [[nodiscard]] int ctz() const;

  // type conversion functions
  [[nodiscard]] unsigned long long to_ull() const;
  [[nodiscard]] bool to_bool() const;
  [[nodiscard]] long long to_ll() const;
  [[nodiscard]] unsigned to_uint() const;
  [[nodiscard]] int to_int() const;
  [[nodiscard]] unsigned short to_ushort() const;
  [[nodiscard]] short to_short() const;
  [[nodiscard]] double to_double() const;
  [[nodiscard]] long double to_ldouble() const;
};
std::vector<FFT *> Unsigned::ffts;
const int Unsigned::NUM;
const double Unsigned::FFT_MUL_COEFFICIENT = 4.4;
const double Unsigned::NEWTON_DIVMOD_COEFFICIENT = 11;
const double Unsigned::NEWTON_DIVMOD_POWER_2_RATIO = 0.4;

class Int {
 private:
  Unsigned a;
  bool neg;
  void update();

 public:
  // comparison functions
  [[nodiscard]] int cmp(const Int &b) const;  //-1:<  0:=  1:>
  bool operator<(const Int &b) const { return cmp(b) < 0; }
  bool operator<=(const Int &b) const { return cmp(b) <= 0; }
  bool operator>(const Int &b) const { return cmp(b) > 0; }
  bool operator>=(const Int &b) const { return cmp(b) >= 0; }
  bool operator==(const Int &b) const { return cmp(b) == 0; }
  bool operator!=(const Int &b) const { return cmp(b) != 0; }

  // IO functions
  void inputnum(long long s);
  void inputnum(unsigned long long s);
  void inputnum(int s) { inputnum((long long)s); }
  void inputnum(unsigned s) { inputnum((unsigned long long)s); }
  void inputnum(short s) { inputnum((long long)s); }
  void inputnum(unsigned short s) { inputnum((unsigned long long)s); }
  void assign(const char *s, int len = -1);
  void assign(const std::string &s) { assign(s.c_str(), s.size()); }
  [[nodiscard]] std::string to_string() const;
  friend std::istream &operator>>(std::istream &cin, Int &b);
  friend std::ostream &operator<<(std::ostream &cout, const Int &b);

  // constructor functions
  Int() : neg(false) {}
  Int(const Unsigned &b, bool neg1 = false) : a(b), neg(neg1) {}
  Int(const Int &b) : a(b.a), neg(b.neg) {}
  Int(short s) { inputnum(s); }
  Int(unsigned short s) { inputnum(s); }
  Int(int s) { inputnum(s); }
  Int(unsigned s) { inputnum(s); }
  Int(long long s) { inputnum(s); }
  Int(unsigned long long s) { inputnum(s); }
  Int(const char *s) { assign(s); }
  Int(const std::string &s) { assign(s); }

  // operators
  Int &operator+=(const Int &b);
  Int operator+(const Int &b) const;
  const Int &operator+() const;
  Int &operator-=(const Int &b);
  Int operator-(const Int &b) const;
  Int operator-() const;
  Int &operator*=(const Int &b);
  Int operator*(const Int &b) const;
  Int &operator/=(const Int &b);
  Int operator/(const Int &b) const;
  Int &operator%=(const Int &b);
  Int operator%(const Int &b) const;
  bool operator!() const;
  Int &operator++();
  const Int operator++(int);
  Int &operator--();
  const Int operator--(int);

  // other operation functions
  Int divmod(const Int &b);
  Int &shift10_eq(int x);
  Int shift10(int x) const;
  friend Int pow(const Int &a, const Int &b);
  friend Int gcd(const Int &a, const Int &b);
  friend Int abs(const Int &a);

  // helper functions
  [[nodiscard]] bool is_zero() const { return a.is_zero(); }
  [[nodiscard]] bool is_odd() const { return a.is_odd(); }
  [[nodiscard]] bool is_neg() const { return neg; }
  [[nodiscard]] int num_digit() const { return a.num_digit(); }
  [[nodiscard]] int ctz() const { return a.ctz(); }

  // type conversion functions
  [[nodiscard]] unsigned long long to_ull() const { return a.to_ull(); }
  [[nodiscard]] long long to_ll() const;
  [[nodiscard]] bool to_bool() const { return !is_zero(); }
  [[nodiscard]] unsigned to_uint() const { return to_ull(); }
  [[nodiscard]] int to_int() const { return to_ll(); }
  [[nodiscard]] unsigned short to_ushort() const { return to_ull(); }
  [[nodiscard]] short to_short() const { return to_ll(); }
  [[nodiscard]] double to_double() const;
  [[nodiscard]] long double to_ldouble() const;
};

class Rational {
 private:
  Int a, b;
  void update();

 public:
  // comparison functions
  [[nodiscard]] int cmp(const Rational &r) const;  //-1:<  0:=  1:>
  bool operator<(const Rational &r) const { return cmp(r) < 0; }
  bool operator<=(const Rational &r) const { return cmp(r) <= 0; }
  bool operator>(const Rational &r) const { return cmp(r) > 0; }
  bool operator>=(const Rational &r) const { return cmp(r) >= 0; }
  bool operator==(const Rational &r) const { return a == r.a && b == r.b; }
  bool operator!=(const Rational &r) const { return a != r.a || b != r.b; }

  // IO functions
  void assign(const char *s, int len = -1);
  void assign(const std::string &s) { assign(s.c_str(), s.size()); }
  [[nodiscard]] std::string to_string() const;
  friend std::istream &operator>>(std::istream &cin, Rational &b);
  friend std::ostream &operator<<(std::ostream &cout, const Rational &b);

  // constructor functions
  Rational() : a(0), b(1) {}
  Rational(const Int &a) : a(a), b(1) {}
  Rational(const Int &a1, const Int &b1);
  Rational(const char *s) { assign(s); }
  Rational(const std::string &s) { assign(s); }

  // operators
  Rational operator+(const Rational &r) const;
  Rational &operator+=(const Rational &r) { return *this = *this + r; }
  const Rational &operator+() const { return *this; }
  Rational operator-(const Rational &r) const;
  Rational &operator-=(const Rational &r) { return *this = *this - r; }
  Rational operator-() const { return Rational(-a, b); }
  Rational operator*(const Rational &r) const;
  Rational &operator*=(const Rational &r) { return *this = *this * r; }
  Rational operator/(const Rational &r) const { return *this * r.reciprocal(); }
  Rational &operator/=(const Rational &r) { return *this = *this / r; }
  bool operator!() const { return is_zero(); }

  // other operation functions
  [[nodiscard]] Rational reciprocal() const;
  friend Int trunc(const Rational &r) { return r.a / r.b; }
  friend Int floor(const Rational &r);
  friend Int ceil(const Rational &r);
  friend Rational pow(const Rational &a, const Int &b);
  friend Rational abs(const Rational &r);

  // helper functions
  [[nodiscard]] bool is_zero() const { return a.is_zero(); }
  [[nodiscard]] bool is_neg() const { return a.is_neg(); }
  [[nodiscard]] const Int &numerator() const { return a; }
  [[nodiscard]] const Int &denominator() const { return b; }

  // type conversion functions
  [[nodiscard]] double to_double() const;
  [[nodiscard]] long double to_ldouble() const;
};

}  // namespace quartz
