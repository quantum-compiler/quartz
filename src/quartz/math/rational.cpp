// Adapted from Mingkuan Xu's template for competitive programming.

#include "rational.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace std;

namespace quartz {

static const double RAW_PI = std::acos(-1.0);

Complex &Complex::operator+=(const Complex &b) {
  re += b.re, im += b.im;
  return *this;
}
Complex Complex::operator+(const Complex &b) const {
  return Complex(re + b.re, im + b.im);
}
const Complex &Complex::operator+() const { return *this; }
Complex &Complex::operator-=(const Complex &b) {
  re -= b.re, im -= b.im;
  return *this;
}
Complex Complex::operator-(const Complex &b) const {
  return Complex(re - b.re, im - b.im);
}
Complex Complex::operator-() const { return Complex(-re, -im); }
Complex Complex::operator*(const Complex &b) const {
  return Complex(re * b.re - im * b.im, re * b.im + im * b.re);
}
Complex &Complex::operator*=(const Complex &b) { return *this = *this * b; }
Complex Complex::operator*(double b) const { return Complex(re * b, im * b); }
Complex &Complex::operator*=(double b) {
  re *= b, im *= b;
  return *this;
}
double Complex::len2() const { return re * re + im * im; }
Complex Complex::operator/(const Complex &b) const {
  double tmp = b.len2();
  return Complex((re * b.re + im * b.im) / tmp, (im * b.re - re * b.im) / tmp);
}
Complex &Complex::operator/=(const Complex &b) { return *this = *this / b; }
Complex Complex::operator/(double b) const { return Complex(re / b, im / b); }
Complex &Complex::operator/=(double b) {
  re /= b, im /= b;
  return *this;
}
double Complex::len() const { return sqrt(len2()); }
Complex Complex::conj() const { return Complex(re, -im); }
Complex &Complex::toconj() {
  im = -im;
  return *this;
}
Complex Complex::muli() const { return Complex(-im, re); }

void FFT::fft_core(Complex *x, Complex *wu) {
  for (int i = 0; i < n; i++)
    if (i < rev[i])
      swap(x[i], x[rev[i]]);
  for (int i = 1; i < n; i <<= 1) {
    Complex *w = wu + i;
    for (int j = 0; j < n; j += (i << 1)) {
      Complex *xj = x + j;
      for (int k = 0; k < i; k++) {
        Complex tmp1 = xj[k], tmp2 = xj[k + i] * w[k];
        xj[k] = tmp1 + tmp2;
        xj[k + i] = tmp1 - tmp2;
      }
    }
  }
}
FFT::FFT(int logn) : n(1 << logn) {
  logn--;
  rev = new int[n];
  wu1 = new Complex[n];
  wu2 = new Complex[n];
  rev[0] = 0;
  for (int i = 1; i < n; i++)
    rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << logn);
  for (int i = 0; i < (n >> 1); i++)
    wu1[i | (n >> 1)] =
        Complex(cos(RAW_PI * i / (n >> 1)), sin(RAW_PI * i / (n >> 1)));
  for (int i = (n >> 1) - 1; i >= 0; i--)
    wu1[i] = wu1[i << 1];
  for (int i = 0; i < n; i++)
    wu2[i] = wu1[i].conj();
}
void FFT::fft(Complex *x) { fft_core(x, wu1); }
void FFT::fft(Complex *x, Complex *y) {
  for (int i = 0; i < n; i++)
    x[i].im = y[i].re;
  fft_core(x, wu1);
  y[0] = Complex(x[0].im, 0);
  x[0].im = 0;
  y[n >> 1] = Complex(x[n >> 1].im, 0);
  x[n >> 1].im = 0;
  for (int i = 1; i < (n >> 1); i++) {
    Complex p = x[i], q = x[n - i];
    x[i] = (q.conj() + p) / 2;
    x[n - i] = (p.conj() + q) / 2;
    y[i] = (q.conj() - p).muli() / 2;
    y[n - i] = (p.conj() - q).muli() / 2;
  }
}
void FFT::ifft(Complex *x) {
  fft_core(x, wu2);
  for (int i = 0; i < n; i++)
    x[i] /= n;
}
void FFT::ifft(Complex *x, Complex *y)  // result: real numbers
{
  for (int i = 0; i < n; i++)
    x[i] += y[i].muli();
  fft_core(x, wu2);
  for (int i = 0; i < n; i++) {
    x[i] /= n;
    y[i] = Complex(x[i].im, 0);
    x[i].im = 0;
  }
}
std::vector<FFT *> Unsigned::ffts;
const int Unsigned::NUM;
const double Unsigned::FFT_MUL_COEFFICIENT = 4.4;
const double Unsigned::NEWTON_DIVMOD_COEFFICIENT = 11;
const double Unsigned::NEWTON_DIVMOD_POWER_2_RATIO = 0.4;

void Unsigned::update() {
  while (!a.back() && a.size() > 1)
    a.pop_back();
}
double Unsigned::force_mul_estimate_time(const Unsigned &b) const {
  return (double)a.size() * b.a.size();
}
Unsigned Unsigned::force_mul(const Unsigned &b) const {
  Unsigned ret;
  ret.a.clear();
  const int asize = a.size(), bsize = b.a.size();
  const int retsize = asize + bsize;
  ret.a.reserve(retsize);
  unsigned long long next = 0;
  for (int i = 0; i < retsize - 1; i++) {
    const int jmn = max(0, i - bsize + 1), jmx = min(i, asize - 1);
    unsigned long long now = next;
    next = 0;
    for (int j = jmn; j <= jmx; j++)
      if ((now += (unsigned long long)a[j] * b.a[i - j]) >= MAXNUM)
        now -= MAXNUM, next += MAXNUM / NUM;
    ret.a.push_back(now % NUM);
    next += now / NUM;
  }
  if (next)
    ret.a.push_back(next);
  return ret;
}
double Unsigned::fft_mul_estimate_time(const Unsigned &b) const {
  int tmp = (a.size() + b.a.size()) * FFT_NUMD;
  int ret = log(tmp) / log(2);
  while ((1 << ret) < tmp)
    ret++;
  ret *= (1 << ret);
  return ret * FFT_MUL_COEFFICIENT;
}
Unsigned Unsigned::fft_mul(const Unsigned &b) const {
  const int asize = a.size(), bsize = b.a.size();
  const int retsize = asize + bsize;
  int tmp = log(retsize * FFT_NUMD) / log(2);
  while ((1 << tmp) < (retsize * FFT_NUMD))
    tmp++;
  if ((int)ffts.size() <= tmp)
    ffts.resize(tmp + 1);
  if (!ffts[tmp])
    ffts[tmp] = new FFT(tmp);
  Complex *x = new Complex[1 << tmp];
  Complex *y = new Complex[1 << tmp];
  memset(x, 0, sizeof(Complex) << tmp);
  memset(y, 0, sizeof(Complex) << tmp);
  for (int i = 0; i < asize; i++) {
    int now = a[i];
    for (int j = 0; j < FFT_NUMD; j++) {
      x[i * FFT_NUMD + j].re = now % FFT_NUM;
      now /= FFT_NUM;
    }
  }
  for (int i = 0; i < bsize; i++) {
    int now = b.a[i];
    for (int j = 0; j < FFT_NUMD; j++) {
      y[i * FFT_NUMD + j].re = now % FFT_NUM;
      now /= FFT_NUM;
    }
  }
  ffts[tmp]->fft(x, y);
  for (int i = 0; i < (1 << tmp); i++)
    x[i] *= y[i];
  ffts[tmp]->ifft(x);
  Unsigned ret;
  ret.a.assign(retsize, 0);
  unsigned long long now = 0;
  for (int i = 0; i < retsize; i++) {
    for (int j = 0, mul = 1; j < FFT_NUMD; j++, mul *= FFT_NUM)
      now += (unsigned long long)(x[i * FFT_NUMD + j].re + 0.5) * mul;
    ret.a[i] = now % NUM;
    now /= NUM;
  }
  ret.update();
  delete[] x;
  delete[] y;
  return ret;
}
double Unsigned::force_divmod_estimate_time(const Unsigned &b) const {
  return (double)(b.a.size() + 1) * (a.size() - b.a.size() + 1);
}
Unsigned Unsigned::force_divmod(const Unsigned &b) {
  Unsigned ret;
  const int asize = a.size(), bsize = b.a.size();
  const int retsize = asize - bsize + 1;
  if (retsize <= 0)
    return Unsigned(0);
  a.push_back(0);
  ret.a.assign(retsize, 0);
  for (int i = retsize - 1; i >= 0; i--) {
    int mul;
    if (bsize == 1)
      mul = ((long long)a[i + bsize] * NUM + a[i + bsize - 1]) / b.a[bsize - 1];
    else {
      mul = ((long long)a[i + bsize] * NUM + a[i + bsize - 1]) /
            (b.a[bsize - 1] + 1);
      mul += (((long long)a[i + bsize] * NUM + a[i + bsize - 1] -
               (long long)b.a[bsize - 1] * mul) *
                  NUM +
              a[i + bsize - 2] - (long long)(b.a[bsize - 2] + 1) * mul) /
             ((long long)b.a[bsize - 1] * NUM + b.a[bsize - 2] + 1);
    }
    if (mul) {
      ret.a[i] = mul;
      int next = 0;
      for (int j = 0; j < bsize; j++) {
        long long now = next + a[i + j] - (long long)b.a[j] * mul;
        next = now < 0 ? (now + 1) / NUM - 1 : 0;
        a[i + j] = now - (long long)next * NUM;
      }
      a[i + bsize] += next;
    }
    if (bsize > 1) {
      bool flag = true;
      for (int j = bsize - 1; j >= 0; j--)
        if (a[i + j] > b.a[j])
          break;
        else if (a[i + j] < b.a[j]) {
          flag = false;
          break;
        }
      if (flag) {
        ret.a[i]++;
        for (int j = 0; j < bsize; j++)
          if ((a[i + j] -= b.a[j]) < 0)
            a[i + j] += NUM, a[i + j + 1]--;
      }
    }
  }
  update();
  ret.update();
  return ret;
}
double Unsigned::newton_divmod_estimate_time(const Unsigned &b) const {
  int tmp = (max(a.size() - b.a.size(), b.a.size()) + 1) * FFT_NUMD * 2;
  int ret = log(tmp) / log(2);
  while ((1 << ret) < tmp)
    ret++;
  ret *= (1 << ret);
  return (ret * NEWTON_DIVMOD_POWER_2_RATIO +
          tmp * log(tmp) / log(2) * (1 - NEWTON_DIVMOD_POWER_2_RATIO)) *
         NEWTON_DIVMOD_COEFFICIENT;
}
Unsigned Unsigned::rev_10(int n) const {
  int l = num_digit();
  if (Unsigned(1).shift10(n).force_divmod_estimate_time(*this) <
      Unsigned(1).shift10(n).newton_divmod_estimate_time(*this))
    return Unsigned(1).shift10(n) / *this;
  Unsigned ret;
  if ((n - l) >= l * 2) {
    int m = l + (n - l) / 2 + 1;
    Unsigned prev = rev_10(m);
    ret = (prev * 2).shift10(n - m) - (*this * prev * prev).shift10(n - 2 * m);
  } else {
    int k = max(l, n - l) / 2 + 1;
    Unsigned prev = shift10(k - l).rev_10(k * 2);
    ret = (prev * 2).shift10(n - l - k) -
          (prev * prev * *this).shift10(n - 2 * l - 2 * k);
  }
  Unsigned tmp = ret * *this;
  if (tmp > Unsigned(1).shift10(n))
    ret -= ((tmp - Unsigned(1).shift10(n) - 1) / *this + 1);
  else
    ret += (Unsigned(1).shift10(n) - tmp) / *this;
  return ret;
}
bool Unsigned::is_power_of_10() const {
  for (int i = 0; i < (int)a.size() - 1; i++)
    if (a[i])
      return false;
  for (int i = 1; i <= a.back(); i *= 10)
    if (i == a.back())
      return true;
  return false;
}
Unsigned Unsigned::newton_divmod(const Unsigned &b) {
  if (is_power_of_10()) {
    Unsigned ret = b.rev_10(num_digit() - 1);
    *this -= ret * b;
    return ret;
  }
  int n = num_digit() + 2;
  Unsigned ret = (b.rev_10(n) * *this).shift10(-n);
  Unsigned tmp = ret * b;
  if (tmp > *this) {
    ret -= ((tmp -= (*this + 1)).divmod(b) + 1);
    *this = b - tmp - 1;
  } else
    ret += (*this -= tmp).divmod(b);
  return ret;
}
int Unsigned::cmp(const Unsigned &b) const  //-1:<  0:=  1:>
{
  if (a.size() != b.a.size())
    return a.size() < b.a.size() ? -1 : 1;
  for (int i = a.size() - 1; i >= 0; i--)
    if (a[i] != b.a[i])
      return a[i] < b.a[i] ? -1 : 1;
  return 0;
}
void Unsigned::inputnum(unsigned long long int s) {
  if (!s) {
    a.assign(1, 0);
    return;
  }
  a.clear();
  while (s) {
    a.push_back(s % NUM);
    s /= NUM;
  }
}
void Unsigned::assign(const char *s, int len) {
  if (len == -1)
    len = strlen(s);
  const int l = (len - 1) / NUM_DIGIT + 1;
  a.assign(l, 0);
  for (int i = 0; i < len; i++)
    a[(len - i - 1) / NUM_DIGIT] =
        a[(len - i - 1) / NUM_DIGIT] * 10 + s[i] - '0';
}
void Unsigned::assign(const string &s) { assign(s.c_str(), s.size()); }
string Unsigned::to_string() const {
  string ret;
  ret.reserve(a.size() * NUM_DIGIT);
  int tmp = a.back();
  int tmp2 = 1;
  while (tmp2 * 10 <= tmp)
    tmp2 *= 10;
  while (tmp2) {
    const int tmp0 = tmp / tmp2;
    tmp -= tmp0 * tmp2;
    ret += tmp0 + '0';
    tmp2 /= 10;
  }
  for (int i = a.size() - 2; i >= 0; i--) {
    tmp = a[i];
    for (int j = NUM / 10; j; j /= 10) {
      const int tmp0 = tmp / j;
      tmp -= tmp0 * j;
      ret += tmp0 + '0';
    }
  }
  return ret;
}
istream &operator>>(istream &cin, Unsigned &b) {
  string s;
  cin >> s;
  b.assign(s);
  return cin;
}
ostream &operator<<(ostream &cout, const Unsigned &b) {
  cout << b.to_string();
  return cout;
}
Unsigned &Unsigned::operator+=(const Unsigned &b) {
  const int asize = a.size(), bsize = b.a.size();
  int i;
  if (asize > bsize) {
    for (i = 0; i < bsize; i++)
      if ((a[i] += b.a[i]) >= NUM)
        a[i] -= NUM, a[i + 1]++;
    while (a[i] >= NUM) {
      a[i] -= NUM;
      if (i == asize - 1) {
        a.push_back(1);
        break;
      }
      a[++i]++;
    }
  } else {
    a.reserve(bsize + 1);
    a.resize(bsize, 0);
    for (i = 0; i < bsize - 1; i++)
      if ((a[i] += b.a[i]) >= NUM)
        a[i] -= NUM, a[i + 1]++;
    if ((a[bsize - 1] += b.a[bsize - 1]) >= NUM) {
      a[bsize - 1] -= NUM;
      a.push_back(1);
    }
  }
  return *this;
}
Unsigned Unsigned::operator+(const Unsigned &b) const {
  if (a.size() > b.a.size()) {
    Unsigned ret(*this);
    return ret += b;
  } else {
    Unsigned ret(b);
    return ret += *this;
  }
}
const Unsigned &Unsigned::operator+() const { return *this; }
Unsigned &Unsigned::operator-=(const Unsigned &b) {
  const int bsize = b.a.size();
  int i;
  for (i = 0; i < bsize; i++)
    if ((a[i] -= b.a[i]) < 0)
      a[i] += NUM, a[i + 1]--;
  while (i < (int)a.size() && a[i] < 0) {
    a[i] += NUM;
    a[++i]--;
  }
  update();
  return *this;
}
Unsigned Unsigned::operator-(const Unsigned &b) const {
  Unsigned ret(*this);
  return ret -= b;
}
Unsigned Unsigned::operator*(const Unsigned &b) const {
  if (is_zero() || b.is_zero())
    return Unsigned();
  int tmp1, tmp2;
  if ((tmp1 = ctz()) + (tmp2 = b.ctz()) > 0)
    return (shift10(-tmp1) * b.shift10(-tmp2)).shift10(tmp1 + tmp2);
  if (*this == Unsigned(1))
    return b;
  if (b == Unsigned(1))
    return *this;

  if (fft_mul_estimate_time(b) < force_mul_estimate_time(b))
    return fft_mul(b);
  else
    return force_mul(b);
}
Unsigned &Unsigned::operator*=(const Unsigned &b) { return *this = *this * b; }
Unsigned &Unsigned::operator/=(const Unsigned &b) { return *this = divmod(b); }
Unsigned Unsigned::operator/(const Unsigned &b) const {
  Unsigned ret(*this);
  return ret /= b;
}
Unsigned &Unsigned::operator%=(const Unsigned &b) {
  divmod(b);
  return *this;
}
Unsigned Unsigned::operator%(const Unsigned &b) const {
  Unsigned ret(*this);
  return ret %= b;
}
bool Unsigned::operator!() const { return is_zero(); }
Unsigned &Unsigned::operator++() { return *this += Unsigned(1); }
const Unsigned Unsigned::operator++(int) {
  Unsigned ret(*this);
  *this += Unsigned(1);
  return ret;
}
Unsigned &Unsigned::operator--() { return *this -= Unsigned(1); }
const Unsigned Unsigned::operator--(int) {
  Unsigned ret(*this);
  *this -= Unsigned(1);
  return ret;
}
Unsigned Unsigned::divmod(const Unsigned &b) {
  assert(!b.is_zero());
  if (*this < b)
    return Unsigned();
  int tmp;
  if ((tmp = b.ctz()) > 0) {
    if (tmp / NUM_DIGIT + 1 == (int)b.a.size() &&
        b.a.back() == std::round(std::pow(10.0, tmp % NUM_DIGIT))) {
      Unsigned ret = shift10(-tmp);
      *this -= ret.shift10(tmp);
      return ret;
    }
    if ((tmp = min(tmp, ctz())) > 0) {
      Unsigned ret = shift10_eq(-tmp).divmod(b.shift10(-tmp));
      shift10_eq(tmp);
      return ret;
    }
  }
  if (newton_divmod_estimate_time(b) < force_divmod_estimate_time(b))
    return newton_divmod(b);
  else
    return force_divmod(b);
}
Unsigned &Unsigned::shift10_eq(int x) {
  if (x <= 0) {
    x = -x;
    if (num_digit() - x <= 0)
      return *this = Unsigned();
    a.erase(a.begin(), a.begin() + x / NUM_DIGIT);
    if (x % NUM_DIGIT != 0) {
      int tmp = 1;
      for (int i = 1; i <= x % NUM_DIGIT; i++)
        tmp *= 10;
      a.push_back(0);
      for (int i = 0; i < (int)a.size() - 1; i++)
        a[i] = a[i + 1] % tmp * (NUM / tmp) + a[i] / tmp;
      update();
    }
    return *this;
  } else {
    a.insert(a.begin(), (x - 1) / NUM_DIGIT + 1, 0);
    if (x % NUM_DIGIT != 0) {
      int tmp = 1;
      for (int i = 1; i <= NUM_DIGIT - x % NUM_DIGIT; i++)
        tmp *= 10;
      a.push_back(0);
      for (int i = 0; i < (int)a.size() - 1; i++)
        a[i] = a[i + 1] % tmp * (NUM / tmp) + a[i] / tmp;
      update();
    }
    return *this;
  }
}
Unsigned Unsigned::shift10(int x) const {
  if (x <= 0) {
    x = -x;
    if (num_digit() - x <= 0)
      return Unsigned();
    Unsigned ret;
    ret.a.assign(a.begin() + x / NUM_DIGIT, a.end());
    if (x % NUM_DIGIT != 0) {
      int tmp = 1;
      for (int i = 1; i <= x % NUM_DIGIT; i++)
        tmp *= 10;
      ret.a.push_back(0);
      for (int i = 0; i < (int)ret.a.size() - 1; i++)
        ret.a[i] = ret.a[i + 1] % tmp * (NUM / tmp) + ret.a[i] / tmp;
      ret.update();
    }
    return ret;
  } else {
    Unsigned ret;
    ret.a = a;
    ret.a.insert(ret.a.begin(), (x - 1) / NUM_DIGIT + 1, 0);
    if (x % NUM_DIGIT != 0) {
      int tmp = 1;
      for (int i = 1; i <= NUM_DIGIT - x % NUM_DIGIT; i++)
        tmp *= 10;
      ret.a.push_back(0);
      for (int i = 0; i < (int)ret.a.size() - 1; i++)
        ret.a[i] = ret.a[i + 1] % tmp * (NUM / tmp) + ret.a[i] / tmp;
      ret.update();
    }
    return ret;
  }
}
Unsigned Unsigned::random() const {
  auto randint = []() { return ((rand() & 32767) << 15) | (rand() & 32767); };
  auto randdouble = [&randint]() { return (double)randint() / (1 << 30); };
  Unsigned ret = *this;
  bool mx = true;
  for (int i = (int)ret.a.size() - 1; i >= 0; i--) {
    if (mx) {
      if ((randint() % (a[i] + 1) == 0) &&
          (i > 0 ? randdouble() < (double)a[i - 1] / NUM : true)) {
        ret.a[i] = a[i];
      } else {
        ret.a[i] = randint() % a[i];
        mx = false;
      }
    } else {
      ret.a[i] = randint() % NUM;
    }
  }
  return ret;
}
Unsigned pow(const Unsigned &a, const Unsigned &b) {
  if ((a.a.size() == 1 && a.a[0] == 1) || b.is_zero())
    return Unsigned(1);
  if (a.is_zero())
    return Unsigned();
  int p = b.to_int();
  if (a == Unsigned(10))
    return Unsigned(1).shift10(p);
  Unsigned ret(1), mul(a);
  if (p & 1)
    ret = a;
  for (p >>= 1; p; p >>= 1) {
    mul *= mul;
    if (p & 1)
      ret *= mul;
  }
  return ret;
}
Unsigned gcd(const Unsigned &a, const Unsigned &b) {
  return !b ? a : gcd(b, a % b);
}
Unsigned abs(const Unsigned &a) { return a; }
bool Unsigned::is_zero() const { return a.size() == 1 && !a[0]; }
bool Unsigned::is_odd() const { return a[0] & 1; }
bool Unsigned::is_neg() const { return false; }
int Unsigned::num_digit() const {
  int ret = (a.size() - 1) * NUM_DIGIT;
  for (int i = a.back(); i; i /= 10, ret++)
    ;
  return ret;
}
int Unsigned::ctz() const {
  for (int i = 0; i < (int)a.size(); i++)
    if (a[i]) {
      int ret = 0;
      for (int j = a[i]; j % 10 == 0; j /= 10, ret++)
        ;
      return i * NUM_DIGIT + ret;
    }
  return -1;
}
unsigned long long Unsigned::to_ull() const {
  unsigned long long ret = 0;
  for (int i = a.size() - 1; i >= 0; i--)
    ret = ret * NUM + a[i];
  return ret;
}
bool Unsigned::to_bool() const { return !is_zero(); }
long long Unsigned::to_ll() const { return to_ull(); }
unsigned Unsigned::to_uint() const { return to_ull(); }
int Unsigned::to_int() const { return to_ull(); }
unsigned short Unsigned::to_ushort() const { return to_ull(); }
short Unsigned::to_short() const { return to_ull(); }
double Unsigned::to_double() const {
  double ret = 0, tmp;
  for (int i = a.size() - 1;
       i >= 0 && (tmp = ret + a[i] * std::pow((double)NUM, i)) != ret; i--)
    ret = tmp;
  return ret;
}
long double Unsigned::to_ldouble() const {
  long double ret = 0, tmp;
  for (int i = a.size() - 1;
       i >= 0 && (tmp = ret + a[i] * std::pow((long double)NUM, i)) != ret; i--)
    ret = tmp;
  return ret;
}

void Int::update() {
  if (is_zero())
    neg = false;  // change "-0" to "0"
}
int Int::cmp(const Int &b) const {
  if (neg != b.neg)
    return neg ? -1 : 1;
  return (neg ? -1 : 1) * a.cmp(b.a);
}
void Int::inputnum(long long int s) {
  if (s < 0)
    neg = true, s = -s;
  else
    neg = false;
  a.inputnum(s);
}
void Int::inputnum(unsigned long long int s) {
  neg = false;
  a.inputnum(s);
}
void Int::assign(const char *s, int len) {
  neg = false;
  if (*s == '-' || *s == '+') {
    neg = (*s == '-');
    if (len != -1)
      len--;
    a.assign(s + 1, len);
  } else
    a.assign(s, len);
}
std::string Int::to_string() const {
  if (neg)
    return '-' + a.to_string();
  else
    return a.to_string();
}
std::istream &operator>>(istream &cin, Int &b) {
  std::string s;
  cin >> s;
  b.assign(s);
  return cin;
}
std::ostream &operator<<(ostream &cout, const Int &b) {
  cout << b.to_string();
  return cout;
}
Int &Int::operator+=(const Int &b) {
  if (neg != b.neg) {
    if (a >= b.a) {
      a -= b.a;
      update();
    } else {
      a = b.a - a;
      neg = !neg;
    }
  } else
    a += b.a;
  return *this;
}
Int Int::operator+(const Int &b) const {
  Int ret(*this);
  ret += b;
  return ret;
}
const Int &Int::operator+() const { return *this; }
Int &Int::operator-=(const Int &b) {
  if (neg == b.neg) {
    if (a >= b.a) {
      a -= b.a;
      update();
    } else {
      a = b.a - a;
      neg = !neg;
    }
  } else
    a += b.a;
  return *this;
}
Int Int::operator-(const Int &b) const {
  Int ret(*this);
  return ret -= b;
}
Int Int::operator-() const {
  Int ret(*this);
  if (!is_zero())
    ret.neg = !ret.neg;
  return ret;
}
Int &Int::operator*=(const Int &b) {
  a *= b.a;
  if (b.neg)
    neg = !neg;
  update();
  return *this;
}
Int Int::operator*(const Int &b) const {
  Int ret(*this);
  return ret *= b;
}
Int &Int::operator/=(const Int &b) { return *this = divmod(b); }
Int Int::operator/(const Int &b) const {
  Int ret(*this);
  return ret /= b;
}
Int &Int::operator%=(const Int &b) {
  divmod(b);
  return *this;
}
Int Int::operator%(const Int &b) const {
  Int ret(*this);
  return ret %= b;
}
bool Int::operator!() const { return is_zero(); }
Int &Int::operator++() { return *this += Int(1); }
const Int Int::operator++(int) {
  Int ret(*this);
  *this += Int(1);
  return ret;
}
Int &Int::operator--() { return *this -= Int(1); }
const Int Int::operator--(int) {
  Int ret(*this);
  *this -= Int(1);
  return ret;
}
Int Int::divmod(const Int &b) {
  Int ret(a.divmod(b.a));
  if (neg != b.neg)
    ret.neg = !ret.neg;
  ret.update();
  update();
  return ret;
}
Int &Int::shift10_eq(int x) {
  a.shift10_eq(x);
  return *this;
}
Int Int::shift10(int x) const { return Int(a.shift10(x), neg); }
Int pow(const Int &a, const Int &b) {
  if (b.neg)
    return Int(0);
  Int ret(pow(a.a, b.a));
  if (a.neg && b.is_odd())
    ret.neg = true;
  return ret;
}
Int gcd(const Int &a, const Int &b) { return gcd(a.a, b.a); }
Int abs(const Int &a) { return a.a; }
long long Int::to_ll() const {
  long long ret = a.to_ll();
  if (neg)
    ret = -ret;
  return ret;
}
double Int::to_double() const {
  double ret = a.to_double();
  if (neg)
    ret = -ret;
  return ret;
}
long double Int::to_ldouble() const {
  long double ret = a.to_ldouble();
  if (neg)
    ret = -ret;
  return ret;
}

void Rational::update() {
  Int tmp = gcd(a, b);
  a /= tmp;
  b /= tmp;
}
int Rational::cmp(const Rational &r) const {
  if (is_neg() != r.is_neg())
    return is_neg() ? -1 : 1;
  if (is_zero())
    return r.is_zero() ? 0 : -1;
  if (r.is_zero())
    return 1;
  int tmp1 = a.num_digit() + r.b.num_digit(),
      tmp2 = b.num_digit() + r.a.num_digit();
  if (tmp1 >= tmp2 + 2)
    return is_neg() ? -1 : 1;
  if (tmp2 >= tmp1 + 2)
    return is_neg() ? 1 : -1;
  return (a * r.b).cmp(b * r.a);
}
void Rational::assign(const char *s, int len) {
  if (!s) {
    a.inputnum(0);
    b.inputnum(1);
    return;
  }
  if (len == -1)
    len = (int)strlen(s);
  int div_loc = -1;
  int decimal_point_loc = -1;
  int e_loc = -1;
  for (int i = 0; i < len; i++) {
    if (s[i] == '/') {
      div_loc = i;
      break;
    }
    if (s[i] == '.') {
      decimal_point_loc = i;
    }
    if (s[i] == 'e' || s[i] == 'E') {
      e_loc = i;
    }
  }
  if (div_loc == -1) {
    if (decimal_point_loc == -1) {
      a.assign(s, len);
      b.inputnum(1);
    } else {
      string s_without_point(s, decimal_point_loc);
      int b_pow = len - decimal_point_loc - 1;
      if (e_loc == -1) {
        s_without_point +=
            string(s + decimal_point_loc + 1, len - decimal_point_loc - 1);
      } else {
        // [s_without_point].[remaining]e[a_pow]
        s_without_point +=
            string(s + decimal_point_loc + 1, e_loc - decimal_point_loc - 1);
        int a_pow = stoi(string(s + e_loc + 1, len - e_loc - 1));
        if (b_pow >= a_pow) {
          b_pow -= a_pow;
        } else {
          s_without_point += string(a_pow - b_pow, '0');
          b_pow = 0;
        }
      }
      a.assign(s_without_point);
      b = pow(Int(10), b_pow);
      update();
    }
  } else {
    a.assign(s, div_loc);
    b.assign(s + div_loc + 1, len - div_loc - 1);
  }
}
std::string Rational::to_string() const {
  if (b == Int(1))
    return a.to_string();
  return a.to_string() + '/' + b.to_string();
}
std::istream &operator>>(istream &cin, Rational &b) {
  std::string s;
  int ch;
  while (cin) {
    ch = cin.get();
    if (!isspace(ch)) {
      cin.unget();
      break;
    }
  }
  while (cin) {
    ch = cin.get();
    // Only supports the following formats:
    // a, a/b, a.b, a.bec
    // where a and c are integers, b is a positive integer
    if (isdigit(ch) || ch == '.' || ch == '-' || ch == '/' || ch == 'e' ||
        ch == 'E' || ch == '+') {
      s += (char)ch;
    } else {
      cin.unget();
      break;
    }
  }
  b.assign(s);
  return cin;
}
std::ostream &operator<<(ostream &cout, const Rational &b) {
  cout << b.to_string();
  return cout;
}
Rational::Rational(const Int &a1, const Int &b1) : a(a1), b(b1) {
  if (b.is_neg())
    a = -a, b = -b;
  update();
}
Rational Rational::operator+(const Rational &r) const {
  Int tmp = gcd(b, r.b);
  Int tmp1 = b / tmp, tmp2 = r.b / tmp;
  Rational ret(a * tmp2 + r.a * tmp1, tmp);
  ret.update();
  if (!ret.is_zero())
    ret.b *= tmp1 * tmp2;
  return ret;
}
Rational Rational::operator-(const Rational &r) const {
  Int tmp = gcd(b, r.b);
  Int tmp1 = b / tmp, tmp2 = r.b / tmp;
  Rational ret(a * tmp2 - r.a * tmp1, tmp);
  ret.update();
  if (!ret.is_zero())
    ret.b *= tmp1 * tmp2;
  return ret;
}
Rational Rational::operator*(const Rational &r) const {
  Int tmp1 = gcd(a, r.b);
  Int tmp2 = gcd(b, r.a);
  return Rational((a / tmp1) * (r.a / tmp2), (b / tmp2) * (r.b / tmp1));
}
Rational Rational::reciprocal() const {
  if (is_zero())
    return Rational();
  if (is_neg())
    return Rational(-b, -a);
  return Rational(b, a);
}
Int floor(const Rational &r) {
  if (r.is_neg())
    return (r.a + 1) / r.b - 1;
  return r.a / r.b;
}
Int ceil(const Rational &r) {
  if (!r.is_neg() && !r.is_zero())
    return (r.a - 1) / r.b + 1;
  return r.a / r.b;
}
Int round(const Rational &r) {
  if (r.is_neg())
    return (r.a - r.b / 2) / r.b;
  return (r.a + r.b / 2) / r.b;
}
Rational pow(const Rational &a, const Int &b) {
  if (b.is_neg())
    return pow(a.reciprocal(), -b);
  return Rational(pow(a.a, b), pow(a.b, b));
}
Rational abs(const Rational &r) { return Rational(abs(r.a), r.b); }
double Rational::to_double() const { return a.to_double() / b.to_double(); }
long double Rational::to_ldouble() const {
  return a.to_ldouble() / b.to_ldouble();
}
Rational::operator long long() const {
  assert(b == Int(1));
  return a.to_ll();
}
}  // namespace quartz
