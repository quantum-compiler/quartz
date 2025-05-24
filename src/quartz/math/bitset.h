#pragma once

#include <vector>

namespace quartz {

class Bitset;
struct BitsetHash {
 public:
  std::size_t operator()(const Bitset &x) const;
};
class Bitset {
 public:
  using value_t = unsigned long long;
  Bitset() = default;
  Bitset(size_t n);
  class reference {
   public:
    value_t *pos;
    value_t dig;
    reference() = default;
    reference(value_t *a, int x);
    operator bool() const;
    bool operator~() const;
    reference &operator=(bool x);
    reference &operator=(const reference &x);
    reference &flip();
  };
  void flip(int x);
  bool operator[](int x) const;
  reference operator[](int x);
  Bitset operator^(const Bitset &b) const;
  bool operator==(const Bitset &b) const;
  friend std::size_t BitsetHash::operator()(const Bitset &x) const;

 private:
  static const int LOGBLOCK = 6;
  static const int BLOCK = (1 << LOGBLOCK);
  static const int BLOCK1 = BLOCK - 1;
  static const value_t MASK = ((value_t)-1);
  std::vector<value_t> a;
};

}  // namespace quartz
