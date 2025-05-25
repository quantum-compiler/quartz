#include "bitset.h"

#include <cassert>

namespace quartz {

Bitset::Bitset(size_t n) { a.assign((n + BLOCK - 1) / BLOCK, 0); }

Bitset::reference::reference(Bitset::value_t *a, int x)
    : pos(a + (x >> LOGBLOCK)), dig(((value_t)1) << (x & BLOCK1)) {}
Bitset::reference::operator bool() const { return *pos & dig; }
bool Bitset::reference::operator~() const { return ~*pos & dig; }
Bitset::reference &Bitset::reference::operator=(bool x) {
  if (x)
    *pos |= dig;
  else
    *pos &= MASK ^ dig;
  return *this;
}
Bitset::reference &Bitset::reference::operator=(const Bitset::reference &x) {
  if (*x.pos & x.dig)
    *pos |= dig;
  else
    *pos &= MASK ^ dig;
  return *this;
}
Bitset::reference &Bitset::reference::flip() {
  *pos ^= dig;
  return *this;
}

void Bitset::flip(int x) { a[x >> LOGBLOCK] ^= ((value_t)1) << (x & BLOCK1); }
bool Bitset::operator[](int x) const {
  return (a[x >> LOGBLOCK] >> (x & BLOCK1)) & 1;
}
Bitset::reference Bitset::operator[](int x) { return {a.data(), x}; }
Bitset Bitset::operator^(const Bitset &b) const {
  assert(a.size() == b.a.size());
  Bitset result(a.size() * BLOCK);
  for (int i = 0; i < (int)a.size(); i++) {
    result.a[i] = a[i] ^ b.a[i];
  }
  return result;
}
bool Bitset::operator==(const Bitset &b) const {
  return a == b.a;  // std::vector<>::operator== does the job!
}

std::size_t BitsetHash::operator()(const Bitset &x) const {
  std::hash<size_t> hash_fn;
  std::size_t result = x.a.size();
  for (auto &val : x.a) {
    result = result * 17 + hash_fn(val);
  }
  return result;
}
}  // namespace quartz
