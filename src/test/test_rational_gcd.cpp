#include "quartz/math/rational.h"

#include <iostream>

using namespace quartz;

// Shifting a zero left by a whole number of limbs must leave a normalised
// zero. divmod's trailing-zero shortcut does this to the remainder, and an
// unnormalised zero made gcd recurse until the stack overflowed.
int main() {
  bool all_ok = true;
  Unsigned zero;
  zero.shift10_eq(9);
  if (!zero.is_zero()) {
    std::cout << "0 * 10^9 is not a normalised zero: " << zero << std::endl;
    all_ok = false;
  }
  Unsigned r = Unsigned("1250000000000") % Unsigned("625000000000");
  if (!r.is_zero()) {
    std::cout << "1250000000000 % 625000000000 = " << r << ", expected 0"
              << std::endl;
    all_ok = false;
  }
  Unsigned g = gcd(Unsigned("1875000000000"), Unsigned("1250000000000"));
  if (!(g == Unsigned("625000000000"))) {
    std::cout << "gcd(1875000000000, 1250000000000) = " << g
              << ", expected 625000000000" << std::endl;
    all_ok = false;
  }
  std::cout << (all_ok ? "All rational checks passed." : "FAILED") << std::endl;
  return all_ok ? 0 : 1;
}
