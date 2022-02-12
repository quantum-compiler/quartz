#include "arb.h"

// Code example on https://github.com/fredrik-johansson/arb

int main() {
  slong prec;
  arb_t x, y;
  arb_init(x);
  arb_init(y);

  for (prec = 64;; prec *= 2) {
    arb_const_pi(x, prec);
    arb_set_si(y, -10000);
    arb_exp(y, y, prec);
    arb_add(x, x, y, prec);
    arb_sin(y, x, prec);
    arb_printn(y, 15, 0);
    printf("\n");
    if (arb_rel_accuracy_bits(y) >= 53)
      break;
  }

  arb_clear(x);
  arb_clear(y);
  flint_cleanup();
}
