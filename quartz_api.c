#include "quartz_api.h"
#include "string.h"
#include <stdlib.h>
#include <stdio.h>

int opt_circuit(const char* cqasm, char* buffer, int buff_size, unsigned char* xfers_) {
  int res = opt_circuit_ (cqasm, buffer, buff_size, xfers_);
  return res;
}


int preprocess(const char* cqasm, char* buffer, int buff_size) {
  int res = preprocess_ (cqasm, buffer, buff_size);
  return res;
}

long unsigned int load_eqset (const char* eqset_fn_, unsigned char** store) {
  return load_eqset_(eqset_fn_, store);
}

long unsigned int load_greedy_xfers (const char* eqset_fn_, unsigned char** store) {
   return load_greedy_xfers_(eqset_fn_, store);
}
