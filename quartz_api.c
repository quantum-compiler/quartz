#include "quartz_api.h"
#include "string.h"
#include <stdlib.h>
#include <stdio.h>

int opt_circuit(const char* cqasm, char* buffer, int buff_size, unsigned char* ecc_set_, long unsigned int ecc_set_size) {
  int res = opt_circuit_ (cqasm, buffer, buff_size, ecc_set_, ecc_set_size);
  printf("ecc size = %d", ecc_set_size);
  return res;
}


int preprocess(const char* cqasm, char* buffer, int buff_size) {
  int res = preprocess_ (cqasm, buffer, buff_size);
  return res;
}

long unsigned int load_eqset (const char* eqset_fn_, unsigned char** store) {
  return load_eqset_(eqset_fn_, store);
}
