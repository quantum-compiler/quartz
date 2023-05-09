#ifndef QUARTZ_H
#define QUARTZ_H

#ifdef __cplusplus
extern "C" {
#endif

int opt_circuit_ (const char* cqasm, int timeout, char* buffer, int buff_size, unsigned char* xfers_);
int preprocess_ (const char* cqasm, char* buffer, int buff_size);
long unsigned int load_eqset_ (const char* eqset_fn_, unsigned char** store);
long unsigned int load_greedy_xfers_ (const char* eqset_fn_, unsigned char** store);
void load_xfers_ (const char* eqset_fn_,
  unsigned char** gstore, long unsigned int* glen,
  unsigned char** allstore, long unsigned int* alen);

#ifdef __cplusplus
}
#endif

#endif