#ifndef QUARTZ_H
#define QUARTZ_H

#ifdef __cplusplus
extern "C" {
#endif

int opt_circuit_ (const char* cqasm, char* buffer, int buff_size, unsigned char* ecc_set_, long unsigned int ecc_set_size);
int preprocess_ (const char* cqasm, char* buffer, int buff_size);
long unsigned int load_eqset_ (const char* eqset_fn_, unsigned char** store);

#ifdef __cplusplus
}
#endif

#endif