#include "wrapper.h"

#include "oracle.h"

#include <stdio.h>
rust::String optimize(rust::String circ_string, rust::String cost_func,
                      rust::String ecc_path, rust::String gateset,
                      float timeout)
{
    return optimize_(std::string(circ_string), std::string(cost_func),
                     std::string(ecc_path), std::string(gateset), timeout);
}
// int main() {
//   int result = optimize();
//   printf("The result is: %d\n", result);
//   return 0;
// }