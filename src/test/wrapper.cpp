#include "wrapper.h"

#include "oracle.h"

#include <stdio.h>
rust::String optimize(rust::String s) { return optimize_(std::string(s)); }
// int main() {
//   int result = optimize();
//   printf("The result is: %d\n", result);
//   return 0;
// }