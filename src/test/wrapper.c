#include "wrapper.h"
#include "oracle.h"
#include <stdio.h>
int optimize() {
  int result = optimize_();
  return result;
}
int main() {
  int result = optimize();
  printf("The result is: %d\n", result);
  return 0;
}