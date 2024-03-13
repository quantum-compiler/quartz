#include "oracle.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
using namespace quartz;

int main() {
  auto supercontext = get_context_(
      "Nam", 48,
      "/home/pengyul/quicr/soam/resources/Nam_4_3_complete_ECC_set.json");

  return 0;
}