#include "gen_ecc_set.h"
#include "quartz/context/context.h"
#include "quartz/generator/generator.h"

#include <chrono>
#include <fstream>

using namespace quartz;

int main() {
  // gen_ecc_set({GateType::u1, GateType::u2, GateType::u3, GateType::cx,
  //              GateType::add},
  //             "IBM_3_3_", true, 3, 4, 3);
  // gen_ecc_set({GateType::h, GateType::cz}, "H_CZ_2_2_", false, 2, 0, 2);
  // for (int n = 5; n <= 8; n++) {
  //   std::string file_prefix = "Rigetti_";
  //   file_prefix += std::to_string(n);
  //   file_prefix += "_3_";
  //   gen_ecc_set({GateType::rx, GateType::rz, GateType::cz, GateType::add},
  //               file_prefix, true, 3, 2, n);
  // }
  // gen_ecc_set({GateType::u1, GateType::u2, GateType::u3, GateType::cx,
  //              GateType::add},
  //             "IBM_4_3_", true, 3, 4, 4);
  // for (int n = 1; n <= 8; n++) {
  //   for (int q = 1; q <= 4 - (n >= 7); q++) {
  //     std::string file_prefix = "Nam_";
  //     file_prefix += std::to_string(n);
  //     file_prefix += "_";
  //     file_prefix += std::to_string(q);
  //     file_prefix += "_";
  //     gen_ecc_set(
  //         {GateType::rz, GateType::h, GateType::cx, GateType::x,
  //         GateType::add}, file_prefix, true, q, 2, n);
  //   }
  // }
  gen_ecc_set({GateType::t, GateType::tdg, GateType::h, GateType::x,
               GateType::cx, GateType::add},
              "3_2_5_", true, true, 3, 0, 5);
  return 0;
}
