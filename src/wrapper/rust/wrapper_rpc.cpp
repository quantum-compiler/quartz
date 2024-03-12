#include "oracle.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "rpc/server.h"

#include <thread>
using namespace quartz;

int main(int argc, char **argv) {
  int port = atoi(argv[1]);
  auto supercontext = get_context_("Nam", 24,
                                   "/home/pengyul/quicr/soam/resources/"
                                   "Nam_4_3_complete_ECC_set.json");
  rpc::server srv(port);
  srv.bind("optimize", [supercontext](std::string my_circ) {
    return optimize_(my_circ, "Gate", 10, supercontext);
  });
  srv.run();

  return 0;
}