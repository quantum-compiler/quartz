#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "rpc/server.h"
#include "wrapper/oracle.h"

#include <thread>
using namespace quartz;

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <port> <optimization_gateset> <eccfile> <cost> "
                 "<timeout_type> <timeout_value>";
    return 1;
  }
  int port = atoi(argv[1]);
  std::string optimization_gateset = argv[2];
  std::string eccfile = argv[3];
  std::string cost = argv[4];
  std::string timeout_type = argv[5];
  float timeout_value = atof(argv[6]);
  rpc::server srv(port);

  auto supercontext = get_context_(optimization_gateset, 1, eccfile);
  srv.bind("optimize", [supercontext, cost, timeout_type,
                        timeout_value](std::string my_circ) {
    return optimize_(my_circ, cost, timeout_type, timeout_value, supercontext);
  });
  srv.bind("rotation_merging",
           [](std::string my_circ) { return rotation_merging_(my_circ); });
  srv.bind("clifford_decomposition", [](std::string my_circ) {
    return clifford_decomposition_(my_circ);
  });
  // std::cout << "Server started on port " << port << std::endl;
  srv.run();

  return 0;
}