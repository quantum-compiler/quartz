#include "oracle.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "rpc/server.h"

#include <thread>
using namespace quartz;

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <port> <gateset> <eccfile> <cost> <timeout> <nqubits>\n";
    return 1;
  }
  int port = atoi(argv[1]);
  float timeout = atof(argv[5]);
  int nqubits = atoi(argv[6]);
  rpc::server srv(port);
  auto supercontext = get_context_(argv[2], nqubits, argv[3]);
  std::cout << "Server started on port " << port << std::endl;
  srv.bind("optimize", [supercontext, timeout, argv](std::string my_circ) {
    return optimize_(my_circ, argv[4], timeout, supercontext);
  });
  srv.run();

  return 0;
}