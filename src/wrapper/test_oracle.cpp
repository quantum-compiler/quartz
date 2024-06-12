#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "wrapper/oracle.h"

#include <thread>
using namespace quartz;

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <circfile> <optimization_gateset> <eccfile> <cost> "
                 "<timeout_type> <timeout_value>";
    return 1;
  }
  std::string optimization_gateset = argv[2];
  std::string eccfile = argv[3];
  std::string cost = argv[4];
  std::string timeout_type = argv[5];
  std::string circ_path = argv[1];
  std::ifstream circ_file(circ_path);
  std::string my_circ((std::istreambuf_iterator<char>(circ_file)),
                      std::istreambuf_iterator<char>());
  float timeout_value = atof(argv[6]);

  auto supercontext = get_context_(optimization_gateset, 1, eccfile);
  std::string new_circ =
      optimize_(my_circ, cost, timeout_type, timeout_value, supercontext);
  std::cout << new_circ << std::endl;
}