#include "../context/context.h"

class Generator {
 public:
  void generate(Context *ctx,
                int num_qubits,
                int max_num_parameters,
                int max_num_gates);
};
