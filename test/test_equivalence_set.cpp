#include "test_dataset.h"

int main() {
  test_equivalence_set(all_supported_gates(),
                       "bfs_verified.json",
                       "equivalences_simplified.json");
}
