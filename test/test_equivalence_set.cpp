#include "test_dataset.h"

int main() {
  test_equivalence_set(all_supported_gates(),
                       "equivalences.json",
                       "equivalences_sorted.json");
}
