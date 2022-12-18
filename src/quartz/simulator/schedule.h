#pragma once

#include "../context/context.h"
#include "quartz/circuitseq/circuitseq.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <map>
#include <vector>

namespace quartz {
/**
 * A simulation schedule of a circuit sequence on one device.
 */
class Schedule {
public:
  Schedule(const CircuitSeq &sequence, const std::vector<bool> &local_qubit,
           Context *ctx)
      : sequence_(sequence), local_qubit_(local_qubit), ctx_(ctx) {}

  // Compute the number of down sets for the circuit sequence.
  [[nodiscard]] size_t num_down_sets();

  // TODO: a function to compute the schedule

  // The result simulation schedule. We will execute the kernels one by one,
  // and each kernel contains a sequence of gates.
  std::vector<CircuitSeq> kernels;

private:
  // The original circuit sequence.
  CircuitSeq sequence_;

  // The mask for local qubits.
  std::vector<bool> local_qubit_;

  Context *ctx_;
};

std::vector<Schedule>
get_schedules(const CircuitSeq &sequence,
              const std::vector<std::vector<bool>> &local_qubits, Context *ctx);
} // namespace quartz
