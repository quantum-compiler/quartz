# Compressed optimizer search queue

`Graph::optimize` can store candidates as immutable, flat snapshots rather
than retaining each candidate's complete graph. Compression is opt-in; existing
source calls keep the uncompressed path.

```cpp
OptimizerSearchOptions options;
options.compress_candidates = true;
auto result = graph->optimize(xfers, cost_upper_bound, circuit_name, "",
                              false, nullptr, 3600, "", false, options);
```

The overload with an ECC set and greedy preprocessing also accepts the options
as its final argument. This does not change `optimize_legacy`, simulated
annealing, or the Python API.

## Representation and ownership

`PackedGraph` stores operations and connections in two contiguous arrays.
Connections are stored once, instead of in both adjacency maps. It retains
operation IDs, input-qubit and parameter indices, empty adjacency entries, and
the special-operation counter. Logical-qubit lookup tables are reconstructed.
Unpacking does not allocate new global operation IDs, so traversal and rewrite
order remain unchanged.

Gate definitions are borrowed from the original graph's contexts. Retaining
these pointers preserves controlled-gate variants as well as ordinary gates.
Parameter IDs refer to the same parameter table, including constants and
symbolic expressions. The contexts, gate definitions, and parameter table must
outlive the search and its snapshots. This is an in-process representation,
not a portable serialization format.

Queued graphs are released after packing. A graph is reconstructed when popped
for expansion; the current graph, best result, and temporary rewrite result
remain materialized. Optional step history shares immutable snapshots through
parent links. Pruning releases branches no longer needed by a queued candidate
or the best result, and step export reconstructs the winning path.

Compressed candidates cache their insertion cost. Custom cost functions must be
deterministic, non-mutating, independent of graph addresses, and stable for the
duration of the search. The uncompressed comparator continues to call the cost
function. Neither path adds a tie-breaker or changes duplicate detection.

## Correctness checks

After building Quartz following `INSTALL.md`, run:

```sh
cmake --build build --target test_packed_graph benchmark_compressed_search
./build/test_packed_graph
```

The self-contained regression executable uses explicit failures even in Release
builds. It checks exact operation/edge round trips, parameter expressions,
controlled gates, isolated qubits, allocator state, matrix equivalence, custom
costs, matching search trajectories, queue shrinking, history export and
continuation, timeout behavior, and option forwarding through greedy
preprocessing. It needs no external ECC set.

For GCC/Clang sanitizer checks, configure a separate build with
`-DCMAKE_CXX_FLAGS_RELEASE='-O1 -g -UNDEBUG -fsanitize=address,undefined -fno-omit-frame-pointer'`,
build `test_packed_graph`, and run with `ASAN_OPTIONS=detect_leaks=1`.
The root CMake file currently selects Release explicitly; `-UNDEBUG` keeps
upstream assertions enabled in this configuration.

## Reproducible measurements

Run from the repository root. The Python runner uses Linux `wait4` to collect
kernel peak RSS for each fresh child process and alternates mode order across
repetitions. It requires only the Python standard library.

```sh
python scripts/benchmark_compressed_search.py \
  circuit/nam_circs/barenco_tof_4.qasm \
  eccset/Nam_5_3_complete_ECC_set.json --expansions 300 --repeats 3

python scripts/benchmark_compressed_search.py \
  circuit/nam_circs/barenco_tof_4.qasm \
  eccset/Nam_5_3_complete_ECC_set.json \
  --expansions 0 --timeout 5 --repeats 3
```

Equal-work runs fail if modes disagree on expansions, accepted candidates,
queue peaks, shrink counts, popped-hash digest, final cost, or result hash.
They also fail if the search exhausts or times out before the requested budget.
Equal-time runs report work and final quality without expecting identical
trajectories: serialization overhead can affect how much work fits the timeout.

The benchmark executable accepts an optional final step-file prefix for measuring
history-enabled searches and inspecting their exported QASM files.
`OptimizerSearchStats::peak_packed_queue_bytes` counts snapshot objects and
owned vector capacity in the queue. It excludes queue entries/control blocks,
history retained outside the queue, temporary packing allocations, context data,
the visited-hash set, and materialized graphs. Process RSS includes these costs;
it is not interchangeable with the snapshot-byte counter.

## Local measurements

Measured on Linux with GCC 16.2.1 and an AMD Ryzen 7 7840HS, using the Release
build and three fresh-process repetitions per mode (alternating order).
The reference is the uncompressed path in this change. Step recording was off.

| Circuit / ECC | Expansions | Peak RSS, reference / compressed (MiB) | Search time, reference / compressed (s) | Best cost, both |
| --- | ---: | ---: | ---: | ---: |
| barenco_tof_3 / Nam_3_3 | 1,000 | 93.6 / 20.9 | 4.761 / 4.502 | 52 |
| barenco_tof_4 / Nam_5_3 | 300 | 179.4 / 37.6 | 26.606 / 25.931 | 112 |

Numbers are medians. Both modes had identical accepted counts, popped-hash
digests, final hashes, queue peaks, and shrink counts in every equal-work run.
They accepted 18,711 and 46,681 candidates and shrank 16 and 45 times,
respectively. These two circuits demonstrate local memory savings, not a
performance guarantee across gate sets or machines.

The small `Nam_3_3_complete_ECC_set.json` is generated locally by the existing
`./build/test_optimize` smoke test when absent; it is not checked into the
repository. After generating it, the smaller measurement is reproducible with:

```sh
python scripts/benchmark_compressed_search.py \
  circuit/nam_circs/barenco_tof_3.qasm \
  eccset/Nam_3_3_complete_ECC_set.json --expansions 1000 --repeats 3
```

In three five-second runs per mode, the smaller circuit expanded a median of
951 reference versus 1,108 compressed candidates. Reference best costs were
52, 52, and 56; compressed best costs were 52 in all three runs. On the larger
circuit, the compressed path expanded slightly fewer candidates in this short
budget (58 versus 59 median expansions), with the same best cost of 114 in both modes.
Timing varies with host load; the equal-work comparisons provide the exact
trajectory check, and the memory reduction is the primary result.

A separate 1,000-expansion smaller-circuit run with history enabled produced
byte-identical step-count files, all 25 exported circuits, and final QASM in
both modes. Peak RSS was 106.7 MiB reference versus 22.7 MiB compressed. This
history measurement is one run per mode, rather than a repeated median.
