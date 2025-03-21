# The Quartz Quantum Circuit Optimizer

Quartz is a quantum circuit optimizer that automatically generates and verifies
circuit transformations for an arbitrary quantum gate set. To optimize an input
quantum circuit, Quartz uses these auto-generated circuit transformations to
construct a search space of functionally equivalent quantum circuits. Quartz
uses a cost-based search algorithm to explore the space and discovers highly
optimized quantum circuits.

## PLDI 2022 Artifact

If you would like to compare with the Quartz version published in PLDI 2022,
please go to https://github.com/quantum-compiler/quartz-artifact. Note that the
format of ECC sets has been changed since then (after v0.2.0).

## Install Quartz

See [instructions](INSTALL.md) to install Quartz from source code.

## Use Quartz

Quartz targets the logical optimization stage in quantum circuit compilation and
can be used to optimize quantum circuits for arbitrary gate sets (e.g., IBM or
Regetti quantum processors). Quartz works in two steps. First, for a given gate
set, the Quartz circuit generator and circuit equivalence verifier can
automatically generate and verify possible circuit transformations, represented
as an equivalent circuit class (ECC) set. Second, Quartz's circuit optimizer
takes a quantum circuit and an ECC set as inputs and uses cost-based backtracking
search to discover a super-optimized quantum circuit.

### Generate and verify an ECC set

To generate and verify pre-defined ECC sets, you can simply
run `./gen_ecc_set.sh`.

To generate an `(n,q)`-complete ECC set with `m` input parameters for some gate
set, you can change the main function in `src/test/gen_ecc_set.cpp` to the
following:

```c++
gen_ecc_set({Gate set}, "{Name of the gate set}_{n}_{q}_", true, true, q, m, n);
return 0;
```

where `{Gate set}` can
be `{GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add}` for
the Nam gate set,
`{GateType::u1, GateType::u2, GateType::u3, GateType::cx, GateType::add}` for
the IBM gate set,
`{GateType::rx, GateType::rz, GateType::cz, GateType::add}` for the Rigetti gate
set, or any gate set you want. `GateType::add` is to enable using a sum of two
input parameters as an input to a parameterized quantum gate. See all supported
gate types in [gates.inc.h](src/quartz/gate/gates.inc.h) and their
implementations in [gate/](src/quartz/gate).

And then you can run `./gen_ecc_set.sh` to generate the ECC set.

## Optimize a quantum circuit

We show the steps to super-optimize a quantum circuit in Quartz.

#### Input the circuit

To optimize a circuit, you can write your circuit
in [OpenQASM](https://openqasm.com/) and write it to a `qasm` file. Currently,
we only support a subset of OpenQASM 2.0 and OpenQASM 3.0 grammar. Specifically,
the `qasm` files we support should consist of a header and lines of `qasm`
instructions. The header should be in the format below (all quantum registers
should come before gates):

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
```

The instructions should be in the format below:

```qasm
cx q[3], q[2];
cx q[8], q[7];
cx q[14], q[13];
cx q[21], q[20];
h q[3];
```

For gates with symbolic parameters, you can write the circuit in OpenQASM 3.0:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
input array[angle,2] p;
qubit[2] q;
rz(pi/2) q[0];
rx(p[0]) q[1];
rx(p[1]) q[1];
```

Similarly, all parameter arrays and qubit arrays should come before gates.

To input a circuit in `qasm` file, you should first create a `Context` object
with a `ParamInfo` object, providing the gate set you use in your input file as
the argument as below:

``` cpp
ParamInfo param_info;
Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                GateType::input_qubit, GateType::input_param},
                &param_info);
```

After that, you need a `QASMParser` object to parse the input `qasm` file. You
can construct it as below:

``` cpp
QASMParser qasm_parser(&src_ctx);
```

By default, gates like `rz(pi/2)` are not symbolic. If you want to make them
symbolic, please toggle this option:

```c++
qasm_parser.use_symbolic_pi(true);
```

Now you can use the `QASMParser` object to load the circuit from the `qasm` file
to a `CircuitSeq` object, as below:

``` cpp
CircuitSeq *seq = nullptr;
if (!qasm_parser.load_qasm(input_fn, seq)) {
    std::cout << "Parser failed" << std::endl;
}
```

After you have the circuit loaded into the `CircuitSeq` object, you can
construct a `Graph` object from it. The `Graph` object is the final circuit
representation used in our optimizer. You can construct it as below:

``` cpp
Graph graph(&src_ctx, seq);
```

#### Context shift

If the input gate set is different from your target gate set, you should
consider using the `context_shift` APIs to shift the context constructed with
the gate sets to a context constructed with the target gate set.

To shift the context, you should create three `Context` objects, one for input,
one for target, and one for their union as below:

``` cpp
ParamInfo param_info;
Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                GateType::input_qubit, GateType::input_param},
                &param_info);
Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                GateType::cx, GateType::input_qubit, GateType::input_param},
                &param_info);
auto union_ctx = union_contexts(&src_ctx, &dst_ctx);
```

In order to shift contexts, you should provide the rules to express a gate in
the input gate set to the target gate set. To do this, you should construct
a `RuleParser` object. As follows:

``` cpp
RuleParser rules(
    {"cx q0 q1 = rx q1 pi; rz q1 0.5pi; rx q1 0.5pi; rz q1 -0.5pi; cz q0 "
        "q1; rx q1 pi; rz q1 0.5pi; rx q1 0.5pi; rz q1 -0.5pi;",
        "h q0 = rx q0 pi; rz q0 0.5pi; rx q0 0.5pi; rz q0 -0.5pi;",
        "x q0 = rx q0 pi;"});
```

As shown in the example above, the grammar for the rules is simple. Also, if a
gate in the input gate set already appears in the target set, you don't have to
provide a rule for it.

#### Optimization

You can use the API:

```cpp
std::shared_ptr<Graph> optimize(Context *ctx,
                                const std::string &equiv_file_name,
                                const std::string &circuit_name,
                                bool print_message,
                                std::function<float(Graph *)> cost_function = nullptr,
                                double cost_upper_bound = -1 /*default = current cost * 1.05*/,
                                double timeout = 3600 /*1 hour*/,
                                const std::string &store_all_steps_file_prefix = std::string());
```

Explanation for the parameters:

- `ctx`: The context object.
- `equiv_file_name`: The file name of the ECC set.
- `circuit_name`: The name of the circuit, which will be printed with the
  intermediate result.
- `print_message`: Print debug message to the console.
- `cost_function`: The cost function used in the search.
- `cost_upper_bound`: Maximum circuit cost to be searched during optimization.
- `timeout`: Timeout for optimization in seconds.
- `store_all_steps_file_prefix`: Experimental, to store all optimization steps
  in files. The default is not to store.

Usage example:

```c++
auto graph_optimized = graph->optimize(&context,
                                       equiv_file_name,
                                       circuit_name,
                                       /*print_message=*/true,
                                       [] (Graph *graph) { return graph->total_cost(); },
                                       /*cost_upper_bound=*/-1,
                                       /*timeout=*/10);
```

You can also use the deprecated API for now (not recommended):

``` cpp
Graph::optimize_legacy(float alpha, int budget, bool print_subst, Context *ctx,
                       const std::string &equiv_file_name, bool use_simulated_annealing,
                       bool enable_early_stop, bool use_rotation_merging_in_searching,
                       GateType target_rotation, std::string circuit_name = "",
                       int timeout = 86400 /*1 day*/);
```

Explanation for some of the parameters:

- `print_subst`: Deprecated will be removed in future version.
- `equiv_file_name`: The file name of the ECC set.
- `use_simulated_annealing`: Use simulated annealing in searching.
- `use_rotation_merging_in_searching`: Enable rotation merging in each iteration
  of the back-track searching.
- `target_rotation`: The target rotation used if you enable rotation merging in
  search.
- `circuit_name`: The name of the circuit, which will be printed with the
  intermediate result.
- `timeout`: Timeout for optimization in seconds.

## Verify circuit equivalence

You can also use Quartz's verifier independently by
calling `python src/python/verifier/verify_equivalences.py` with a Json file
containing a batch of circuits to be verified, or (after installation) compile
and run the following executable to verify the equivalence of two individual
circuits `circuit1.qasm` and `circuit2.qasm`:

```shell
cd build
make verify_openqasm
./verify_openqasm circuit1.qasm circuit2.qasm [timeout] [tmpdir]
```

If `[timeout]` is given, then the value (in milliseconds) will be used as a
timeout for all Z3 queries (otherwise a default value of 30 seconds is used,
which should be enough for most Z3 queries). If `[tmpdir]` is given, the
temporary files during verification will be put into this directory.

## Repository Organization

See [code structure](CODE_STRUCTURE.md) for more information about the
organization of the Quartz code base.

## Issues

Please file an issue or contact mingkuan@cmu.edu if you encounter any problems.

## Contributing

Please let us know if you encounter any bugs or have any suggestions
by [submitting an issue](https://github.com/quantum-compiler/quartz/issues).

We welcome all contributions to Quartz from bug fixes to new features and
extensions.

Please follow [developer guidance](doc/dev_setup.md).

## Citations

* Mingkuan Xu, Zikun Li, Oded Padon, Sina Lin, Jessica Pointing, Auguste Hirth,
  Henry Ma, Jens Palsberg, Alex Aiken, Umut A. Acar, and Zhihao Jia.
  [Quartz: Superoptimization of Quantum Circuits](
  https://arxiv.org/abs/2204.09033). In Proceedings of the Conference on
  Programming Language Design and Implementation (PLDI), June 2022.

## License

Quartz uses Apache License 2.0.
