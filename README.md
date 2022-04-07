# The Quartz Quantum Circuit Optimizer

Quartz is a quantum circuit optimizer that automatically generates and verifies circuit transformations for an arbitrary quantum gate set. To optimize an input quantum circuit, Quartz uses these auto-generated circuit transformations to construct a search space of functionally equivalent quantum circuits.
Quartz uses a cost-based search algorithm to explore the space and discovers highly optimized quantum circuits.

## Install Quartz

See [instructions](INSTALL.md) to install Quartz from source code.

## Use Quartz

Quartz targets the logical optimization stage in quantum circuit compilation and can be used to optimize quantum circuits for arbitrary gate sets (e.g., IBM or Regetti quantum processors). Quartz works in two steps. First, for a given gate set, the Quartz circuit generator and circuit equivalence verifier can automatically generate and verify possible circuit transformations, represented as an equivalent circuit class (ECC) set. Second, Quartz's circuit optimizer takes a quantum circuit and an ECC set as inputs and use cost-based backtracking search to discover a super-optimized quantum circuit. 

### Generate and verify an ECC set

To generate and verify pre-defined ECC sets, you can simply run `./gen_ecc_set.sh`.

To generate an `(n,q)`-complete ECC set with `m` input parameters for some gate set, 
you can change the main function in `src/test/gen_ecc_set.cpp` to the following:

```c++
gen_ecc_set({Gate set}, "{Name of the gate set}_{n}_{q}_", true, q, m, n);
return 0;
```
where `{Gate set}` can be `{GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add}` for the Nam gate set,
`{GateType::u1, GateType::u2, GateType::u3, GateType::cx, GateType::add}` for the IBM gate set,
`{GateType::rx, GateType::rz, GateType::cz, GateType::add}` for the Rigetti gate set,
or any gate set you want. `GateType::add` is to enable using a sum of two input parameters as an input to a parameterized quantum gate.
See all supported gate types in [gates.inc.h](src/quartz/gate/gates.inc.h) and their implementations in [gate/](src/quartz/gate).

And then you can run `./gen_ecc_set.sh` to generate the ECC set.

### Optimize a quantum circuit

todo (zikun): instructions to optimize a quantum circuit (C++ or Python interface, whichever is easier)

#### Input the circuit

To optimize a circuit, you can write your circuit in the `qasm` language and write it to a `qasm` file. Currently we only support a subset of `qasm`'s gramma. Specifically, the `qasm` files we support should consist of a header and lines of `qasm` instructions. The header should be in the format below:

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
```

The instructions should be in the format below:
```
cx q[3], q[2];
cx q[8], q[7];
cx q[14], q[13];
cx q[21], q[20];
```

We do not support parameterized gates currently.

To input a circuit in `qasm` file, you should first create a `Context` object, providing the gate set you use in your input file as parameter as below:

``` cpp
Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                GateType::input_qubit, GateType::input_param});
```

After that, you need a `QASMParser` object to parse the input `qasm` file. You can construct it as below:

``` cpp
QASMParser qasm_parser(&src_ctx);
```

Now you can use the `QASMParser` object to load the circuit from the `qasm` file to a `DAG` object, as below:

``` cpp
DAG *dag = nullptr;
if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
}
```

After you have the circuit loaded into the `DAG` object, you can construct a `Graph` object from it. The `Graph` object is the final circuit representation used in our optimizer. You can construct it as below:

``` cpp
Graph graph(&src_ctx, dag);
```

#### Context shift

If the input gate set is different from your target gate set, you should use the `context_shift` API.

## Repository Organization

See [code structure](CODE_STRUCTURE.md) for more information about the organization of the Quartz code base.

## Contributing

Please let us know if you encounter any bugs or have any suggestions by [submitting an issue](https://github.com/quantum-compiler/quartz/issues).

We welcome all contributions to Quartz from bug fixes to new features and extensions.

Please subscribe to the Quartz users mailing list for 

## Citations

* Mingkuan Xu, Zikun Li, Oded Padon, Sina Lin, Jessica Pointing, Auguste Hirth, Henry Ma, Jens Palsberg, Alex Aiken, Umut A. Acar, and Zhihao Jia. [Quartz: Superoptimization of Quantum Circuits](). In Proceedings of the Conference on Programming Language Design and Implementation (PLDI), June 2022.


## License

Quartz uses Apache License 2.0.
