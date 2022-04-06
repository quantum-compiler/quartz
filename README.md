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
See all supported gate types in [gates.inc.h](src/quartz/gate/gates.inc.h).

And then you can run `./gen_ecc_set.sh` to generate the ECC set.

### Optimize a quantum circuit

todo (zikun): instructions to optimize a quantum circuit (C++ or Python interface, whichever is easier)

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
