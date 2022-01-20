# distutils: language = c++

from CCore cimport GateType
from CCore cimport Gate

cdef class PyGateType:
    cdef GateType gt

    def __cinit__(self, GateType gt_):
        self.gt = gt_

def get_gate_type(gate_type_str):
    if gate_type_str == "h": return GateType.h
    if gate_type_str == "x": return GateType.x
    if gate_type_str == "y": return GateType.y
    if gate_type_str == "rx": return GateType.rx
    if gate_type_str == "ry": return GateType.ry
    if gate_type_str == "rz": return GateType.rz
    if gate_type_str == "cx": return GateType.cx
    if gate_type_str == "ccx" :return GateType.ccx
    if gate_type_str == "add" :return GateType.add
    if gate_type_str == "neg" :return GateType.neg
    if gate_type_str == "z" :return GateType.z
    if gate_type_str == "s" :return GateType.s
    if gate_type_str == "sdg" :return GateType.sdg
    if gate_type_str == "t" :return GateType.t
    if gate_type_str == "tdg" :return GateType.tdg
    if gate_type_str == "ch" :return GateType.ch
    if gate_type_str == "swap" :return GateType.swap
    if gate_type_str == "p" :return GateType.p
    if gate_type_str == "pdg" :return GateType.pdg
    if gate_type_str == "u1" :return GateType.u1
    if gate_type_str == "u2" :return GateType.u2
    if gate_type_str == "u3" :return GateType.u3
    if gate_type_str == "ccz" :return GateType.ccz
    if gate_type_str == "cz" :return GateType.cz
    if gate_type_str == "input_qubit" :return GateType.input_qubit        
    if gate_type_str == "input_param" :return GateType.input_param        

cdef class PyGate:
    cdef Gate *cpp_gate

    def __cinit__(self, GateType gate_type, int num_qubits, int num_parameters):
        self.cpp_gate = new Gate(gate_type, num_qubits, num_parameters)

    def __dealloc__(self):
        del self.cpp_gate

    def is_commutative(self):
        return self.cpp_gate.is_commutative()

    def get_num_qubits(self):
        return self.cpp_gate.get_num_qubits()

    def get_num_parameters(self):
        return self.cpp_gate.get_num_parameters()

    def is_parameter_gate(self):
        return self.cpp_gate.is_parameter_gate()

    def is_quantum_gate(self):
        return self.cpp_gate.is_quantum_gate()

    def is_parametrized_gate(self):
        return self.cpp_gate.is_parametrized_gate()

    def is_toffoli_gate(self):
        return self.cpp_gate.is_toffoli_gate()


    @property
    def tp(self):
        return self.cpp_gate.tp

    @property
    def num_qubits(self):
        return self.cpp_gate.num_qubits

    @property
    def num_parameters(self):
        return self.cpp_gate.num_parameters