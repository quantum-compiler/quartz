# distutils: language = c++

from CCore cimport GateType
from CCore cimport Gate


cdef class PyGate:
    cdef Gate *cpp_gate

    def __cinit__(self, GateType gate_type, int num_qubits, int num_parameters):
        self.cpp_gate = new Gate(gate_type, num_qubits, num_parameters)

    def __dealloc__(self):
        del self.cpp_gate

    @property
    def tp(self):
        return self.cpp_gate.tp

    @property
    def num_qubits(self):
        return self.cpp_gate.num_qubits

    @property
    def num_parameters(self):
        return self.cpp_gate.num_parameters