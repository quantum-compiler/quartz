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
    @tp.setter
    def tp(self, tp_):
        self.cpp_gate.tp = tp_

    @property
    def num_qubits(self):
        return self.cpp_gate.num_qubits
    @num_qubits.setter
    def num_qubits(self, num_qubits_):
        self.cpp_gate.num_qubits = num_qubits_

    @property
    def num_parameters(self):
        return self.cpp_gate.num_parameters
    @num_parameters.setter
    def num_parameters(self, num_parameters_):
        self.cpp_gate.num_parameters = num_parameters_