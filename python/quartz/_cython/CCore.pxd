# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint32_t

ctypedef float ParamType

cdef extern from "gate/gate_utils.h" namespace "quartz":
    cpdef enum class GateType:
        h
        x
        y
        rx
        ry
        rz
        cx
        ccx
        add
        neg
        z
        s
        sdg
        t
        tdg
        ch
        swap
        p
        pdg
        u1
        u2
        u3
        ccz
        cz
        input_qubit
        input_param

cdef extern from "math/matrix.h" namespace "quartz":
    cdef cppclass MatrixBase:
        pass

cdef extern from "gate/gate.h" namespace "quartz":
    cdef cppclass Gate:
        Gate(GateType, int, int) except +
        # MatrixBase *get_matrix()
        # MatrixBase *get_matrix(const vector[ParamType])
        # ParamType compute(const vector[ParamType])
        bool is_commutative() const
        int get_num_qubits() const
        int get_num_parameters() const
        bool is_parameter_gate() const
        bool is_quantum_gate() const
        bool is_parametrized_gate() const
        bool is_toffoli_gate() const
        GateType tp
        int num_qubits, num_parameters

cdef extern from "context/context.h" namespace "quartz":
    cdef cppclass Context:
        Context(vector[GateType]) except +
