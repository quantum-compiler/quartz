# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint32_t
from libcpp.string cimport string


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
        Context(const vector[GateType]) except +
        pass
        # Gate *get_gate(GateType)
        # vector[GateType] get_supported_gates() const
        # vector[GateType] get_supported_parameter_gates() const
        # vector[GateType] get_supported_quantum_gates() const
        # Two deterministic (random) distributions for each number of qubits.
        # const Vector &get_generated_input_dis(int num_qubits);
        # const Vector &get_generated_hashing_dis(int num_qubits);
        # std::vector<ParamType> get_generated_parameters(int num_params);
        # std::vector<ParamType> get_all_generated_parameters() const;
        # size_t next_global_unique_id();

        # A hacky function: set a generated parameter.
        # void set_generated_parameter(int id, ParamType param);

        # This function assumes that two DAGs are equivalent iff they share the same hash value.
        # DAG *get_possible_representative(DAG *dag);

        # This function assumes that two DAGs are equivalent iff they share the same hash value.
        # void set_representative(std::unique_ptr<DAG> dag);
        # void clear_representatives();

        # This function generates a deterministic series of random numbers ranging [0, 1].
        # double random_number();

ctypedef Context* Context_ptr

cdef extern from "dag/dag.h" namespace "quartz":
    cdef cppclass DAG:
        DAG(int, int) except +

ctypedef DAG* DAG_ptr

cdef extern from "tasograph/substitution.h" namespace "quartz":
    cdef cppclass GraphXfer:
        GraphXfer(Context_ptr, const DAG_ptr, const DAG_ptr) except +

cdef extern from "tasograph/tasograph.h" namespace "quartz":
    cdef cppclass Graph:
        pass

cdef extern from "dataset/equivalence_set.h" namespace "quartz":
    cdef cppclass EquivalenceSet:
        EquivalenceSet() except +
        bool load_json(Context *, const string)

        vector[vector[DAG_ptr]] get_all_equivalence_sets() except +