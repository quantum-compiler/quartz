# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint32_t
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr

ctypedef float ParamType

cdef extern from "gate/gate_utils.h" namespace "quartz":
    cpdef enum class GateType:
        h,
        x,
        y,
        rx,
        ry,
        rz,
        cx,
        ccx,
        add,
        neg,
        z,
        s,
        sdg,
        t,
        tdg,
        ch,
        swap,
        p,
        pdg,
        u1,
        u2,
        u3,
        ccz,
        cz,
        input_qubit,
        input_param

cdef extern from "math/matrix.h" namespace "quartz":
    cdef cppclass MatrixBase:
        pass

cdef extern from "gate/gate.h" namespace "quartz":
    cdef cppclass Gate:
        Gate(GateType, int, int) except +
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
        size_t next_global_unique_id()

ctypedef Context* Context_ptr

cdef extern from "dag/dag.h" namespace "quartz":
    cdef cppclass DAG:
        DAG(int, int) except +
        int get_num_qubits() const
        int get_num_input_parameters() const
        int get_num_total_parameters() const
        int get_num_internal_parameters() const
        int get_num_gates() const

ctypedef DAG* DAG_ptr

cdef extern from "tasograph/substitution.h" namespace "quartz":
    cdef cppclass GraphXfer:
        GraphXfer(Context_ptr, const DAG_ptr, const DAG_ptr) except +
        @staticmethod
        GraphXfer* create_GraphXfer(Context_ptr,const DAG_ptr ,const DAG_ptr)

cdef extern from "tasograph/tasograph.h" namespace "quartz":
    cdef cppclass Op:
        Op() except +
        Op(size_t , Gate *) except +
        size_t guid
        Gate * ptr

cdef extern from "tasograph/tasograph.h" namespace "quartz":
    cdef cppclass Edge:
        Op srcOp, dstOp
        int srcIdx, dstIdx


cdef extern from "tasograph/tasograph.h" namespace "quartz":
    cdef cppclass Graph:
        Graph(Context *) except +
        Graph(Context *, const DAG *) except +
        bool xfer_appliable(GraphXfer *, Op) except +
        shared_ptr[Graph] apply_xfer(GraphXfer *, Op) except +
        void all_ops(vector[Op]&) const
        int gate_count() const
        size_t hash()
        void all_edges(vector[Edge]&) const
        void topology_order_ops(vector[Op] &) const
        shared_ptr[Graph] ccz_flip_t(Context *)
        

cdef extern from "dataset/equivalence_set.h" namespace "quartz":
    cdef cppclass EquivalenceSet:
        EquivalenceSet() except +
        int num_equivalence_classes() const
        bool load_json(Context *, const string)
        vector[vector[DAG_ptr]] get_all_equivalence_sets() except +

cdef extern from "parser/qasm_parser.h" namespace "quartz":
    cdef cppclass QASMParser:
        QASMParser(Context *)
        bool load_qasm(const string &, DAG *&) except +