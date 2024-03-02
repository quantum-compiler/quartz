# distutils: language=c++

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

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
        rx1,
        rx3,
        input_qubit,
        input_param,
        ry1,
        ry3,
        rxx1,
        rxx3,
        sx

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

cdef extern from "context/param_info.h" namespace "quartz":
    cdef cppclass ParamInfo:
        ParamInfo() except +

ctypedef ParamInfo* ParamInfo_ptr

cdef extern from "context/context.h" namespace "quartz":
    cdef cppclass Context:
        Context(const vector[GateType], ParamInfo_ptr) except +
        size_t next_global_unique_id()
        bool has_parameterized_gate() const

ctypedef Context* Context_ptr

cdef extern from "circuitseq/circuitseq.h" namespace "quartz":
    cdef cppclass CircuitSeq:
        CircuitSeq(int) except +
        int get_num_qubits() const
        int get_num_gates() const

ctypedef CircuitSeq* CircuitSeq_ptr

cdef extern from "tasograph/substitution.h" namespace "quartz":
    cdef cppclass GraphXfer:
        GraphXfer(Context_ptr, const CircuitSeq_ptr, const CircuitSeq_ptr) except +
        @staticmethod
        GraphXfer* create_GraphXfer(Context_ptr,const CircuitSeq_ptr ,const CircuitSeq_ptr, bool equal_num_input_params)
        @staticmethod
        GraphXfer* create_GraphXfer_from_qasm_str(Context_ptr, const string &, const string &)
        int num_src_op()
        int num_dst_op()
        string src_str()
        string dst_str()

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
        Graph(Context *, const CircuitSeq *) except +
        bool xfer_appliable(GraphXfer *, Op) const
        shared_ptr[Graph] apply_xfer(GraphXfer *, Op, bool) const
        pair[shared_ptr[Graph], vector[int]] apply_xfer_and_track_node(GraphXfer *, Op, bool, int) const
        vector[size_t] appliable_xfers(Op, const vector[GraphXfer *] &) const
        vector[size_t] appliable_xfers_parallel(Op, const vector[GraphXfer *] &) const
        void all_ops(vector[Op]&) const
        int gate_count() const
        int cx_count() const
        int specific_gate_count(GateType) const
        size_t hash()
        void all_edges(vector[Edge]&) const
        void topology_order_ops(vector[Op] &) const
        shared_ptr[Graph] ccz_flip_t(Context *)
        void to_qasm(const string &, bool, bool) const
        @staticmethod
        shared_ptr[Graph] from_qasm_file(Context *, const string &)
        @staticmethod
        shared_ptr[Graph] from_qasm_str(Context *, const string &)
        string to_qasm(bool, bool) const
        shared_ptr[Graph] ccz_flip_greedy_rz()
        bool equal(const Graph &) const
        void rotation_merging(GateType)
        int circuit_depth() const


cdef extern from "dataset/equivalence_set.h" namespace "quartz":
    cdef cppclass EquivalenceSet:
        EquivalenceSet() except +
        int num_equivalence_classes() const
        bool load_json(Context *, const string, bool)
        vector[vector[CircuitSeq_ptr]] get_all_equivalence_sets() except +


cdef extern from "parser/qasm_parser.h" namespace "quartz":
    cdef cppclass QASMParser:
        QASMParser(Context *)
        bool load_qasm(const string &, CircuitSeq *&) except +
        bool load_qasm_str(const string &, CircuitSeq *&) except +
