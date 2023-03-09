# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint32_t
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair

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
        bool has_parameterized_gate() const

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
        GraphXfer* create_GraphXfer(Context_ptr,const DAG_ptr ,const DAG_ptr, bool no_increase)
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
        Graph(Context *, const DAG *) except +
        bool xfer_appliable(GraphXfer *, Op) except +
        shared_ptr[Graph] apply_xfer(GraphXfer *, Op, bool) except +
        pair[shared_ptr[Graph], vector[int]] apply_xfer_and_track_node(GraphXfer *, Op, bool) except +
        vector[size_t] appliable_xfers(Op, const vector[GraphXfer *] &)
        vector[size_t] appliable_xfers_parallel(Op, const vector[GraphXfer *] &)
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
        bool load_qasm_str(const string &, DAG *&) except +


# Cython for physical mapping
cdef extern from "supported_devices/supported_devices.h" namespace "quartz":
    cpdef enum class BackendType:
        Q20_CLIQUE,
        IBM_Q20_TOKYO,
        Q5_TEST,
        IBM_Q127_EAGLE,
        IBM_Q27_FALCON,
        IBM_Q65_HUMMINGBIRD


cdef extern from "tasograph/tasograph.h" namespace "quartz":
    cdef cppclass GraphState:
        int number_of_nodes
        vector[int] node_id
        vector[bool] is_input
        vector[int] input_logical_idx
        vector[int] input_physical_idx
        vector[int] node_type
        int number_of_edges
        vector[int] edge_from
        vector[int] edge_to
        vector[bool] edge_reversed
        vector[int] edge_logical_idx
        vector[int] edge_physical_idx


cdef extern from "game/game_utils.h" namespace "quartz":
    ctypedef double Reward

    cdef cppclass State:
        State(vector[pair[int, int]], vector[int], vector[int], GraphState)
        vector[pair[int, int]] device_edges
        vector[int] logical2physical
        vector[int] physical2logical
        GraphState graph_state
        bool is_initial_phase

    cpdef enum class ActionType:
        PhysicalFull,
        PhysicalFront,
        Logical,
        SearchFull,
        Unknown

    cdef cppclass Action:
        Action(ActionType, int, int)
        ActionType type
        int qubit_idx_0
        int qubit_idx_1


cdef extern from "env/simple_physical_env.h" namespace "quartz":
    cdef cppclass SimplePhysicalEnv:
        SimplePhysicalEnv(const string &, BackendType, int, double, const string &)
        void reset() except +
        Reward step(Action)
        bool is_finished()
        int total_cost()
        State get_state()
        vector[Action] get_action_space()

cdef extern from "env/simple_initial_env.h" namespace "quartz":
    cdef cppclass SimpleInitialEnv:
        SimpleInitialEnv(const string &, BackendType)
        void reset() except +
        Reward step(Action)
        State get_state()
        vector[Action] get_action_space()

cdef extern from "env/simple_search_env.h" namespace "quartz":
    cdef cppclass SimpleSearchEnv:
        SimpleSearchEnv()
        SimpleSearchEnv(const string &, BackendType, int, double, const string &)
        void reset() except +
        Reward step(Action)
        State get_state()
        vector[Action] get_action_space()
        shared_ptr[SimpleSearchEnv] copy()

cdef extern from "env/simple_hybrid_env.h" namespace "quartz":
    cdef cppclass SimpleHybridEnv:
        SimpleHybridEnv()
        SimpleHybridEnv(const string &, BackendType, const string &, int, double, int, int, int, bool, double)
        void reset()
        Reward step(Action)
        bool is_finished()
        int total_cost()
        double sum_ln_cx_fidelity()
        State get_state()
        vector[Action] get_action_space()
        void save_context_to_file(const string &, const string &)
        void generate_mapped_qasm(const string &, bool)
