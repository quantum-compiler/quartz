# distutils: language = c++

from CCore cimport (
    CircuitSeq,
    CircuitSeq_ptr,
    Context,
    Edge,
    EquivalenceSet,
    Gate,
    GateType,
    Graph,
    GraphXfer,
    Op,
    ParamInfo,
    QASMParser,
)
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

import ctypes
from enum import Enum

import dgl
import torch

ctypedef GraphXfer* GraphXfer_ptr


def get_gate_type_from_str(gate_type_str):
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
    if gate_type_str == "rx1" :return GateType.rx1
    if gate_type_str == "rx3" :return GateType.rx3
    if gate_type_str == "input_qubit" :return GateType.input_qubit
    if gate_type_str == "input_param" :return GateType.input_param
    if gate_type_str == "ry1": return GateType.ry1
    if gate_type_str == "ry3": return GateType.ry3
    if gate_type_str == "rxx1": return GateType.rxx1
    if gate_type_str == "rxx3": return GateType.rxx3
    if gate_type_str == "sx": return GateType.sx

cdef class PyQASMParser:
    cdef QASMParser *parser

    def __cinit__(self, *, QuartzContext context):
        self.parser = new QASMParser(context.context)

    def __dealloc__(self):
        pass

    def load_qasm(self, *, str filename) -> PyDAG:
        dag = PyDAG()
        filename_bytes = filename.encode('utf-8')
        success = self.parser.load_qasm(filename_bytes, dag.dag)
        assert(success, "Failed to load qasm file!")
        return dag

    def load_qasm_str(self, str qasm_str) -> PyDAG:
        dag = PyDAG()
        qasm_str_bytes = qasm_str.encode('utf-8')
        success = self.parser.load_qasm_str(qasm_str_bytes, dag.dag)
        assert(success, "Failed to load qasm file!")
        return dag

cdef class PyGate:
    cdef Gate *gate

    def __cinit__(self, *, str type_name=None, int num_qubits=-1, int num_parameters=-1):
        if type_name is not None and num_qubits >= 0 and num_parameters >= 0:
            gate_type = get_gate_type_from_str(type_name.lower())
            self.gate = new Gate(gate_type, num_qubits, num_parameters)
        else:
            self.gate = NULL

    def __dealloc__(self):
        pass

    cdef set_this(self, Gate* gate_):
        self.gate = gate_
        return self

    def is_commutative(self):
        return self.gate.is_commutative()

    def get_num_qubits(self):
        return self.gate.get_num_qubits()

    def get_num_parameters(self):
        return self.gate.get_num_parameters()

    def is_parameter_gate(self):
        return self.gate.is_parameter_gate()

    def is_quantum_gate(self):
        return self.gate.is_quantum_gate()

    def is_parametrized_gate(self):
        return self.gate.is_parametrized_gate()

    def is_toffoli_gate(self):
        return self.gate.is_toffoli_gate()

    @property
    def tp(self):
        return self.gate.tp

    @property
    def num_qubits(self):
        return self.gate.num_qubits

    @property
    def num_parameters(self):
        return self.gate.num_parameters

    @staticmethod
    def rebuild(GateType _type, int _num_qubits, int _num_params):
        gate = PyGate()
        inner_gate = new Gate(_type, _num_qubits, _num_params)
        gate.set_this(inner_gate)
        return gate

    def __reduce__(self):
        return (self.__class__.rebuild, (self.tp, self.num_qubits, self.num_parameters))


cdef class PyDAG:
    cdef CircuitSeq_ptr dag

    def __cinit__(self, *, int num_qubits=-1):
        if num_qubits >= 0:
            self.dag = new CircuitSeq(num_qubits)
        else:
            self.dag = NULL

    def __dealloc__(self):
        pass

    cdef set_this(self, CircuitSeq_ptr dag_):
        self.dag = dag_
        return self

    @property
    def num_qubits(self):
        return self.dag.get_num_qubits()

    @property
    def num_gates(self):
        return self.dag.get_num_gates()

cdef class PyXfer:
    cdef GraphXfer *graphXfer
    cdef bool is_nop

    def __cinit__(self, *, QuartzContext context=None, PyDAG dag_from=None, PyDAG dag_to=None, bool is_nop=False):
        self.is_nop = is_nop
        if context == None:
            self.graphXfer = NULL
        elif is_nop:
            self.graphXfer = NULL
        elif dag_from is not None and dag_to is not None:
            self.graphXfer = GraphXfer.create_GraphXfer(context.context, dag_from.dag, dag_to.dag, True)
        else:
            self.graphXfer = NULL

    def __dealloc__(self):
        pass

    cdef set_this(self, GraphXfer *graphXfer_, bool is_nop=False):
        self.graphXfer = graphXfer_
        self.is_nop = is_nop
        # TODO: maybe delete before setting to None?
        if self.is_nop:
            self.graphXfer = NULL
        return self

    # TODO: raise exception if NULL or NOP
    @property
    def src_gate_count(self):
        return self.graphXfer.num_src_op()

    # TODO: raise exception if NULL or NOP
    @property
    def dst_gate_count(self):
        return self.graphXfer.num_dst_op()

    @property
    def is_nop(self):
        return self.is_nop

    @property
    def is_NOP(self):
        return self.is_nop

    @property
    def src_str(self):
        if self.is_nop:
            return 'NOP'
        return self.graphXfer.src_str().decode('utf-8')

    @property
    def dst_str(self):
        if self.is_nop:
            return 'NOP'
        return self.graphXfer.dst_str().decode('utf-8')

cdef class QuartzContext:
    cdef ParamInfo *param_info
    cdef Context *context
    cdef EquivalenceSet *eqs
    cdef vector[GraphXfer *] v_xfers
    cdef bool include_nop

    def __cinit__(self, *,  gate_set, filename, no_increase=False, include_nop=True):
        gate_type_list = []
        for s in gate_set:
            gate_type_list.append(get_gate_type_from_str(s))
        if GateType.input_param not in gate_type_list:
            gate_type_list.append(GateType.input_param)
        if GateType.input_qubit not in gate_type_list:
            gate_type_list.append(GateType.input_qubit)
        self.param_info = new ParamInfo()
        self.context = new Context(gate_type_list, self.param_info)
        self.eqs = new EquivalenceSet()
        self.load_json(filename)

        eq_sets = self.eqs.get_all_equivalence_sets()

        # for i in range(eq_sets.size()):
        #     for j in range(eq_sets[i].size()):
        #         if j != 0:
        #             dag_ptr_0 = eq_sets[i][0]
        #             dag_ptr_1 = eq_sets[i][j]
        #             xfer_0 = GraphXfer.create_GraphXfer(self.context, dag_ptr_0, dag_ptr_1, no_increase)
        #             xfer_1 = GraphXfer.create_GraphXfer(self.context, dag_ptr_1, dag_ptr_0, no_increase)
        #             if xfer_0 != NULL:
        #                 self.v_xfers.push_back(xfer_0)
        #             if xfer_1 != NULL:
        #                 self.v_xfers.push_back(xfer_1)

        # for i in range(eq_sets.size()):
        #     for j in range(eq_sets[i].size()):
        #         if j != 0:
        #             dag_ptr_0 = eq_sets[i][0]
        #             dag_ptr_1 = eq_sets[i][j]
        #             xfer_0 = GraphXfer.create_GraphXfer(self.context, dag_ptr_0, dag_ptr_1, no_increase)
        #             xfer_1 = GraphXfer.create_GraphXfer(self.context, dag_ptr_1, dag_ptr_0, no_increase)
        #             if xfer_0 != NULL and xfer_0.num_dst_op() - xfer_0.num_src_op() < 2:
        #                 self.v_xfers.push_back(xfer_0)
        #             if xfer_1 != NULL and xfer_1.num_dst_op() - xfer_1.num_src_op() < 2:
        #                 self.v_xfers.push_back(xfer_1)

        for i in range(eq_sets.size()):
            for j in range(eq_sets[i].size()):
                for k in range(eq_sets[i].size()):
                    if j != k:
                        dag_ptr_0 = eq_sets[i][j]
                        dag_ptr_1 = eq_sets[i][k]
                        xfer = GraphXfer.create_GraphXfer(self.context, dag_ptr_0, dag_ptr_1, True)
                        if xfer == NULL:
                            continue
                        if no_increase and xfer.num_dst_op() - xfer.num_src_op() > 0:
                            continue
                        self.v_xfers.push_back(xfer)
        self.include_nop = include_nop

    cdef load_json(self, filename):
        # Load ECC from file
        filename_bytes = filename.encode('utf-8')
        assert(self.eqs.load_json(self.context, filename_bytes, False), "Failed to load equivalence set.")

    # size_t next_global_unique_id();
    def next_global_unique_id(self):
        return self.context.next_global_unique_id()

    def get_xfers(self):
        # Get all the equivalence sets
        # And convert them into xfers
        num_xfers = self.v_xfers.size()

        xfers = []
        for i in range(num_xfers):
            xfers.append(PyXfer().set_this(self.v_xfers[i]))
        if self.include_nop:
            xfers.append(PyXfer(is_nop=True))
        return xfers

    def get_xfer_from_id(self, *, id) -> PyXfer:
        if id < self.v_xfers.size():
            xfer = PyXfer().set_this(self.v_xfers[id])
        elif self.include_nop and id == self.v_xfers.size():
            xfer = PyXfer(is_nop=True)
        else:
            xfer = None
        return xfer

    def xfer_id_is_nop(self, *, xfer_id) -> bool:
        if xfer_id == self.v_xfers.size():
            if self.include_nop:
                return True
            else:
                assert False
        else:
            return False

    def has_parameterized_gate(self) -> bool:
        return self.context.has_parameterized_gate()

    def add_xfer_from_qasm_str(self, *, src_str: str, dst_str: str):
        src_bytes = src_str.encode('utf-8')
        dst_bytes = dst_str.encode('utf-8')
        xfer = GraphXfer.create_GraphXfer_from_qasm_str(self.context, src_bytes, dst_bytes)
        if xfer != NULL:
            self.v_xfers.push_back(xfer)

    @property
    def num_equivalence_classes(self):
        return self.eqs.num_equivalence_classes()

    @property
    def num_xfers(self):
        num = self.v_xfers.size()
        if self.include_nop:
            num += 1
        return num

from functools import partial


cdef class PyNode:
    cdef Op node

    def __cinit__(self, *, size_t guid = -1, PyGate gate = None):
        if id != -1 and gate != None:
            self.node = Op(guid, gate.gate)
        else:
            self.node = Op()

    def __dealloc__(self):
        pass

    @property
    def node_guid(self):
        return self.node.guid

    @property
    def guid(self):
        return self.node.guid

    @property
    def gate(self):
        return PyGate().set_this(self.node.ptr)

    @property
    def gate_tp(self):
        return self.node.ptr.tp

    def __reduce__(self):
        return (
            partial(self.__class__, guid=self.node_guid, gate=self.gate), ()
        )

cdef class PyGraph:
    cdef shared_ptr[Graph] graph
    cdef object _nodes
    cdef size_t _hash

    property nodes:
        def __get__(self):
            return self._nodes

        def __set__(self, nodes):
            self._nodes = nodes

    property _hash:
        def __get__(self):
            return self._hash

        def __set__(self, _hash):
            self._hash = _hash

    def __cinit__(self, *, QuartzContext context = None, PyDAG dag = None):
        self.nodes = []
        if context != None and dag != None:
            self.graph = make_shared[Graph](context.context, dag.dag)
            self.get_nodes()
            self._hash = deref(self.graph).hash()
        else:
            self.graph = shared_ptr[Graph](NULL)
            self._hash = 0


    def __dealloc__(self):
        self.graph.reset()

    def __hash__(self):
        return self._hash

    def get_nodes(self):
        gate_count = self.gate_count
        cdef vector[Op] nodes_vec
        nodes_vec.reserve(gate_count)
        deref(self.graph).topology_order_ops(nodes_vec)

        self.nodes = []
        for i in range(gate_count):
            self.nodes.append(PyNode(
                guid=nodes_vec[i].guid,
                gate=PyGate().set_this(nodes_vec[i].ptr)
            ))

    cdef set_this(self, shared_ptr[Graph] graph_):
        self.graph = graph_
        self.get_nodes()
        self._hash = deref(self.graph).hash()
        return self

    # TODO: deprecate this function
    cdef _xfer_appliable(self, PyXfer xfer, PyNode node):
        return deref(self.graph).xfer_appliable(xfer.graphXfer, node.node)

    # TODO: use node_id directly instead of using PyNode
    def xfer_appliable(self, *, PyXfer xfer, PyNode node):
        if xfer.is_nop:
            return True
        return self._xfer_appliable(xfer, node)

    # TODO: use node_id directly instead of using PyNode
    def available_xfers(self, *, QuartzContext context, PyNode node, output_format="int"):
        result = deref(self.graph).appliable_xfers(node.node, context.v_xfers)
        if context.include_nop:
            result.push_back(context.num_xfers - 1)
        return result

    def available_xfers_parallel(self, *, QuartzContext context, PyNode node, output_format="int"):
        result = deref(self.graph).appliable_xfers_parallel(node.node, context.v_xfers)
        if context.include_nop:
            result.push_back(context.num_xfers - 1)
        return result

    # TODO: use node_id directly instead of using PyNode
    def apply_xfer(self, *, PyXfer xfer, PyNode node, eliminate_rotation:bool = False) -> PyGraph:
        if xfer.is_nop:
            return self
        ret = deref(self.graph).apply_xfer(xfer.graphXfer, node.node, eliminate_rotation)
        if ret.get() == NULL:
            return None
        else:
            return PyGraph().set_this(ret)

    # TODO: use node_id directly instead of using PyNode
    def apply_xfer_with_local_state_tracking(self, *, PyXfer xfer, PyNode node, eliminate_rotation:bool = False, predecessor_layers: int = 1):
        if xfer.is_nop:
            return self, []
        ret = deref(self.graph).apply_xfer_and_track_node(xfer.graphXfer, node.node, eliminate_rotation, predecessor_layers)
        if ret.first.get() == NULL:
            return None, []
        else:
            return PyGraph().set_this(ret.first), ret.second

    def all_nodes(self):
        return self.nodes

    def all_nodes_with_id(self) -> list:
        nodes_with_id = [
            { "id": i, 'node': node }
            for (i, node) in enumerate(self.nodes)
        ]
        return nodes_with_id

    def get_node_from_id(self, *, id : int) -> PyNode:
        n = self.num_nodes
        if id >= n:
            print(id)
            print(n)
            self.to_qasm(filename='a.qasm')
        assert(id < self.num_nodes)
        return self.nodes[id]

    def hash(self):
        return self._hash

    def all_edges(self):
        id_guid_mapping = {}
        gate_cnt = len(self.nodes)
        for i in range(gate_cnt):
            id_guid_mapping[self.nodes[i].guid] = i

        cdef vector[Edge] edge_v
        deref(self.graph).all_edges(edge_v)
        cdef int edge_cnt = edge_v.size()
        edges = []
        for i in range(edge_cnt):
            e = (id_guid_mapping[edge_v[i].srcOp.guid], id_guid_mapping[edge_v[i].dstOp.guid], edge_v[i].srcIdx, edge_v[i].dstIdx)
            edges.append(e)
        return edges

    def to_dgl_graph(self):
        edges = self.all_edges()
        src_id = []
        dst_id = []
        src_idx = []
        dst_idx = []

        for e in edges:
            src_id.append(e[0])
            dst_id.append(e[1])
            src_idx.append(e[2])
            dst_idx.append(e[3])
        src_id2 = src_id + dst_id
        dst_id2 = dst_id + src_id
        src_idx2 = src_idx + dst_idx
        dst_idx2 = dst_idx + src_idx
        reverse = [0] * len(src_id) + [1] * len(src_id)

        g = dgl.graph((torch.tensor(src_id2), torch.tensor(dst_id2)))
        g.edata['src_idx'] = torch.tensor(src_idx2, dtype=torch.int32)
        g.edata['dst_idx'] = torch.tensor(dst_idx2, dtype=torch.int32)
        g.edata['reversed'] = torch.tensor(reverse, dtype=torch.int32)

        node_gate_tp = [node.gate_tp for node in self.nodes]
        g.ndata['gate_type'] = torch.tensor(node_gate_tp, dtype=torch.int32)

        return g

    def get_available_xfers_matrix(self, *, context):
        rows, cols = (self.num_nodes, context.num_xfers)
        arr = [[0 for i in range(cols)] for j in range(rows)]
        for i in range(rows):
            available_list = self.available_xfers(context=context, node=self.nodes[i], output_format='int')
            for xfer_id in available_list:
                arr[i][xfer_id] = 1
        return arr

    # def toffoli_flip(self, *, QuartzContext context, str target):
    #     if target == "t":
    #         return PyGraph().set_this(deref(self.graph).ccz_flip_t(context.context))
    #     return None

    def to_qasm(self, *, str filename):
        fn_bytes = filename.encode('utf-8')
        deref(self.graph).to_qasm(fn_bytes, False, False)

    def to_qasm_str(self, *) -> str:
        cdef string s = deref(self.graph).to_qasm(False, False)
        return s.decode('utf-8')

    def rotation_merging(self, gate_type:str):
        deref(self.graph).rotation_merging(get_gate_type_from_str(gate_type))
        self.get_nodes()
        return self

    @staticmethod
    def from_qasm(*, context : QuartzContext, filename : str):
        filename_bytes = filename.encode('utf-8')
        return PyGraph().set_this(Graph.from_qasm_file(context.context, filename_bytes))

    @staticmethod
    def from_qasm_str(*, context : QuartzContext, qasm_str : str):
        qasm_str_bytes = qasm_str.encode('utf-8')
        return PyGraph().set_this(Graph.from_qasm_str(context.context, qasm_str_bytes))

    def ccz_flip_greedy_rz(self, *, rotation_merging=False):
        return PyGraph().set_this(deref(self.graph).ccz_flip_greedy_rz())

    def __lt__(self, other):
        return self.gate_count < other.gate_count

    def __le__(self, other):
        return self.gate_count <= other.gate_count

    def __eq__(self, other: PyGraph):
        if other is None:
            return False
        return deref(self.graph).equal(deref(other.graph))

    @property
    def gate_count(self):
        return deref(self.graph).gate_count()

    @property
    def cx_count(self):
        return deref(self.graph).specific_gate_count(GateType.cx)

    @property
    def t_count(self):
        return deref(self.graph).specific_gate_count(GateType.t) + deref(self.graph).specific_gate_count(GateType.tdg)

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        cdef vector[Edge] edge_v
        deref(self.graph).all_edges(edge_v)
        return edge_v.size()

    @property
    def depth(self) -> int:
        return deref(self.graph).circuit_depth()
