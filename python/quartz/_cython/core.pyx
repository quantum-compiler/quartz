# distutils: language = c++

from cython.operator import dereference
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from CCore cimport GateType
from CCore cimport Gate
from CCore cimport DAG
from CCore cimport DAG_ptr
from CCore cimport GraphXfer
from CCore cimport Graph
from CCore cimport Op
from CCore cimport Context
from CCore cimport EquivalenceSet
from CCore cimport QASMParser
from CCore cimport Edge
from enum import Enum
import ctypes
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
    if gate_type_str == "input_qubit" :return GateType.input_qubit        
    if gate_type_str == "input_param" :return GateType.input_param        

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

cdef class PyDAG:
    cdef DAG_ptr dag

    def __cinit__(self, *, int num_qubits=-1, int num_input_params=-1):
        if num_qubits >= 0 and num_input_params >= 0:
            self.dag = new DAG(num_qubits, num_input_params)
        else:
            self.dag = NULL

    def __dealloc__(self):
        pass

    cdef set_this(self, DAG_ptr dag_):
        self.dag = dag_
        return self

    @property
    def num_qubits(self):
        return self.dag.get_num_qubits()
    
    @property
    def num_input_parameters(self):
        return self.dag.get_num_input_parameters()
    
    @property
    def num_total_parameters(self):
        return self.dag.get_num_total_parameters() 

    @property
    def num_internal_parameters(self):
        return self.dag.get_num_internal_parameters() 
    
    @property
    def num_gates(self):
        return self.dag.get_num_gates() 

cdef class PyXfer:
    cdef GraphXfer *graphXfer

    def __cinit__(self, *, QuartzContext context=None, PyDAG dag_from=None, PyDAG dag_to=None):
        if context == None:
            self.graphXfer = NULL
        elif dag_from is not None and dag_to is not None:
            self.graphXfer = GraphXfer.create_GraphXfer(context.context, dag_from.dag, dag_to.dag)

    def __dealloc__(self):
        pass

    cdef set_this(self, GraphXfer *graphXfer_):
        self.graphXfer = graphXfer_
        return self

cdef class QuartzContext:
    cdef Context *context
    cdef EquivalenceSet *eqs
    cdef vector[GraphXfer *] v_xfers

    def __cinit__(self, *,  gate_set, filename):
        gate_type_list = []
        for s in gate_set:
            gate_type_list.append(get_gate_type_from_str(s))
        if GateType.input_param not in gate_type_list:
            gate_type_list.append(GateType.input_param)
        if GateType.input_qubit not in gate_type_list:
            gate_type_list.append(GateType.input_qubit)
        self.context = new Context(gate_type_list)
        self.eqs = new EquivalenceSet()
        self.load_json(filename)

        eq_sets = self.eqs.get_all_equivalence_sets()

        for i in range(eq_sets.size()):
            for j in range(eq_sets[i].size()):
                if j != 0:
                    dag_ptr_0 = eq_sets[i][0]
                    dag_ptr_1 = eq_sets[i][j]
                    xfer_0 = GraphXfer.create_GraphXfer(self.context, dag_ptr_0, dag_ptr_1)
                    xfer_1 = GraphXfer.create_GraphXfer(self.context, dag_ptr_1, dag_ptr_0)
                    if xfer_0 != NULL:
                        self.v_xfers.push_back(xfer_0)
                    if xfer_1 != NULL:
                        self.v_xfers.push_back(xfer_1)
        
        
    cdef load_json(self, filename):
        # Load ECC from file
        filename_bytes = filename.encode('utf-8')
        assert(self.eqs.load_json(self.context, filename_bytes), "Failed to load equivalence set.")

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
        return xfers

    def get_xfer_from_id(self, *, id):
        xfer = PyXfer().set_this(self.v_xfers[id])
        return xfer

    @property
    def num_equivalence_classes(self):
        return self.eqs.num_equivalence_classes()
    
    @property
    def num_xfers(self):
        return self.v_xfers.size()

cdef class PyNode:
    cdef Op node

    def __cinit__(self, *, int guid = -1, PyGate gate = None):
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
    def gate(self):
        return PyGate().set_this(self.node.ptr)

    @property
    def gate_tp(self):
        return self.node.ptr.tp


cdef class PyGraph:
    cdef Graph *graph
    cdef vector[Op] nodes

    def __cinit__(self, *, QuartzContext context = None, PyDAG dag = None):
        if context != None and dag != None:
            self.graph = new Graph(context.context, dag.dag)
            gate_count = self.gate_count
            self.nodes.reserve(gate_count)
            self.graph.topology_order_ops(self.nodes)
        else:
            self.graph = NULL
            self.nodes.clear()

    def __dealloc__(self):
        pass

    cdef set_this(self, Graph *graph_):
        self.graph = graph_
        gate_count = self.gate_count
        self.nodes.clear()
        self.nodes.reserve(gate_count)
        self.graph.topology_order_ops(self.nodes)
        return self

    cdef _xfer_appliable(self, PyXfer xfer, PyNode node):
        return self.graph.xfer_appliable(xfer.graphXfer, node.node)

    def available_xfers(self, quartz_context, py_node, output_format):
        xfers = quartz_context.get_xfers()
        result = []
        for i in range(len(xfers)):
            if self._xfer_appliable(xfers[i], py_node):
                if output_format in ['int']:
                    result.append(i)
                else:
                    result.append(xfers[i])
        return result
                    
    def apply_xfer(self, *, PyXfer xfer, PyNode node) -> PyGraph:
        ret = self.graph.apply_xfer(xfer.graphXfer, node.node)
        if ret == NULL:
            return None
        else:
            return PyGraph().set_this(ret)
        
    def all_nodes_with_id(self) -> list:
        py_node_list = []
        gate_count = self.gate_count
        for i in range(gate_count):
            node_dict = {}
            node_dict['id'] = i
            node_dict['node'] = PyNode(guid=self.nodes[i].guid, gate=PyGate().set_this(self.nodes[i].ptr))
            py_node_list.append(node_dict)
        return py_node_list

    def all_nodes(self) -> list:
        py_node_list = []
        gate_count = self.gate_count
        for i in range(gate_count):
            py_node_list.append(PyNode(guid=self.nodes[i].guid, gate=PyGate().set_this(self.nodes[i].ptr)))
        return py_node_list

    def get_node_from_id(self, *, id) -> PyNode:
        assert(id < self.num_nodes)
        return PyNode(guid=self.nodes[id].guid, gate=PyGate().set_this(self.nodes[id].ptr))

    def hash(self):
        return self.graph.hash()

    def all_edges(self):
        id_guid_mapping = {}
        gate_cnt = self.nodes.size()
        for i in range(gate_cnt):
            id_guid_mapping[self.nodes[i].guid] = i

        cdef vector[Edge] edge_v
        self.graph.all_edges(edge_v)
        edges = []
        cdef int edge_cnt = edge_v.size()
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
        g.edata['src_idx'] = torch.tensor(src_idx2)
        g.edata['dst_idx'] = torch.tensor(dst_idx2)
        g.edata['reversed'] = torch.tensor(reverse)

        nodes = self.all_nodes()
        node_gate_tp = [node.gate_tp for node in nodes]
        g.ndata['gate_type'] = torch.tensor(node_gate_tp)

        return g


    def __lt__(self, other):
        return self.gate_count < other.gate_count
    
    def __le__(self, other):
        return self.gate_count <= other.gate_count

    @property
    def gate_count(self):
        return self.nodes.size()

    @property
    def num_nodes(self):
        return self.nodes.size()

    @property
    def num_edges(self):
        cdef vector[Edge] edge_v
        self.graph.all_edges(edge_v)
        return edge_v.size()
        
