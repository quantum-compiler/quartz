# distutils: language = c++

from cython.operator import dereference
from libcpp.vector cimport vector
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
from enum import Enum
import ctypes

ctypedef GraphXfer* GraphXfer_ptr


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

cdef class PyQASMParser:
    cdef QASMParser *parser

    def __cinit__(self, QuartzContext qc):
        self.parser = new QASMParser(qc.context)

    def __dealloc__(self):
        pass

    def load_qasm(self, qasm_fn, PyDAG py_dag):
        qasm_fn_bytes = qasm_fn.encode('utf-8')
        return self.parser.load_qasm(qasm_fn_bytes, py_dag.dag)

cdef class PyGate:
    cdef Gate *gate

    def __cinit__(self, GateType gate_type=GateType.h, int num_qubits=1, int num_parameters=0):
        self.gate = new Gate(gate_type, num_qubits, num_parameters)

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

    def __cinit__(self, int num_qubits=1, int num_input_parameters=1):
        self.dag = new DAG(num_qubits, num_input_parameters)

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

    def __cinit__(self, QuartzContext q_context=None, PyDAG py_dag_from=None, PyDAG py_dag_to=None):
        if q_context == None:
            self.graphXfer = NULL
        else:
            self.graphXfer = GraphXfer.create_GraphXfer(q_context.context, py_dag_from.dag, py_dag_to.dag)

    def __dealloc__(self):
        pass

    cdef set_this(self, GraphXfer *graphXfer_):
        self.graphXfer = graphXfer_
        return self

cdef class QuartzContext:
    cdef Context *context
    cdef EquivalenceSet *eqs
    cdef vector[GraphXfer *] v_xfers

    def __cinit__(self, gate_type_list, equivalence_set_filename):
        if GateType.input_param not in gate_type_list:
            gate_type_list.append(GateType.input_param)
        if GateType.input_qubit not in gate_type_list:
            gate_type_list.append(GateType.input_qubit)
        self.context = new Context(gate_type_list)
        self.eqs = new EquivalenceSet()
        self.load_json(equivalence_set_filename)

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

    def get_xfer_from_id(self, id):
        xfer = PyXfer().set_this(self.v_xfers[id])
        return xfer

    @property
    def num_equivalence_classes(self):
        return self.eqs.num_equivalence_classes()
    
    @property
    def num_xfers(self):
        return self.v_xfers.size()

cdef class PyNode:
    cdef Op *node

    def __cinit__(self, int node_id = -1, PyGate py_gate = None):
        if node_id == -1:
            self.node = NULL
        else:
            self.node = new Op(node_id, py_gate.gate)

    def __dealloc__(self):
        pass
    
    cdef set_this(self, Op *node_):
        self.node = node_
        return self

    @property
    def node_id(self):
        return self.node.guid

    @property
    def gate(self):
        pass

    @property
    def gate_tp(self):
        if self.node == NULL:
            print("node NULL")
        if self.node.ptr == NULL:
            print("gate NULL")
        return self.node.ptr.tp


cdef class PyGraph:
    cdef Graph *graph

    def __cinit__(self, QuartzContext quartz_context, PyDAG py_dag = None):
        if py_dag == None:
            self.graph = new Graph(quartz_context.context)
        else:
            self.graph = new Graph(quartz_context.context, py_dag.dag)

    def __dealloc__(self):
        pass

    cdef set_this(self, Graph *graph_):
        self.graph = graph_
        return self

    cdef _xfer_appliable(self, PyXfer xfer, PyNode node):
        if xfer.graphXfer == NULL:
            print("xfer NULL")
        if node.node == NULL:
            print("node NULL")
        return self.graph.xfer_appliable(xfer.graphXfer, node.node)

    def available_xfers(self, quartz_context, py_node, output_format):
        xfers = quartz_context.get_xfers()
        result = []
        for i in range(len(xfers)):
            if self._xfer_appliable(xfers[i], py_node):
                print("_xfer_appliable")
                if output_format in ['int']:
                    result.append(i)
                else:
                    result.append(xfers[i])
        return result
                    
    cdef _apply_xfer(self, QuartzContext quartz_context, PyXfer xfer, PyNode node):
        graph = PyGraph(quartz_context)
        return graph.set_this(self.graph.apply_transfer(xfer.graphXfer, node.node))

    def apply_xfer(self,quartz_context, py_xfer, py_node):
        return self._apply_xfer(quartz_context, py_xfer, py_node)
        
    cdef _all_nodes(self):
        cdef int gate_count = self.gate_count
        cdef vector[Op] vec
        vec.reserve(gate_count)
        self.graph.all_ops(vec)
        py_node_list = []
        for i in range(gate_count):
            py_node_list.append(PyNode(vec[i].guid, PyGate().set_this(vec[i].ptr)))
        return py_node_list
    
    def all_nodes(self):
        return self._all_nodes()

    @property
    def gate_count(self):
        return self.graph.gate_count()
        