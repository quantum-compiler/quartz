# distutils: language = c++

from CCore cimport GateType
from CCore cimport Gate
from CCore cimport DAG
from CCore cimport DAG_ptr
from CCore cimport GraphXfer
from CCore cimport Graph
from CCore cimport Context
from CCore cimport EquivalenceSet
import ctypes

ctypedef GraphXfer* GraphXfer_ptr

cdef class PyGateType:
    cdef GateType gt

    def __cinit__(self, GateType gt_):
        self.gt = gt_

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

cdef class PyGate:
    cdef Gate *cpp_gate

    def __cinit__(self, GateType gate_type, int num_qubits, int num_parameters):
        self.cpp_gate = new Gate(gate_type, num_qubits, num_parameters)

    def __dealloc__(self):
        del self.cpp_gate

    def is_commutative(self):
        return self.cpp_gate.is_commutative()

    def get_num_qubits(self):
        return self.cpp_gate.get_num_qubits()

    def get_num_parameters(self):
        return self.cpp_gate.get_num_parameters()

    def is_parameter_gate(self):
        return self.cpp_gate.is_parameter_gate()

    def is_quantum_gate(self):
        return self.cpp_gate.is_quantum_gate()

    def is_parametrized_gate(self):
        return self.cpp_gate.is_parametrized_gate()

    def is_toffoli_gate(self):
        return self.cpp_gate.is_toffoli_gate()

    @property
    def tp(self):
        return self.cpp_gate.tp

    @property
    def num_qubits(self):
        return self.cpp_gate.num_qubits

    @property
    def num_parameters(self):
        return self.cpp_gate.num_parameters

cdef class PyDAG:
    cdef DAG_ptr dag

    def __cinit__(self):
        pass

    def __dealloc__(self):
        del self.dag

    cdef set_this(self, DAG_ptr dag_):
        del self.dag
        self.dag = dag_
        return self

cdef class PyXfer:
    cdef GraphXfer *graphXfer

    def __cinit__(self, QuartzContext q_context, PyDAG py_dag_from, PyDAG py_dag_to):
        self.graphXfer = new GraphXfer(q_context.context, py_dag_from.dag, py_dag_to.dag)

    def __dealloc__(self):
        del self.graphXfer

    cdef set_this(self, GraphXfer *graphXfer_):
        del self.graphXfer
        self.graphXfer = graphXfer_
        return self

cdef class QuartzContext:
    cdef Context *context
    cdef EquivalenceSet *eqs

    def __cinit__(self, gate_type_list, equivalence_set_filename):
        self.context = new Context(gate_type_list)
        self.eqs = new EquivalenceSet()
        self.load_json(equivalence_set_filename)
        
    cdef load_json(self, filename):
        # Load ECC from file
        filename_bytes = filename.encode('utf-8')
        assert(self.eqs.load_json(self.context, filename_bytes), "Failed to load equivalence set.")

    @property
    def xfers(self):
        # Get all the equivalence sets
        # And convert them into xfers
        eq_sets = self.eqs.get_all_equivalence_sets()

        xfers = []
        for eq_set in eq_sets:
            first = True
            for dag_ptr in eq_set:
                if first:
                    first = False
                else:
                    xfers.append(PyXfer(self, PyDAG().set_this(eq_set[0]), PyDAG().set_this(dag_ptr)))
        return xfers

    def get_xfer_from_id(self, id):
        return self.xfers[id]

