cdef extern from "context/context.h" namespace "quartz":
    cdef cppclass Context:
        pass


cdef extern from "gate/gate_utils.h" namespace "quartz":
    cdef enum class GateType:
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