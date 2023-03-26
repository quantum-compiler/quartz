import math

import z3

# parameter gates


def add(a, b):
    assert len(a) == 2
    assert len(b) == 2
    cos_a, sin_a = a
    cos_b, sin_b = b
    return cos_a * cos_b - sin_a * sin_b, sin_a * cos_b + cos_a * sin_b


def neg(a):
    assert len(a) == 2
    cos_a, sin_a = a
    return cos_a, -sin_a


# quantum gates


def x(use_z3=True):
    return [[(0, 0), (1, 0)], [(1, 0), (0, 0)]]


def y(use_z3=True):
    return [[(0, 0), (0, -1)], [(0, 1), (0, 0)]]


def rx(theta, use_z3=True):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, 0), (0, -sin_theta)], [(0, -sin_theta), (cos_theta, 0)]]


def ry(theta, use_z3=True):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, 0), (-sin_theta, 0)], [(sin_theta, 0), (cos_theta, 0)]]


def rz(theta, use_z3=True):
    # e ^ {i * theta} = cos theta + i sin theta
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, -sin_theta), (0, 0)], [(0, 0), (cos_theta, sin_theta)]]


def u1(theta, use_z3=True):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(1, 0), (0, 0)], [(0, 0), (cos_theta, sin_theta)]]


def u2(phi, l, use_z3=True):
    assert len(phi) == 2
    assert len(l) == 2
    cos_phi, sin_phi = phi
    cos_l, sin_l = l
    if use_z3:
        return [
            [(1 / z3.Sqrt(2), 0), (-1 / z3.Sqrt(2) * cos_l, -1 / z3.Sqrt(2) * sin_l)],
            [
                (1 / z3.Sqrt(2) * cos_phi, 1 / z3.Sqrt(2) * sin_phi),
                (
                    1 / z3.Sqrt(2) * (cos_l * cos_phi - sin_l * sin_phi),
                    1 / z3.Sqrt(2) * (sin_phi * cos_l + sin_l * cos_phi),
                ),
            ],
        ]
    else:
        return [
            [
                (1 / math.sqrt(2), 0),
                (-1 / math.sqrt(2) * cos_l, -1 / math.sqrt(2) * sin_l),
            ],
            [
                (1 / math.sqrt(2) * cos_phi, 1 / math.sqrt(2) * sin_phi),
                (
                    1 / math.sqrt(2) * (cos_l * cos_phi - sin_l * sin_phi),
                    1 / math.sqrt(2) * (sin_phi * cos_l + sin_l * cos_phi),
                ),
            ],
        ]


def u3(theta, phi, l, use_z3=True):
    assert len(theta) == 2
    assert len(phi) == 2
    assert len(l) == 2
    cos_theta, sin_theta = theta
    cos_phi, sin_phi = phi
    cos_l, sin_l = l
    return [
        [(cos_theta, 0), (-sin_theta * cos_l, -sin_theta * sin_l)],
        [
            (sin_theta * cos_phi, sin_theta * sin_phi),
            (
                cos_theta * (cos_phi * cos_l - sin_phi * sin_l),
                cos_theta * (sin_phi * cos_l + sin_l * cos_phi),
            ),
        ],
    ]


def cx(use_z3=True):
    return [
        [(1, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (1, 0)],
        [(0, 0), (0, 0), (1, 0), (0, 0)],
        [(0, 0), (1, 0), (0, 0), (0, 0)],
    ]


def cp(phi, use_z3=True):
    assert len(phi) == 2
    cos_phi, sin_phi = phi
    return [
        [(1, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (1, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (1, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (cos_phi, sin_phi)],
    ]


def h(use_z3=True):
    if use_z3:
        return [
            [(1 / z3.Sqrt(2), 0), (1 / z3.Sqrt(2), 0)],
            [(1 / z3.Sqrt(2), 0), (-1 / z3.Sqrt(2), 0)],
        ]
    else:
        return [
            [(1 / math.sqrt(2), 0), (1 / math.sqrt(2), 0)],
            [(1 / math.sqrt(2), 0), (-1 / math.sqrt(2), 0)],
        ]


def s(use_z3=True):
    return [[(1, 0), (0, 0)], [(0, 0), (0, 1)]]


def sdg(use_z3=True):
    return [[(1, 0), (0, 0)], [(0, 0), (0, -1)]]


def t(use_z3=True):
    if use_z3:
        return [[(1, 0), (0, 0)], [(0, 0), (z3.Sqrt(2) / 2, z3.Sqrt(2) / 2)]]
    else:
        return [[(1, 0), (0, 0)], [(0, 0), (math.sqrt(2) / 2, math.sqrt(2) / 2)]]


def tdg(use_z3=True):
    if use_z3:
        return [[(1, 0), (0, 0)], [(0, 0), (z3.Sqrt(2) / 2, -z3.Sqrt(2) / 2)]]
    else:
        return [[(1, 0), (0, 0)], [(0, 0), (math.sqrt(2) / 2, -math.sqrt(2) / 2)]]


def z(use_z3=True):
    return [[(1, 0), (0, 0)], [(0, 0), (-1, 0)]]


def p(phi, use_z3=True):
    assert len(phi) == 2
    cos_phi, sin_phi = phi
    return [[(1, 0), (0, 0)], [(0, 0), (cos_phi, sin_phi)]]


def pdg(phi, use_z3=True):
    assert len(phi) == 2
    cos_phi, sin_phi = phi
    return [[(1, 0), (0, 0)], [(0, 0), (cos_phi, -sin_phi)]]


def rx1(use_z3=True):
    if use_z3:
        return [
            [(z3.Sqrt(2) / 2, 0), (0, -z3.Sqrt(2) / 2)],
            [(0, -z3.Sqrt(2) / 2), (z3.Sqrt(2) / 2, 0)],
        ]
    else:
        return [
            [(math.Sqrt(2) / 2, 0), (0, -math.Sqrt(2) / 2)],
            [(0, -math.Sqrt(2) / 2), (math.Sqrt(2) / 2, 0)],
        ]


def rx3(use_z3=True):
    if use_z3:
        return [
            [(z3.Sqrt(2) / 2, 0), (0, z3.Sqrt(2) / 2)],
            [(0, z3.Sqrt(2) / 2), (z3.Sqrt(2) / 2, 0)],
        ]
    else:
        return [
            [(math.Sqrt(2) / 2, 0), (0, math.Sqrt(2) / 2)],
            [(0, math.Sqrt(2) / 2), (math.Sqrt(2) / 2, 0)],
        ]


def cz(use_z3=True):
    return [
        [(1, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (1, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (1, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (-1, 0)],
    ]


def ry1(use_z3=True):
    if use_z3:
        return [
            [(z3.Sqrt(2) / 2, 0), (-z3.Sqrt(2) / 2, 0)],
            [(z3.Sqrt(2) / 2, 0), (z3.Sqrt(2) / 2, 0)],
        ]
    else:
        return [
            [(math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0)],
            [(math.sqrt(2) / 2, 0), (math.sqrt(2) / 2, 0)],
        ]


def ry3(use_z3=True):
    if use_z3:
        return [
            [(-z3.Sqrt(2) / 2, 0), (-z3.Sqrt(2) / 2, 0)],
            [(z3.Sqrt(2) / 2, 0), (-z3.Sqrt(2) / 2, 0)],
        ]
    else:
        return [
            [(-math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0)],
            [(math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0)],
        ]


def rxx1(use_z3=True):
    if use_z3:
        return [
            [(z3.Sqrt(2) / 2, 0), (0, 0), (0, 0), (-z3.Sqrt(2) / 2, 0)],
            [(0, 0), (z3.Sqrt(2) / 2, 0), (-z3.Sqrt(2) / 2, 0), (0, 0)],
            [(0, 0), (-z3.Sqrt(2) / 2, 0), (z3.Sqrt(2) / 2, 0), (0, 0)],
            [(-z3.Sqrt(2) / 2, 0), (0, 0), (0, 0), (z3.Sqrt(2) / 2, 0)],
        ]
    else:
        return [
            [(math.sqrt(2) / 2, 0), (0, 0), (0, 0), (-math.sqrt(2) / 2, 0)],
            [(0, 0), (math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0), (0, 0)],
            [(0, 0), (-math.sqrt(2) / 2, 0), (math.sqrt(2) / 2, 0), (0, 0)],
            [(-math.sqrt(2) / 2, 0), (0, 0), (0, 0), (math.sqrt(2) / 2, 0)],
        ]


def rxx3(use_z3=True):
    if use_z3:
        return [
            [(-z3.Sqrt(2) / 2, 0), (0, 0), (0, 0), (-z3.Sqrt(2) / 2, 0)],
            [(0, 0), (-z3.Sqrt(2) / 2, 0), (-z3.Sqrt(2) / 2, 0), (0, 0)],
            [(0, 0), (-z3.Sqrt(2) / 2, 0), (-z3.Sqrt(2) / 2, 0), (0, 0)],
            [(-z3.Sqrt(2) / 2, 0), (0, 0), (0, 0), (-z3.Sqrt(2) / 2, 0)],
        ]
    else:
        return [
            [(-math.sqrt(2) / 2, 0), (0, 0), (0, 0), (-math.sqrt(2) / 2, 0)],
            [(0, 0), (-math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0), (0, 0)],
            [(0, 0), (-math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0), (0, 0)],
            [(-math.sqrt(2) / 2, 0), (0, 0), (0, 0), (-math.sqrt(2) / 2, 0)],
        ]


def sx(use_z3=True):
    return [
        [(1 / 2, 1 / 2), (1 / 2, -1 / 2)],
        [(1 / 2, -1 / 2), (1 / 2, 1 / 2)],
    ]


# functions exposed to verifier


def get_matrix(gate_name, *params):
    try:
        result = eval(gate_name)(*params)
    except NameError:
        raise Exception(f"Gate '{gate_name}' is not implemented.")
    return result


def compute(gate_name, *params):
    # For parameter gates, but same as quantum gates
    return get_matrix(gate_name, *params)
