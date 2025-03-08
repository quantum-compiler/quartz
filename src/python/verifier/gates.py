import math
import os
import sys

import z3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.utils import (
    sqrt2,
    sqrt3,
    sqrt5,
    sqrt_of_2_minus_sqrt2,
    sqrt_of_2_plus_sqrt2,
    sqrt_of_5_minus_sqrt5,
)

# helper solver


nonnegativity_solver = z3.Solver()
nonnegativity_solver.add(
    1.414 < sqrt2,
    sqrt2 < 1.415,
    1.732 < sqrt3,
    sqrt3 < 1.733,
    2.236 < sqrt5,
    sqrt5 < 2.237,
    1.847 < sqrt_of_2_plus_sqrt2,
    sqrt_of_2_plus_sqrt2 < 1.848,
    0.765 < sqrt_of_2_minus_sqrt2,
    sqrt_of_2_minus_sqrt2 < 0.766,
    1.662 < sqrt_of_5_minus_sqrt5,
    sqrt_of_5_minus_sqrt5 < 1.663,
)
nonnegativity_solver.set("timeout", 10)  # timeout in milliseconds


# helper methods


def half(a):
    # Unchecked Assertion: a is defined as (cos(x), sin(x)) for x in [0, 2pi].
    assert len(a) == 2
    cos_a, sin_a = a

    sin_sgn = 1
    cos_sgn = z3.If(sin_a >= 0, 1, -1)
    # Simplify the if-expression if possible
    nonnegativity_solver.push()
    nonnegativity_solver.add(sin_a < 0)
    if nonnegativity_solver.check() == z3.unsat:  # must be non-negative
        cos_sgn = 1
    else:
        nonnegativity_solver.pop()
        nonnegativity_solver.push()
        nonnegativity_solver.add(sin_a > 0)
        if nonnegativity_solver.check() == z3.unsat:  # must be non-positive
            cos_sgn = -1
    nonnegativity_solver.pop()

    cos_half_a = cos_sgn * z3.Sqrt((1 + cos_a) / 2)
    sin_half_a = sin_sgn * z3.Sqrt((1 - cos_a) / 2)
    return cos_half_a, sin_half_a


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


def mult(x, y):
    # In this special case, both the lhs and the rhs are numbers.
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x * y

    # To apply trigonometric angle formulas, one side must be a number.
    # Without loss of generality, the left-hand side will be a number.
    assert isinstance(x, (int, float)) or isinstance(y, (int, float))
    if isinstance(y, (int, float)):
        x, y = y, x

    # This block ensures that the lhs is not only a number, but also an integer.
    # This is because angle-reducing formula only exist for integer multipliers.
    # Of course, other formulas exist, such as the half-angle formula.
    # However, this formula is not determined (for arbitrary a) by (cos_a, sin_a) alone.
    if isinstance(x, float):
        assert x.is_integer()
        x = int(x)

    # Moves negative signs from the left-hand side to the right-hand side.
    if x < 0:
        x = -x
        y = neg(y)

    # Base Cases.
    if x == 0:
        return 1, 0
    elif x == 1:
        return y
    # Triple-angle formula.
    elif x % 3 == 0:
        cos_y, sin_y = mult(x // 3, y)
        cos_z = 4 * cos_y * cos_y * cos_y - 3 * cos_y
        sin_z = 3 * sin_y - 4 * sin_y * sin_y * sin_y
        return cos_z, sin_z
    # Double-angle formula.
    elif x % 2 == 0:
        cos_y, sin_y = mult(x // 2, y)
        cos_z = cos_y * cos_y - sin_y * sin_y
        sin_z = 2 * cos_y * sin_y
        return cos_z, sin_z
    # Otherwise, use the sum formula to decrease x by 1.
    else:
        z = mult(x - 1, y)
        return add(y, z)


def pi(n, use_z3=True):
    # This function handles fractions of pi with integer denominators.
    assert isinstance(n, (int, float))
    if isinstance(n, float):
        assert n.is_integer()
        n = int(n)

    # Negative Multiples of Pi.
    if n < 0:
        return neg(pi(-n))
    # Exactly Pi.
    if n == 1:
        return -1, 0
    elif n == 2:
        return 0, 1
    elif n == 4:
        if use_z3:
            return sqrt2 / 2, sqrt2 / 2
        else:
            return math.sqrt(2) / 2, math.sqrt(2) / 2
    elif n == 8:
        if use_z3:
            return sqrt_of_2_plus_sqrt2 / 2, sqrt_of_2_minus_sqrt2 / 2
        else:
            return math.sqrt(2 + math.sqrt(2)) / 2, math.sqrt(2 - math.sqrt(2)) / 2
    # Half-Angle Formula.
    elif n % 2 == 0:
        return half(pi(n // 2))
    # Some Fermat Prime Denominators.
    #
    # This extends to other Fermat primes, according to the Gauss-Wantzel theorem.
    # However, there are conjectured to be only five Fermat primes.
    # The remaining Fermat primes are (17, 257, 65537).
    # The equations are horrendous, and probably not useful in practice.
    # If necessary, they could be implemented (at least for n = 17).
    elif n == 3:
        if use_z3:
            cos_a = 1 / 2
            sin_a = sqrt3 / 2
        else:
            cos_a = 1 / 2
            sin_a = math.sqrt(3) / 2
        return cos_a, sin_a
    elif n == 5:
        if use_z3:
            cos_a = (sqrt5 + 1) / 4
            sin_a = sqrt2 * sqrt_of_5_minus_sqrt5 / 4
        else:
            cos_a = (math.sqrt(5) + 1) / 4
            sin_a = math.sqrt(2) * math.sqrt(5 - math.sqrt(5)) / 4
        return cos_a, sin_a
    elif n == 15:
        return add(mult(2, pi(5)), neg(pi(3)))

    # An Unhandled Case.
    assert False


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
            [(1 / sqrt2, 0), (-1 / sqrt2 * cos_l, -1 / sqrt2 * sin_l)],
            [
                (1 / sqrt2 * cos_phi, 1 / sqrt2 * sin_phi),
                (
                    1 / sqrt2 * (cos_l * cos_phi - sin_l * sin_phi),
                    1 / sqrt2 * (sin_phi * cos_l + sin_l * cos_phi),
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
            [(1 / sqrt2, 0), (1 / sqrt2, 0)],
            [(1 / sqrt2, 0), (-1 / sqrt2, 0)],
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
        return [[(1, 0), (0, 0)], [(0, 0), (sqrt2 / 2, sqrt2 / 2)]]
    else:
        return [[(1, 0), (0, 0)], [(0, 0), (math.sqrt(2) / 2, math.sqrt(2) / 2)]]


def tdg(use_z3=True):
    if use_z3:
        return [[(1, 0), (0, 0)], [(0, 0), (sqrt2 / 2, -sqrt2 / 2)]]
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
            [(sqrt2 / 2, 0), (0, -sqrt2 / 2)],
            [(0, -sqrt2 / 2), (sqrt2 / 2, 0)],
        ]
    else:
        return [
            [(math.sqrt(2) / 2, 0), (0, -math.sqrt(2) / 2)],
            [(0, -math.sqrt(2) / 2), (math.sqrt(2) / 2, 0)],
        ]


def rx3(use_z3=True):
    if use_z3:
        return [
            [(sqrt2 / 2, 0), (0, sqrt2 / 2)],
            [(0, sqrt2 / 2), (sqrt2 / 2, 0)],
        ]
    else:
        return [
            [(math.sqrt(2) / 2, 0), (0, math.sqrt(2) / 2)],
            [(0, math.sqrt(2) / 2), (math.sqrt(2) / 2, 0)],
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
            [(sqrt2 / 2, 0), (-sqrt2 / 2, 0)],
            [(sqrt2 / 2, 0), (sqrt2 / 2, 0)],
        ]
    else:
        return [
            [(math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0)],
            [(math.sqrt(2) / 2, 0), (math.sqrt(2) / 2, 0)],
        ]


def ry3(use_z3=True):
    if use_z3:
        return [
            [(-sqrt2 / 2, 0), (-sqrt2 / 2, 0)],
            [(sqrt2 / 2, 0), (-sqrt2 / 2, 0)],
        ]
    else:
        return [
            [(-math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0)],
            [(math.sqrt(2) / 2, 0), (-math.sqrt(2) / 2, 0)],
        ]


def rxx1(use_z3=True):
    if use_z3:
        return [
            [(sqrt2 / 2, 0), (0, 0), (0, 0), (-sqrt2 / 2, 0)],
            [(0, 0), (sqrt2 / 2, 0), (-sqrt2 / 2, 0), (0, 0)],
            [(0, 0), (-sqrt2 / 2, 0), (sqrt2 / 2, 0), (0, 0)],
            [(-sqrt2 / 2, 0), (0, 0), (0, 0), (sqrt2 / 2, 0)],
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
            [(-sqrt2 / 2, 0), (0, 0), (0, 0), (-sqrt2 / 2, 0)],
            [(0, 0), (-sqrt2 / 2, 0), (-sqrt2 / 2, 0), (0, 0)],
            [(0, 0), (-sqrt2 / 2, 0), (-sqrt2 / 2, 0), (0, 0)],
            [(-sqrt2 / 2, 0), (0, 0), (0, 0), (-sqrt2 / 2, 0)],
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


def ccz(use_z3=True):
    return [
        [(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (-1, 0)],
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
