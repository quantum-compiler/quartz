#!/usr/bin/python3

"""
We represent a matrix as list of list of pairs, where a pair
is represented as a real and an imaginary part of a complex number.
Angles are represented with two real numbers, s and c, satisfying s*s+c*c=1
"""

import z3
import math

# functions for generating z3 constraints

def eq(A, B):
    assert len(A) == len(B)
    assert all(len(ra) == len(rb) for ra, rb in zip(A, B))
    assert all(len(z) == 2 for M in (A, B) for r in M for z in r)
    return z3.And([
        x == y
        for ra, rb in zip(A, B)
        for za, zb in zip(ra, rb)
        for x, y in zip(za, zb)
    ])

def matmul(A, B):
    n = len(A)
    assert len(B) == n
    assert all(len(ra) == len(rb) == n for ra, rb in zip(A, B))
    assert all(len(z) == 2 for M in (A, B) for r in M for z in r)
    matrix = list()
    for r in range(n):
        row = list()
        for c in range(n):
            val_r = 0
            val_c = 0
            for k in range(n):
                val_r += A[r][k][0] * B[k][c][0] - A[r][k][1] * B[k][c][1]
                val_c += A[r][k][0] * B[k][c][1] + A[r][k][1] * B[k][c][0]
            row.append((val_r, val_c))
        matrix.append(row)
    return matrix

def angle(s, c):
    return s*s + c*c == 1

# parameter gates
def Add(a, b):
    assert len(a) == 2
    assert len(b) == 2
    cos_a, sin_a = a
    cos_b, sin_b = b
    return (cos_a*cos_b-sin_a*sin_b, sin_a*cos_b+cos_a*sin_b)

# quantum gates

def RX(theta):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, 0), (0, -sin_theta)],
            [(0, -sin_theta), (cos_theta, 0)]]

if __name__ == "__main__":
    s1, c1, s2, c2 = z3.Reals('s1 c1 s2 c2')
    print('\nProving Rx(p1) Rx(p2) = Rx(p1 + p2)')
    slv = z3.Solver()
    slv.add(angle(s1, c1))
    slv.add(angle(s2, c2))
    slv.add(z3.Not(eq(
        matmul(RX((c1, s1)), RX((c2, s2))),
        RX(Add((c1,s1), (c2,s2))))))
    print(slv.check())
