import sys

import z3

sys.path.append("..")

from src.python.verifier.gates import *


def check_pi(n, cos_b, sin_b):
    cos_a, sin_a = pi(n)

    ctx = z3.Solver()
    ctx.add(kConstantEquations)
    ctx.add(cos_a == cos_b)
    ctx.add(sin_a == sin_b)
    assert ctx.check() == z3.sat


def test_1_1_1():
    cos_b = -1
    sin_b = 0
    check_pi(1, cos_b, sin_b)


def test_1_1_2():
    cos_b = 0
    sin_b = 1
    check_pi(2, cos_b, sin_b)


def test_1_3_1():
    cos_b = 1 / 2
    sin_b = sqrt3 / 2
    check_pi(3, cos_b, sin_b)


def test_5_1_1():
    cos_b = (sqrt5 + 1) / 4
    sin_b = sqrt2 * sqrt_of_5_minus_sqrt5 / 4
    check_pi(5, cos_b, sin_b)


def test_1_3_2():
    cos_b = sqrt3 / 2
    sin_b = 1 / 2
    check_pi(6, cos_b, sin_b)


def test_5_1_2():
    cos_b = z3.Sqrt(2) * z3.Sqrt(5 + z3.Sqrt(5)) / 4
    sin_b = (z3.Sqrt(5) - 1) / 4
    check_pi(10, cos_b, sin_b)


def test_5_3_1():
    # See: https://mathworld.wolfram.com/TrigonometryAnglesPi15.html
    cos_b = (z3.Sqrt(30 + 6 * z3.Sqrt(5)) + z3.Sqrt(5) - 1) / 8
    sin_b = z3.Sqrt(7 - z3.Sqrt(5) - z3.Sqrt(30 - 6 * z3.Sqrt(5))) / 4
    check_pi(15, cos_b, sin_b)


def test_1_1_4():
    cos_b = 1 / z3.Sqrt(2)
    sin_b = 1 / z3.Sqrt(2)
    check_pi(4, cos_b, sin_b)


def test_5_3_2():
    # See: https://mathworld.wolfram.com/TrigonometryAnglesPi30.html
    cos_b = z3.Sqrt(7 + z3.Sqrt(5) + z3.Sqrt(6 * (5 + z3.Sqrt(5)))) / 4
    sin_b = (-1 - z3.Sqrt(5) + z3.Sqrt(30 - 6 * z3.Sqrt(5))) / 8
    check_pi(30, cos_b, sin_b)


def test_1_3_4():
    cos_b = z3.Sqrt(2) * (z3.Sqrt(3) + 1) / 4
    sin_b = z3.Sqrt(2) * (z3.Sqrt(3) - 1) / 4
    check_pi(12, cos_b, sin_b)


def test_5_1_4():
    # See: https://mathworld.wolfram.com/TrigonometryAnglesPi20.html
    cos_b = z3.Sqrt(8 + 2 * z3.Sqrt(10 + 2 * z3.Sqrt(5))) / 4
    sin_b = z3.Sqrt(8 - 2 * z3.Sqrt(10 + 2 * z3.Sqrt(5))) / 4
    check_pi(20, cos_b, sin_b)


def test_1_1_8():
    cos_b = z3.Sqrt(2 + z3.Sqrt(2)) / 2
    sin_b = z3.Sqrt(2 - z3.Sqrt(2)) / 2
    check_pi(8, cos_b, sin_b)


def test_5_3_4():
    # Note that rad(pi/60) is equal to deg(3).
    # Moreover, deg(3) = deg(18 - 15) with deg(18) = rad(pi/10) and deg(15) = rad(pi/12).
    # The previous tests have validated that pi(10) and pi(12) are correct.
    #
    # This yields an alternative formula for pi(60), againist which we can validate pi(60).
    cos_b, sin_b = add(pi(10), neg(pi(12)))
    check_pi(60, cos_b, sin_b)


def test_1_3_8():
    # See: https://mathworld.wolfram.com/TrigonometryAnglesPi24.html
    cos_b = z3.Sqrt(2 + z3.Sqrt(2 + z3.Sqrt(3))) / 2
    sin_b = z3.Sqrt(2 - z3.Sqrt(2 + z3.Sqrt(3))) / 2
    check_pi(24, cos_b, sin_b)


def test_5_1_8():
    # Note that 5*(pi/8) - 3*(pi/5) = pi/40.
    # The previous tests have validated that pi(5) and pi(8) are correct.
    #
    # This yields an alternative formula for pi(60), againist which we can validate pi(40).
    cos_b, sin_b = add(mult(5, pi(8)), mult(-3, pi(5)))
    check_pi(40, cos_b, sin_b)


def test_1_1_16():
    # See: https://mathworld.wolfram.com/TrigonometryAnglesPi16.html
    cos_b = z3.Sqrt(2 + z3.Sqrt(2 + z3.Sqrt(2))) / 2
    sin_b = z3.Sqrt(2 - z3.Sqrt(2 + z3.Sqrt(2))) / 2
    check_pi(16, cos_b, sin_b)


def test_float():
    cos_b, sin_b = pi(2)
    check_pi(2.0, cos_b, sin_b)


def test_neg():
    cos_b, sin_b = pi(15)
    check_pi(-15, cos_b, -sin_b)


if __name__ == '__main__':
    # Note: Z3 cannot verify some of the tests. They are commented out here.
    # No Factors.
    test_1_1_1()
    # One Factor.
    test_1_1_2()
    test_1_3_1()
    test_5_1_1()
    # Two Factors.
    test_1_3_2()
    # test_5_1_2()
    test_5_3_1()
    test_1_1_4()
    # Three Factors.
    # test_5_3_2()
    test_1_3_4()
    # test_5_1_4()
    # test_1_1_8()
    # Four Factors.
    # test_5_3_4()
    # test_1_3_8()
    # test_5_1_8()
    # test_1_1_16()
    # Edge Cases.
    test_float()
    test_neg()
