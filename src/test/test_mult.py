import sys

sys.path.append("..")

from src.python.verifier.gates import *


def approx_eq(a, b):
    assert len(a) == 2
    assert len(b) == 2
    cos_a, sin_a = a
    cos_b, sin_b = b

    err = max(abs(cos_a - cos_b), abs(sin_a - sin_b))
    return err < 0.0000000000001


def mult_test(expected, n, a):
    actual = mult(n, a)
    swapped = mult(a, n)

    assert actual == swapped
    assert approx_eq(actual, expected)


def test_positive(a):
    expected = 1, 0
    for n in range(0, 41):
        mult_test(expected, n, a)
        expected = add(a, expected)


def test_negative(a):
    expected = neg(a)
    for n in range(-1, -41, -1):
        mult_test(expected, n, a)
        expected = add(neg(a), expected)


def test_floats(a):
    n = 5
    expected = mult(n, a)

    actual = mult(float(n), a)
    swapped = mult(a, float(n))

    assert actual == swapped
    assert actual == expected


def test_numbers():
    assert mult(2, 3.0) == 6.0
    assert mult(3.0, 2) == 6.0
    assert mult(3, 4) == 12
    assert mult(3.0, 0.1) == 0.3


if __name__ == '__main__':
    v = 1 / math.sqrt(2)
    test_positive((v, v))
    test_negative((v, v))
    test_floats((v, v))
