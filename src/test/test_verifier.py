import sys

sys.path.append("..")

from src.python.verifier.gates import *
from src.python.verifier.verifier import *


def test_apply_matrix():
    s1, c1, s2, c2 = z3.Reals('s1 c1 s2 c2')
    print('\nProving Rx(p1) Rx(p2) = Rx(p1 + p2)')
    slv = z3.Solver()
    slv.add(angle(s1, c1))
    slv.add(angle(s2, c2))
    # slv.add(z3.Not(eq_matrix(
    #     matmul(RX((c1, s1)), RX((c2, s2))),
    #     RX(Add((c1,s1), (c2,s2))))))
    vec = input_distribution(1, slv)
    output_vec1 = apply_matrix(apply_matrix(vec, rx((c1, s1)), [0]), rx((c2, s2)), [0])
    output_vec2 = apply_matrix(vec, rx(add((c1, s1), (c2, s2))), [0])
    slv.add(z3.Not(z3.And(eq_vector(output_vec1, output_vec2))))
    print(slv.check())


if __name__ == '__main__':
    test_apply_matrix()
    find_equivalences(
        'data.json',
        'equivalences.json',
        verbose=True,
        check_equivalence_with_different_hash=True,
    )
