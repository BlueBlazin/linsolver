import random
import numpy as np
from src.solver import Solver


def test_random():
    for _ in range(1000):
        rows = random.randrange(1, 21)
        cols = random.randrange(1, 21)
        A = np.random.randn(rows, cols)
        b = np.random.randn(rows)

        xp, null_space = Solver(A, b).solve()
        if xp is not None:
            assert np.allclose(A @ xp, b)

            for xn in null_space:
                assert np.allclose(A @ xn, 0)


def test_random_square():
    for _ in range(1000):
        size = random.randrange(1, 21)
        A = np.random.randn(size, size)
        b = np.random.randn(size)

        xp, null_space = Solver(A, b).solve()
        assert np.allclose(A @ xp, b)
        assert null_space == []
