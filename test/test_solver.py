import numpy as np
from src.solver import Solver


def test_nonsingular():
    A = np.array([
        [1, 2],
        [3, 4],
    ])

    b = np.array([0, 2])
    x = np.array([2, -1])

    solver = Solver(A, b)
    xp, null_space = solver.solve()
    assert xp is not None
    assert np.all(np.abs(xp - x) < 1e-7)
    assert null_space == []


def test_singular_without_solution():
    A = np.array([
        [1, 2],
        [2, 4],
    ])

    b = np.array([3, 0])

    solver = Solver(A, b)
    # we don't care about the null space if a solution
    # doesn't exist
    xp, _ = solver.solve()
    assert xp is None


def test_singular_with_solution():
    A = np.array([
        [1, 2],
        [2, 4],
    ])

    b = np.array([1, 2])
    x = np.array([1, 0])

    solver = Solver(A, b)
    xp, null_space = solver.solve()
    assert xp is not None
    assert np.all(np.abs(xp - x) < 1e-7)

    for xn in null_space:
        assert np.all(A @ xn == 0)


def test_zeros_matrix():
    A = np.zeros((5, 4))
    b = np.ones((5,))

    solver = Solver(A, b)
    xp, _ = solver.solve()
    assert xp is None
