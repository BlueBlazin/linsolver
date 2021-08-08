import numpy as np
from typing import Callable, Optional


class Solver:
    """
    System of linear equations solver. Solves equations of the form Ax = b 
    using the gaussian elimination method.

    Args:
        `A`: numpy.ndarray - the A matrix
        `b`: numpy.ndarray - the targets
    """

    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self.A = A.astype(np.float64)
        self.b = b.astype(np.float64)
        self.rows = A.shape[0]
        self.cols = A.shape[1]
        self.pivots: list[tuple[int, int]] = []

    def solve(self) -> tuple[Optional[np.ndarray], list[np.ndarray]]:
        """
        Solves Ax = b.

        Returns:
        A tuple (xp, null_space) containing
            `xp`: numpy.ndarray - a particular solution of Ax = b
            `null_space`: list[numpy.ndarray] - a list of vectors
            spanning the null space of `A`.
        """
        # create the augmented matrix from `A` and `b`
        augmatrix = self._augment()
        # get the row echelon form of the agumented matrix
        self._row_echelon_form(augmatrix)
        # get the reduced row echelon form
        self._reduced_row_echelon_form(augmatrix)
        # exit if solution doesn't exist`
        if len(self.pivots) == 0:
            return (None, self._null_space(augmatrix))
        # get a particular solution
        xp = self._particular_sol(augmatrix)
        # get the null space of the matrix
        null_space = self._null_space(augmatrix)

        return (xp, null_space)

    def _null_space(self, augmatrix: np.ndarray) -> list[np.ndarray]:
        sols: list[np.ndarray] = []
        nsols = 0
        pivot_cols = self._get_pivot_cols(set)

        # last pivot column
        col = 0
        for row in range(self.rows):
            # find next pivot column
            for j in range(col, self.cols):
                if j in pivot_cols:
                    # set last pivot column to j
                    col = j
                    break
            # initialize solution idx to 0
            sol_idx = 0
            # iterate through remaining columns to right
            for j in range(col + 1, self.cols):
                if j not in pivot_cols:
                    # add solution vector if it does not exist
                    if nsols <= sol_idx:
                        sols.append(np.zeros((self.cols,)))
                        nsols += 1
                    # if j is a non pivot column, modify solution such
                    # that (row, j) and (row, col) will cancel each other
                    sols[sol_idx][j] = -1.0
                    sols[sol_idx][col] = augmatrix[row, j]
                    # increment solution idx
                    sol_idx += 1

            # increment column by one for next iteration since it's been used
            col += 1

        return sols

    def _particular_sol(self, augmatrix: np.ndarray) -> Optional[np.ndarray]:
        # get list of pivot columns for indexing
        pivot_cols = self._get_pivot_cols(list)
        nonpivot_cols = [col for col in range(self.cols) if col not in pivot_cols]

        xp = np.zeros((self.cols,))
        xp[pivot_cols] = augmatrix[pivot_cols, -1]

        if augmatrix[nonpivot_cols, -1] == np.zeros((len(nonpivot_cols),)):
            return xp
        else:
            return None

    def _reduced_row_echelon_form(self, augmatrix: np.ndarray):
        for row, col in reversed(self.pivots):
            vals = augmatrix[row, :]

            for i in range(row):
                augmatrix[i, :] -= augmatrix[i, col] * vals

    def _row_echelon_form(self, augmatrix: np.ndarray):
        row = 0
        for col in range(self.cols):
            self._swap_first_nonzero(augmatrix, row, col)

            if (x := augmatrix[row, col]) != 0:
                augmatrix[row, :] /= x
                self._reduce_below(augmatrix, row, col)
                self.pivots.append((row, col))

                row += 1
                if row == self.rows:
                    break

    def _reduce_below(self, augmatrix: np.ndarray, row: int, col: int):
        vals = augmatrix[row, :]
        row += 1

        while row < self.rows:
            augmatrix[row, :] -= augmatrix[row, col] * vals
            row += 1

    def _swap_first_nonzero(self, augmatrix: np.ndarray, row: int, col: int):
        if augmatrix[row, col] != 0:
            return

        for i in range(row + 1, self.rows):
            if augmatrix[i, col] != 0:
                augmatrix[[row, i]] = augmatrix[[i, row]]
                break

    def _augment(self) -> np.ndarray:
        return np.c_[self.A, self.b]

    def _get_pivot_cols(self, container: Callable) -> list[int]:
        return container(col for _, col in self.pivots)


if __name__ == "__main__":
    A = np.array([
        [-2, 4, -2, 1, 4],
        [4, -8, 3, -3, 1],
        [1, -2, 1, -1, 1],
        [1, -2, 0, -3, 4]
    ])

    b = np.array([-3, 2, 0, 1])

    solver = Solver(A, b)
    xp, null_space = solver.solve()
    print("Particular solution:", xp)
    print("Null space:", null_space)
