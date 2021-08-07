import numpy as np


class Solver:
    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self.A = A.astype(np.float64)
        self.b = b.astype(np.float64)
        self.rows = A.shape[0]
        self.cols = A.shape[1]
        self.pivots: list[tuple[int, int]] = []

    def solve(self):
        # create the augmented matrix from `A` and `b`
        augmatrix = self._augment()
        print(augmatrix)
        print("=============================")
        # get the row echelon form of the agumented matrix
        self._row_echelon_form(augmatrix)
        print(augmatrix)
        print("=============================")
        # get the reduced row echelon form
        self._reduced_row_echelon_form(augmatrix)
        print(augmatrix)
        # get a particular solution
        bp = self._particular_sol(augmatrix)
        print(self.A @ bp)
        print("pivots:", self.pivots)

    def _particular_sol(self, augmatrix: np.ndarray):
        pivot_cols = [col for _, col in self.pivots]

        bp = np.zeros((self.cols,))
        bp[pivot_cols] = augmatrix[:, -1]

        return bp

    def _reduced_row_echelon_form(self, augmatrix: np.ndarray):
        for row, col in reversed(self.pivots):
            vals = augmatrix[row, :]

            for i in range(row):
                augmatrix[i, :] -= augmatrix[i, col] * vals

    def _row_echelon_form(self, augmatrix: np.ndarray):
        row = 0
        for col in range(self.cols):
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

    def _augment(self) -> np.ndarray:
        return np.c_[self.A, self.b]


if __name__ == "__main__":
    A = np.array([
        [-2, 4, -2, 1, 4],
        [4, -8, 3, -3, 1],
        [1, -2, 1, -1, 1],
        [1, -2, 0, -3, 4]
    ])

    b = np.array([-3, 2, 0, 1])
    solver = Solver(A, b)
    solver.solve()
