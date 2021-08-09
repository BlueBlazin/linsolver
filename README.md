A linear equation solver using gaussian elimination. Implemented for fun and learning/teaching.

The solver will solve equations of the type:

<!-- $$
\mathbf{A}\mathbf{x} = \mathbf{b}
$$ -->

<div align="center"><img style="background: white;" src="svg/Mv09d16ZdS.svg"></div>

**A** can be rectangular and/or singular. The solver will return a particular solution as well as a list of vectors spanning the null space of **A** required for the general solution.

## How it works

The solution is acquired following these steps:

1. The solver creates the augmented matrix `[A | b]`.
2. The augmetned matrix is reduced to row-echelon form (REF) using Gaussian elimination.
3. If the number of pivots are fewer than the number of nonzero entries in REF `b`, the solver exits.
4. The augmented matrix is then further reduced into a reduced row-echelon form (RREF).
5. The solver reads off the particular from the RREF.
6. Finally it finds the null space using back-substitution on the RREF.

## Example

```py
A = np.array([
    [-2, 4, -2, 1, 4],
    [4, -8, 3, -3, 1],
    [1, -2, 1, -1, 1],
    [1, -2, 0, -3, 4]
])

b = np.array([-3, 2, 0, 1])

solver = Solver(A, b)
xp, null_space = solver.solve()
```
