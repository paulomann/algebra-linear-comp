import numpy as np
from typing import Tuple
np.set_printoptions(precision=2)


def backward_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    size = len(A)
    x = np.empty(size, dtype=A.dtype)
    x[-1] = b[-1] / A[-1, -1]
    for i in reversed(range(0, size - 1)):
        x[i] = (b[i] - sum(A[i, (i + 1) :] * x[i + 1 :])) / A[i, i]

    return x

def partial_pivoting(
    A: np.ndarray, col: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    size = len(A)
    perm = np.eye(size)
    # print(col, A[col:, col])
    row_idx = np.argmax(np.abs(A[col:, col])) + col
    # print(row_idx)
    perm[col], perm[col, row_idx] = 0.0, 1.0
    perm[row_idx], perm[row_idx, col] = 0.0, 1.0
    return (perm @ A, perm, None)


def complete_pivoting(
    A: np.ndarray, col: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    # Get submatrix
    A2 = A[col:, col:]
    size = len(A)
    perm_row = np.eye(size)
    perm_col = np.eye(size)
    row_idx, col_idx = np.unravel_index(np.abs(A2).argmax(), A2.shape)
    row_idx += col
    col_idx += col
    # Row Permutation Matrix
    if row_idx != col:
        perm_row[col], perm_row[col, row_idx] = 0.0, 1.0
        perm_row[row_idx], perm_row[row_idx, col] = 0.0, 1.0
    # Column Permutation Matrix
    if col_idx != col:
        perm_col[:, col], perm_col[col_idx, col] = 0.0, 1.0
        perm_col[:, col_idx], perm_col[col, col_idx] = 0.0, 1.0
    
    return (perm_row @ A @ perm_col, perm_row, perm_col)


def do_matrix_pivoting(
    m: np.ndarray, col: int, method: str
) -> Tuple[np.ndarray, np.ndarray]:
    size = m.shape[0]
    A, b = m[:size, :size], m[:, -1]
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    if col == size - 1:
        return m, np.eye(size), np.eye(size)
    permuted_A, row_perm, col_perm = globals()[method + "_pivoting"](A, col)
    if row_perm is not None:
        b = row_perm @ b
    return (
        np.concatenate([permuted_A, np.expand_dims(b, axis=1)], axis=1),
        row_perm,
        col_perm
    )


def gaussian_elim(A: np.ndarray, b: np.ndarray, pivoting_method="partial") -> np.ndarray:
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    print("Input Augmented Matrix:")
    m = np.concatenate([matrix, np.expand_dims(b, axis=1)], axis=1)
    eps = 1e-12
    print(m)
    col_permutation = np.eye(len(A))
    size = m.shape[0]
    for j in range(size):

        if pivoting_method != "default":
            # print(m)
            m, _, col_perm = do_matrix_pivoting(m, j, pivoting_method)
            # print(m)
            if col_perm is not None: col_permutation = col_permutation @ col_perm

        for i in range(j + 1, size):
            if m[j, j] == 0.0:
                raise ValueError(
                    f"Pivot value is 0 in position {[j, j]}. Elimination cannot proceed."
                )
            multiplier = m[i, j] / m[j, j]
            m[i] = m[i] - multiplier * m[j]
            m[np.abs(m) <= eps] = 0.0

    A = m[:size, :size]
    b = m[:, -1]
    x = backward_substitution(A, b) @ col_permutation
    print("Output Augmented Matrix:")
    print(m)
    print("Solution:")
    print(x)
    return x

if "__main__" == __name__:
    # matrix = np.array([[4.0, 5.0, 9.0], [1.0, 3.0, 2.0], [9.0, 2.0, 3.0]])
    # b = np.array([1, 2, 3])
    matrix = np.array([[4, 6, 9, 12], [0, 0, 50, 13], [5, 32, 4, 31], [13, 0, 14, 5]])
    b = np.array([40, 20, 10, 5])
    x = gaussian_elim(matrix, b, pivoting_method="complete")
