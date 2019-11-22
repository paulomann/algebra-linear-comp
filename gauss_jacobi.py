import numpy as np
from typing import Tuple
import sys
from numpy import linalg

np.set_printoptions(suppress=True)

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

def do_matrix_pivoting(A: np.ndarray, b:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    size = A.shape[0]
    permuted_A = A
    col_permutation = np.eye(len(A))
    for i in range(0, A.shape[0] - 1):
        permuted_A, row_perm, col_perm = complete_pivoting(permuted_A, i)
        if col_perm is not None:
            col_permutation = col_perm.transpose() @ col_permutation
        if row_perm is not None:
            b = row_perm @ b
    return permuted_A, b, col_permutation


def jacobi(
    orig_A: np.ndarray, orig_b: np.ndarray, x0=None, tol=10e-3, n_iter=100
) -> np.ndarray:
    assert orig_A.shape[0] == orig_A.shape[1], "Matrix is not squared."
    print("Input Matrix:")
    print(orig_A)
    print(orig_b)
    x0 = x0 if x0 is not None else np.zeros(orig_A.shape[1])
    x = np.zeros(orig_A.shape[1], dtype="double")
    A = orig_A.copy().astype("double")
    b = orig_b.copy().astype("double")
    x0 = x0.astype("double")  
    # A, b, col_permutation = do_matrix_pivoting(A, b)
    n = A.shape[1]
    print("After Pivoting: ")
    print(A)
    print(b)
    print("="*40)

    if 0 in A.diagonal():
        print("Cannot solve this matrix when 0 is in diagonal of A.")
        return x

    for k in range(0, n_iter):
        for i in range(0, n):
            x[i] = (b[i] - sum([A[i,j] * x0[j] for j in range(0, n) if j != i]))/A[i,i]
        print(x)
        if np.linalg.norm(x - x0, ord=float("inf")) / np.linalg.norm(x, ord=float("inf")) < tol:
            print(f"Threshold {tol} reached in iteration {k}. Stopping iterative process...")
            # x = x @ col_permutation
            return x
        x0[:] = x
    
    raise NameError("Algorithm does not converge.")

if "__main__" == __name__:
    print("=" * 23)
    print(f"Executing Gauss-Jacobi.")
    print("=" * 23)
    # matrix = np.array([[4.0, 5.0, 9.0], [1.0, 3.0, 2.0], [9.0, 2.0, 3.0]])
    # b = np.array([1, 2, 3])
    # matrix = np.array([[4, 6, 9, 12], [0, 0, 50, 13], [5, 32, 4, 31], [13, 0, 14, 5]])
    # b = np.array([40, 20, 10, 5])
    # matrix = np.array([[4.0, 5.0], [1.0, 3.0]])
    # b = np.array([4, 5])
    # matrix = np.array([[1.,-1.,1.,-1.],[1.,0.,0.,0.],[1.,1.,1.,1.],[1.,2.,4.,8.]])
    # b =  np.array([14., 4. , 2. , 2.])
    # matrix = np.array(
    #     [
    #         [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    #         [1.00, 0.63, 0.39, 0.25, 0.16, 0.10],
    #         [1.00, 1.26, 1.58, 1.98, 2.49, 3.13],
    #         [1.00, 1.88, 3.55, 6.70, 12.62, 23.80],
    #         [1.00, 2.51, 6.32, 15.88, 39.90, 100.28],
    #         [1.00, 3.14, 9.87, 31.01, 97.41, 306.02],
    #     ]
    # )
    # b = np.array([-0.01, 0.61, 0.91, 0.99, 0.60, 0.02])
    # matrix = np.array([[2.0,1.0],[5.0,7.0]])
    # b = np.array([11.0,13.0])
    matrix = np.array([[-3.0, 1.0, 1.0],[2.0, 5.0, 1.0], [2.0, 3.0, 7.0]])
    b = np.array([2.0, 5.0, -17.0])
    x = jacobi(matrix, b, np.zeros(b.size), 10e-6, 50)
    print(f"Final solution: {x}")