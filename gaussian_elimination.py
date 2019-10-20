import numpy as np


# def


# def permute_matrix(A: np.ndarray, method: str) -> np.ndarray:
#     assert A.shape[0] == A.shape[1], "Matrix is not squared."

#     if method


def gaussian_elim(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    print("Input Augmented Matrix:")
    m = np.concatenate([matrix, np.expand_dims(b, axis=1)], axis=1)
    print(m)
    size = m.shape[0]
    for j in range(size):
        for i in range(j + 1, size):
            if m[j, j] == 0.0:
                raise ValueError(
                    f"Pivot value is 0 in position {[j, j]}. Elimination cannot proceed."
                )
            multiplier = m[i, j] / m[j, j]
            m[i] = m[i] - multiplier * m[j]

    # zero small values
    eps = 1e-12
    m[np.abs(m) <= eps] = 0.0

    return m[:size, :size], m[:, -1]
    # return m


def backward_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert A.shape[0] == A.shape[1], "Matrix is not squared."
    size = len(A)
    x = np.empty(size, dtype=A.dtype)
    x[-1] = b[-1] / A[-1, -1]
    for i in reversed(range(0, size - 1)):
        x[i] = (b[i] - sum(A[i, (i + 1) :] * x[i + 1:])) / A[i, i]
    
    return x


if "__main__" == __name__:
    matrix = np.array([[4.0, 5.0, 9.0], [1.0, 3.0, 2.0], [9.0, 2.0, 3.0]])
    b = np.array([1, 2, 3])
    A,b = gaussian_elim(matrix, b)
    x = backward_substitution(A, b)
    print("Output Augmented Matrix:")
    print(A)
    print("Solution:")
    print(x)
# gaussian_elim(np.array([[1,2,3], [1,2,3]]))
