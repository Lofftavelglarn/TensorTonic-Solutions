import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    B = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            B[j][i] = A[i][j]
    return B
