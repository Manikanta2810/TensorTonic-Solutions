import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    m,n=A.shape
    T = np.zeros((n,m),dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            T[j][i]= A[i][j]
    A = T
    return A
