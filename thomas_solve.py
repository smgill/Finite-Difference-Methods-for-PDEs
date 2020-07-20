import numpy as np

def thomas_solve(l1, u0, u1, b):
    '''
    Solve a tridiagonal matrix equation with the Thomas algorithm.

    The coefficient matrix in the tridiagonal matrix equation Ax = b must be
    decomposed into lower and upper triangular matrices such that A = LU.

    Parameters
    ----------
    l1 : ndarray
        The subdiagonal of the L matrix.
    u0 : ndarray
        The main diagonal of the U matrix.
    u1 : ndarray
        The superdiagonal of the U matrix.
    b : ndarray
        The right-hand side vector.

    Returns
    -------
    x : ndarray
        The system's solution vector.
    '''

    # Preallocate a solution vector:
    n = len(b)
    x = np.empty(n)

    # Forward substitution:
    for i in range(1, n):
        b[i] -= l1[i - 1]*b[i - 1]

    # Back substitution:
    x[-1] = b[-1]/u0[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - u1[i]*x[i + 1])/u0[i]
    return x