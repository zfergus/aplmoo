"""
Use Affine null space to preform A Priori Lexicographical Multi-Objective
Optimization.

Written by Zachary Ferguson and Alec Jacobson.
"""

import pdb

import itertools

import numpy
import scipy.sparse

from affine_null_space import affine_null_space


def NullSpaceMethod(H, f, method="qr", bounds=None):
    """
    LEXMIN Solve the multi-objective minimization problem:

        min  {E1(x), E2(x), ... , Ek(x)}
         x

    where

        Ei = 0.5 * x.T * H[i] * x + x.T * f[i]

    and Ei is deemed "more important" than Ei+1 (lexicographical ordering):
    https://en.wikipedia.org/wiki/Multi-objective_optimization#A_priori_methods

    Inputs:
        H - k-cell list of n by n sparse matrices, so that H[i] contains the
            quadratic coefficients of the ith energy.
        f - k-cell list of n by 1 vectors, so that f[i] contains the linear
            coefficients of the ith energy
    Outputs:
        Z - n by 1 solution vector
    """

    # import scipy.io
    # scipy.io.savemat( "NullSpaceMethod.mat", { 'H': H, 'f': f } )
    # print( "Saved: NullSpaceMethod.mat" )

    k = len(H)
    assert k > 0
    assert k == len(f)

    n = H[0].shape[0]
    assert n == H[0].shape[1]
    assert n == f[0].shape[0]

    # Start with "full" search space and 0s as feasible solution
    # N = 1;% aka speye(n,n)
    N = scipy.sparse.identity(n)
    # Z = zeros(n,1);
    Z = numpy.zeros(f[0].shape)
    # For i in range(k)
    for Hi, fi in itertools.izip(H, f):
        # Original ith energy: 0.5 * x.T * Hi * x + x.T * fi

        # Restrict to running affine subspace, by abuse of notation:
        #       x = N*y + z
        # fi = N' * (Hi * Z + fi)
        fi = N.T.dot(Hi.dot(Z) + fi)
        # Hi = N'*Hi*N
        Hi = N.T.dot(Hi.dot(N))

        # Sparse QR Factorization
        # [Ni,Y] = affine_null_space(Hi,-fi,'Method',null_space_method)
        # Ni is the null space of Hi
        # Y is a solution to Hi * x = fi
        Ni, Y = affine_null_space(Hi, -fi, method=method, bounds=bounds)

        if(len(Y.shape) < 2):
            Y = Y.reshape(-1, 1)

        # Update feasible solution
        Z = N.dot(Y) + Z
        if(Ni.shape[1] == 0):
            # Z is full determined, exit loop early
            break

        # Otherwise, N spans the null space of Hi
        N = N.dot(Ni)

        # Update the bounds
        if not (bounds is None):
            # bounds = (-Z, 1-Z)
            val = N.dot(numpy.zeros(N.shape[1])).reshape(-1, 1)
            bounds = (-val, 1 - val)

    # (If i<k then) the feasible solution Z is now the unique solution.

    # E = numpy.zeros((k, f[0].shape[1]))
    # for i in range(k):
    #     Hi, fi = H[i], f[i]
    #     # E(i) = 0.5*(Z'*(H{i}*Z)) + Z'*f{i};
    #     E[i] = (0.5 * (Z.T.dot(Hi.dot(Z))) + Z.T.dot(fi)).diagonal()

    # return Z, E
    return Z

if __name__ == "__main__":
    import time_aplmoo_method
    time_aplmoo_method.time_aplmoo_method(NullSpaceMethod, print_energy=True)
