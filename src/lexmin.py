#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
A Priori Lexicographical Multi-Objective Optimization. 🍎🐮

Written by Alec Jacobson and Zachary Ferguson.
"""

import pdb

import itertools

import numpy
import scipy.sparse

from affine_null_space import affine_null_space


def LagrangeMultiplierMethod(H, f):
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
    Note:
        This method is bad for multiple reasons:
            1. qr factorization on the ith set of constraints will be very
               expensive,
            2. this is due to the fact that the number of non-zeros will be
               O(n2^i), and
            3. the eventual solve will not necessarily behave well because the
               constraints are not full rank (!duh!).
    """

    if(scipy.sparse.issparse(H[0])):
        raise Exception()

    k = len(H)
    n = H[0].shape[0]

    # C = H{1};
    C = H[0]
    # D = -f{1};
    D = f[0]

    # for i = 1:k
    for i in range(k):
        # [Q,R,E] = qr(C');
        Q, R, E = scipy.linalg.qr(C.T, pivoting=True)
        # nc = find(any(R,2),1,'last');
        nonzero_rows = R.nonzero()[0]
        if(nonzero_rows.shape[0] > 0):
            nc = nonzero_rows[-1] + 1
        else:
            nc = 0
        # if nc == size(C,1) || i==k
        if nc == C.shape[0]:
            # Z = C \ D;
            # Z = Z(1:n);
            pdb.set_trace()
            return numpy.linalg.solve(C, D)[:n]
        else:
            # col(Q) = col(A.T) = row(A)
            m = C.shape[0]
            if(i == (k - 1)):
                pdb.set_trace()
                C = Q[:nc, :nc].T
                C = (Q[:nc, :nc].dot(R[:nc, :nc])).T
                D = D[E[:nc]]
                return numpy.linalg.solve(C, D)[:n]

        #######################################################################
        #   min               ½x'Hix + x'fi + 0*λ₁ + 0*λ₂ + ... + 0*λi-1
        #    x,λ₁,λ₂,...λi-1
        #    s.t.             Ci-1 * [x; λ₁; λ₂; ...; λi-1] = Di-1
        # or
        #   min           ½[x;λ₁;...;λi-1]'A[x;λ₁;...;λi-1] - [x;λ₁;...;λi-1]'B
        #    x,λ₁,λ₂,...λi-1                                ^
        #                                                   |----------negative
        #   s.t.              Ci-1 * [x; λ₁; λ₂; ...; λi-1] = Di-1
        #######################################################################

        # A = sparse(size(C,1),size(C,1));
        A = numpy.zeros((m, m))
        # A(1:n,1:n) = H{i+1};
        A[:n, :n] = H[i + 1]
        # B = zeros(size(C,1),1);
        B = numpy.zeros((m, D.shape[1]))
        # B(1:n) = -f{i+1};
        B[:n] = -f[i + 1]
        # C = [A C';C sparse(size(C,2),size(C,2))];
        C = numpy.vstack([numpy.hstack([A, C.T]),
            numpy.hstack([C, numpy.zeros((C.shape[0], C.shape[0]))])])
        # D = [B;D];
        D = numpy.vstack([B, D])


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
    n = 100

    # Generate a singular matrix
    data = (9 * numpy.random.rand(n, n)).astype("int32")
    # Make sure the data matrix is singular
    # assert abs(numpy.linalg.det(data)) < 1e-8
    # Convert to a sparse version
    A = scipy.sparse.csc_matrix(data)
    A[-3:] = 0
    B = scipy.sparse.csc_matrix(data)
    B[-1] = 0

    # Generate a b that will always have a solution
    a = A.dot(numpy.ones((n, 1)))
    b = B.dot(numpy.ones((n, 1)))

    C = scipy.sparse.identity(n).tocsc()
    c = 0.2053202792 * numpy.ones((n, 1))

    print("The following inputs define our quadratic energies:")
    print("\tx.T*A*x + x.T*b = 0\n")

    fstr = "%s:\n%s\n\n"
    print(("Inputs:\n\n" + 4 * fstr) % ("A", A.A, "b", b, "C", C.A, "c", c))
    Z = NullSpaceMethod([A, B, C], [-a, -b, -c])
    Z = LagrangeMultiplierMethod([A.A, C.A], [a, c])
    Z = LagrangeMultiplierMethod([A.A, B.A, C.A], [a, b, c])
    print(("Outputs:\n\n" + fstr) % ("Z", Z))

    print("Z.T @ A @ Z - Z.T @ a = %f" % abs(Z.T.dot(A.dot(Z)) - Z.T.dot(a)))
    print("Z.T @ B @ Z - Z.T @ b = %f" % abs(Z.T.dot(B.dot(Z)) - Z.T.dot(b)))
    print("Z.T @ C @ Z - Z.T @ c = %f" % abs(Z.T.dot(C.dot(Z)) - Z.T.dot(c)))
