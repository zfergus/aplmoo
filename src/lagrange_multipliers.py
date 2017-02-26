"""
Use Lagrange multipliers to preform A Priori Lexicographical Multi-Objective
Optimization.

See math/lagrange_multipliers.pdf for a full explanation of the algorithm.

Written by Zachary Ferguson and Alec Jacobson.
"""

import pdb

import numpy
import scipy.linalg
import scipy.sparse


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
            pdb.set_trace()
            m = C.shape[0]
            if(i == (k - 1)):
                # pdb.set_trace()
                C = Q[:nc, :nc].T
                C = (Q[:nc, :nc].dot(R[:nc, :nc])).T
                D = D[E[:nc]]
                return numpy.linalg.solve(C, D)[:n]

        # A = sparse(size(C,1),size(C,1));
        A = numpy.zeros((m, m))
        # A(1:n,1:n) = H{i+1};
        A[:H[i + 1].shape[0], :H[i + 1].shape[1]] = H[i + 1]
        # B = zeros(size(C,1),1);
        B = numpy.zeros((m, D.shape[1]))
        # B(1:n) = -f{i+1};
        B[:f[i + 1].shape[0]] = -f[i + 1]
        # C = [A C';C sparse(size(C,2),size(C,2))];
        C = numpy.vstack([numpy.hstack([A, C.T]),
            numpy.hstack([C, numpy.zeros((C.shape[0], C.shape[0]))])])
        # D = [B;D];
        D = numpy.vstack([B, D])

if __name__ == "__main__":
    n = 100

    # Generate a singular matrix
    data = (9 * numpy.random.rand(n, n)).astype("int32")
    data2 = (9 * numpy.random.rand(n, n)).astype("int32")
    # Make sure the data matrix is singular
    # Convert to a sparse version
    A = scipy.sparse.csc_matrix(data)
    A[n // 2:] = 0
    assert abs(numpy.linalg.det(A.A)) < 1e-8
    B = scipy.sparse.csc_matrix(data2)
    B[:n // 2] = 0
    assert abs(numpy.linalg.det(B.A)) < 1e-8

    # Generate a b that will always have a solution
    a = A.dot(numpy.ones((n, 1)))
    b = B.dot(numpy.ones((n, 1)))

    C = scipy.sparse.identity(n).tocsc()
    c = 0.2053202792 * numpy.ones((n, 1))

    print("The following inputs define our quadratic energies:")
    print("\tx.T*A*x + x.T*b = 0\n")

    fstr = "%s:\n%s\n\n"
    print(("Inputs:\n\n" + 6 * fstr) % ("A", A.A, "a", a, "B", B.A, "b", b,
        "C", C.A, "c", c))
    print("Rank A = %d\nRank B = %d\n" % (numpy.linalg.matrix_rank(A.A),
        numpy.linalg.matrix_rank(B.A)))
    Z = LagrangeMultiplierMethod([A.A, C.A], [a, c])
    # Z = LagrangeMultiplierMethod([A.A, B.A, C.A], [a, b, c])
    print(("Outputs:\n\n" + fstr) % ("Z", Z))

    print("Z.T @ A @ Z - Z.T @ a = %f" % abs(Z.T.dot(A.dot(Z)) - Z.T.dot(a)))
    print("Z.T @ B @ Z - Z.T @ b = %f" % abs(Z.T.dot(B.dot(Z)) - Z.T.dot(b)))
    print("Z.T @ C @ Z - Z.T @ c = %f" % abs(Z.T.dot(C.dot(Z)) - Z.T.dot(c)))
