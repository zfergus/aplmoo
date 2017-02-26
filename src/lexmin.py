#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
A Priori Lexicographical Multi-Objective Optimization. 🍎🐮

Written by Alec Jacobson and Zachary Ferguson.
"""

import null_space_method
import lagrange_multipliers


def a_priori_lexmin(H, f, options=None):
    """
    Preforms a priori lexicographical multi-objective optimization on the given
    list of energies.
    Inputs:
        H - List of energies to minimize.
        f - List of rhs for the energies in H
        options - optional dictionary of arguments for the "method", and
            methods options.
            Default: None -> method = NullSpaceMethod
            Options:
                method: {"NullSpaceMethod", "LagrangeMultiplierMethod"}
                NullSpaceMethod: {"qr", "luq"}
    Output:
        z - minimium solution vector
    """
    if options is None or "method" not in options:
        method = "NullSpaceMethod"
    else:
        method = options["method"]

    if method == "NullSpaceMethod":
        if not (options is None) and "NullSpaceMethod" in options:
            return null_space_method(H, f, method=options["NullSpaceMethod"])
        return null_space_method.NullSpaceMethod(H, f)
    elif method == "LagrangeMultiplierMethod":
        return lagrange_multipliers.LagrangeMultiplierMethod(H, f)
    else:
        raise ValueError("Invalid method of preforming APLMOO: %s" % method)

if __name__ == "__main__":
    import numpy
    import scipy.sparse

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
    Z = a_priori_lexmin([A, B, C], [-a, -b, -c])
    print(("Outputs:\n\n" + fstr) % ("Z", Z))

    print("Z.T @ A @ Z - Z.T @ a = %f" % abs(Z.T.dot(A.dot(Z)) - Z.T.dot(a)))
    print("Z.T @ B @ Z - Z.T @ b = %f" % abs(Z.T.dot(B.dot(Z)) - Z.T.dot(b)))
    print("Z.T @ C @ Z - Z.T @ c = %f" % abs(Z.T.dot(C.dot(Z)) - Z.T.dot(c)))
