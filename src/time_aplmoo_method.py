"""
Utility function for timing a given APLMOO method.
Written by Zachary Ferguson
"""

from __future__ import print_function

import time

import numpy
import scipy.sparse

from random_matrix import gen_symmetric_positive_definite_matrix as \
    gen_matrix


def time_aplmoo_method(method, irange=None, print_energy=False):
    """
    Time the given method for various matrix of various size (n x n).
    Inputs:
        method - the APLMOO method to test.
        irange - values of i s.t. n = 2^i (Default: range(1, 11))
        print_energy - should the energies be printed at the end?
    """

    if(irange is None):
        irange = range(1, 11)

    print("%4s %8s" % ("n", "t"))

    energies = []
    for i in irange:
        n = 2**i

        # Generate a singular matrix
        A = scipy.sparse.lil_matrix((n, n))
        A[:n // 2, :n // 2] = gen_matrix(n // 2) # Upper Left block matrix
        A = A.tocsc()

        B = scipy.sparse.lil_matrix((n, n))
        B[n // 2:, n // 2:] = gen_matrix(n // 2) # Lower Right block matrix
        B[-1] = 0 # Make the last row all zeros
        B = B.tocsc()

        # Generate a rhs that will always have a solution
        a = A.dot(numpy.ones((n, 1)))
        b = B.dot(numpy.ones((n, 1)))

        C = scipy.sparse.identity(n).tocsc()
        c = 0.2053202792 * numpy.ones((n, 1))

        startTime = time.time()
        Z = method([A, B, C], [a, b, c])
        print("%4d %.6f" % (n, time.time() - startTime))

        if(print_energy):
            energies.append([
                abs(Z.T.dot(A.dot(Z)) + Z.T.dot(a)),
                abs(Z.T.dot(B.dot(Z)) + Z.T.dot(b)),
                abs(Z.T.dot(C.dot(Z)) + Z.T.dot(c))])

    if(print_energy):
        print()
        for i, ie in enumerate(irange):
            print("n = %d" % 2**ie)
            print(("Z.T @ A @ Z + Z.T @ a = %g\nZ.T @ B @ Z + Z.T @ b = %g\n" +
                "Z.T @ C @ Z + Z.T @ c = %g") % tuple(energies[i]))
