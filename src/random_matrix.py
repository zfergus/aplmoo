"""
Utility functions for generating different types of matrices.

Written by Zachary Ferguson
"""

import scipy.sparse


def gen_symmetric_matrix(n, density=0.1):
    """ Generate a random symmetric matrix in sparse coo format. """
    M = scipy.sparse.random(n, n, density)
    return 0.5 * (M + M.T)


def gen_symmetric_positive_definite_matrix(n, density=0.1):
    """ Generate a random Symmetric Positivie Definite matrix. """
    return gen_symmetric_matrix(n, density) + n * scipy.sparse.identity(n)
