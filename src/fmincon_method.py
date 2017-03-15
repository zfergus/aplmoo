"""
Solve the multi-objective optimization using constraints of the previous
energies.

Written by Zachary Ferguson
"""

import numpy
import scipy
import scipy.optimize


def fminconMethod(H, f):
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
        z - n by 1 solution vector
    """
    E = []

    # Function handles used by fmincon
    def objective(x, i):
        return (0.5 * x.T.dot(H[i]).dot(x) + x.T.dot(f[i]), H[i].dot(x) + f[i])

    # function [ineqval,eqval,ineqgrad,eqgrad] = constraints(X,i)
    def constraints(x, i):
        if i == 0:
            return (numpy.array([]), numpy.array([]), numpy.array([]),
                numpy.array([]))

        ineqval, eqval, ineqgrad, eqgrad = constraints(x, i - 1)
        fval, grad = objective(X, i - 1)
        # values in the rows
        import pdb; pdb.set_trace()
        if(ineqval.shape[0] == 0):
            ineqval = fval - E(i - 1)
        else:
            ineqval  = numpy.vstack([ineqval, fval - E(i - 1)])
        # grads in the columns
        if(ineqgrad.shape[0] == 0):
            ineqgrad = grad
        else:
            ineqgrad = numpy.hstack([ineqgrad, grad])

    # function hess = hessian(X,lambda,i)
    def hessian(x, func, i):
        hess = H[i]
        # for j = 2:i
        for j in range(1, i):
            hess = hess + func.ineqnonlin(j - 1).dot(H[j - 1])
        return hess

    # Number of energies
    k = len(H)
    # Dimension of search space
    n = H[0].shape[0]

    z = -f[0]
    Q = []
    G = []
    C = [lambda x: []]
    GC = [lambda x: []]
    HC = [lambda el: 0]
    for i in range(k):
        Q.append(lambda x: 0.5 * (x.T.dot(H[i].dot(x))) + x.T.dot(f[i]))
        G.append(lambda x: H[i].dot(x) + f[i])
        res = scipy.optimize.minimize(lambda x: objective(x, i),
            z, constraints=({"type": "ineq",
            "fun": lambda x: constraints(x, i)},))
        z = res.x
        E.append(res.fun)

    return z

if __name__ == "__main__":
    import time_aplmoo_method
    time_aplmoo_method.time_aplmoo_method(fminconMethod,
        print_energy=True)
