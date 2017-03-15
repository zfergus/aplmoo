# Null Space Method

Solve the multi-objective minimization problem:

\begin{equation}
min_x  {E1(x), E2(x), \dots , Ek(x)}
\end{equation}
where
\begin{equation}
E_i = 0.5 x^T H_i x + x^T f_i
\end{equation}

and $E_i$ is deemed "more important" than $E_{i+1}$ (lexicographical ordering).

## Computing the Affine Null Space

First we need to find a basis for the null space and a particular solution
$x_i$ to the equation $A x = b$.

### QR

First, we compute the QR decomposition of $A^T$

\begin{equation}
P A^T = Q R = \begin{bmatrix} Q_1 Q_2 \end{bmatrix}
\begin{bmatrix} R_1 R_2 \\ 0 \end{bmatrix}
\end{equation}

The columns of $Q_1$ span the $col(A^T) = row(A)$, and the columns of $Q_2$
span the $null(A)$. $R_1$ is a $r \times r$ matrix where $r$ is
$rank(A^T)$. Therefore,

\begin{equation}
N = Q_2 = Q_{:, r:}
\end{equation}

To find a particular solution to $A x = b$ we solve a linear system

\begin{equation}
x_0 = Q_1 y = Q_1 (R_1^T)^{-1} (P^T b)
\end{equation}

That is we find the solution to $R_1^T y = P^T b$ and transform it to the column
space of $A$.

#### Proof: $A^TQ_2^Ty \equiv 0 \ \forall y$

$A^T = QR \Leftrightarrow A = R^TQ^T = \begin{bmatrix}\hat{R} \ 0\end{bmatrix}
\begin{bmatrix} Q_1 \\ Q_2 \end{bmatrix}$

$\begin{bmatrix}\hat{R} \ 0\end{bmatrix}
\begin{bmatrix} Q_1 \\ Q_2 \end{bmatrix} Q_2^T y =
\begin{bmatrix}\hat{R} \ 0\end{bmatrix}
\begin{bmatrix} Q_1Q_2^T \\ Q_2Q_2^T \end{bmatrix} y =
\begin{bmatrix}\hat{R} \ 0\end{bmatrix}
\begin{bmatrix} 0 \\ Q_2Q_2^T \end{bmatrix} y = 0 y = 0$

$Q_1Q_2^T = 0$ because $Q$ is an orthogonal matrix.

### SVD

First, we compute the singular value decomposition of $A$

\begin{equation}
A = U \Sigma V^T
\end{equation}

where $\Sigma$ is a diagonal matrix containing the singular values of $A$.
The null space of $A$ is spanned by the vectors in $V$ corresponding to zero
values in $\Sigma$.

\begin{equation}
N = V_{:, s}
\end{equation}

where s is the set of indices for zeros along the diagonal of $\Sigma$.

Next to find a particular solution to $H_ix = b$ we invert the SVD.

\begin{equation}
x_0 = A^{-1} b = (U \Sigma V^T)^{-1} b = (V \Sigma^{+} U^T) b
\end{equation}

where $\Sigma^+$ is the Moore-Penrose pseudoinverse of $\Sigma$. Note, U
and V are orthogonal matrices, so their transpose is their inverse.

### LUQ

(See `luq-decomposition.pdf`)

## Multi-Objective Optimization

Using one of the above method for computing the affine null space we can
preform multi-objective optimization on all $E_i$.

\begin{equation}
N_0 = I
\end{equation}
\begin{equation}
z_0 = 0
\end{equation}

\begin{equation}
\bar{N_i}, x_i = \text{$AffineNullSpace$}(N_i^TH_iN_i, \ N_i^TH_iz_i+f_i)
\end{equation}

\begin{equation}
z_{i+1} = N_ix_i + z_i
\end{equation}
\begin{equation}
N_{i+1} = N_i \bar{N_{i}}
\end{equation}

Where `AffineNullSpace` is one of the functions defined in section one. We
repeat this processes until either we have run out of energies or $\bar{N_i}$
is of size $(0 \times 0)$. The resulting solution is the final $z$.

For example,

\begin{equation}
\begin{matrix}
\bar{N_0}, x_0 = AffineNullSpace(H_0, \ f_0) \\
z_1 = x_0 \\
N_1 = \bar{N_0} \\
\\
\bar{N_1}, x_1 = AffineNullSpace(\bar{N_0^T}H_1\Bar{N_0}, \bar{N_0^T}H_1x_0 + f_1) \\
z_2 = N_1 x_1 + z_1 = \bar{N_0}x_1 + x_0 \\
N_2 = N_1\bar{N_1} = \bar{N_0}\bar{N_1} \\
\\
\bar{N_2}, x_2 = AffineNullSpace(...) \\
z_3 = N_2 x_2 + z_2 = \bar{N_0}\bar{N_1}x_2 + \bar{N_0}x_1 + x_0 \\
N_3 = N_2\bar{N_2} = \bar{N_0}\bar{N_1}\bar{N_2} \\
\end{matrix}
\end{equation}

With each iteration we find a minimum solution for the current energy.
Importantly, this new solution preserves the energy value of the previous
solution for all preceding energies.
