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

The columns of $Q_1$ span the column space of A, $col(A^T) = row(A)$, and the
columns of $Q_2$ span the null space of A, $null(A)$. $R_1$ is a $r \times r$
matrix where $r$ is the rank of A. We use QR decomposition with a column
pivoting to get a permutation matrix, $P$, so we can easily compute the rank of
A.

So finding $N$ a matrix whose columns span the null space of A is simple.

\begin{equation}
N = Q_2 = Q_{:, r:}
\end{equation}

To find a particular solution to $A x = b$ we solve a linear system

\begin{equation}
x_0 = Q_1 y = Q_1 (R_1^T)^{-1} (P^T b)
\end{equation}

That is we find the solution to $R_1^T y = P^T b$ and transform it to the
column space of $A$.

#### Proof: $A^TQ_2^Ty \equiv 0 \ \forall y$


$\\ \\ A^T = QR \Leftrightarrow A = R^TQ^T = \begin{bmatrix}\hat{R} \ 0\end{bmatrix}
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
perform multi-objective optimization on all $E_i$.

\begin{equation}
N_0 = I
\end{equation}
\begin{equation}
z_0 = 0
\end{equation}

\begin{equation}
\bar{N_i}, x_i = \text{$AffineNullSpace$}(N_{i-1}^TH_iN_{i-1}, \
-N_{i-1}^T(H_iz_{i-1}+f_i))
\end{equation}

\begin{equation}
z_i = N_{i-1}x_i + z_{i-1}
\end{equation}
\begin{equation}
N_i = N_{i-1} \bar{N_{i}}
\end{equation}

Where `AffineNullSpace` is one of the functions defined in section one. We
repeat this processes until either we have run out of energies or $\bar{N_i}$
is of size $(0 \times 0)$. The resulting solution is the final $z$.

For example,

\begin{equation}
\begin{matrix}
\bar{N_1}, x_1 = AffineNullSpace(H_1, \ -f_1) \\
z_1 = Ix_1 + 0 = x_1 \\
N_1 = I\bar{N_1} = \bar{N_1} \\
\\
\bar{N_2}, x_2 = AffineNullSpace(N_1H_2N_1, \ -N_1^T(H_2z_1 + f_2)) \\
z_2 = N_1 x_2 + z_1 = \bar{N_1}x_2 + x_1 \\
N_2 = N_1\bar{N_2} = \bar{N_1}\bar{N_2} \\
\\
\bar{N_3}, x_3 = AffineNullSpace(N_2H_3N_2, \ -N_2^T(H_3z_2 + f_3)) \\
z_3 = N_2 x_3 + z_2 = \bar{N_1}\bar{N_2}x_3 + \bar{N_1}x_2 + x_1 \\
N_3 = N_2\bar{N_3} = \bar{N_1}\bar{N_2}\bar{N_3} \\
\end{matrix}
\end{equation}

With each iteration we find a minimum solution for the current energy.
Importantly, this new solution preserves the energy value of the previous
solution for all preceding energies.

### Proof

$\left ( {d \over dx} E_1(x) \right ) = H_1x + f_1 = 0$

$x_1$ is a particular solution to $H_1x = -f_1$ and a minimal energy solution
to $E_1(x)$.

$N_1y + x_1$ is a parameterization of all minimal energy solutions for $E_1(x)$.

$H_1(N_1y + x_1) = H_1N_1y + H_1x_1 = 0 + H_1x_1 = -f_1$

**Prove that $z_2$ is a minimal energy solution to $E_1(x)$:**

$E_2(x) = {1 \over 2}x^TH_2x + x^Tf_2 + c_2$

$E_2(N_1y + x_1) = {1 \over 2}(N_1y + x_1)^TH_2(N_1y + x_1) + (N_1y + x_1)^Tf_2 + c_2$

$E_2(N_1y + x_1) = {1 \over 2}y^TN_1^TH_2N_1y + y^TN_1^TH_2x_1 + y^TN_1^Tf_2 + {1 \over 2}x_1^TH_2x_1 + x_1^Tf_2 + c_2$

$\left ( {d \over dx} E_2(N_1y + x_1) \right ) = N_1^TH_2N_1y + N_1^T(H_2x_1 + f_2)$

$x_2$ is a particular solution to $N_1^TH_2N_1y = -N_1^T(H_2x_1 + f_2)$

$z_2 = N_1x_2 + x_1$

$H_1z_2 = H_1N_1x_2 + H_1x_1 = -f_1$

### Example

To illustrate this better let us take two energies in 3D

$E_1(x, y) = z = (y+7)^2$

$E_2(x, y) = z = x^2 + y^2$

The minimal solutions to $E_1$ can be paramaterized as
$N_1\vec{w} + \vec{x_1} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \vec{w} + \begin{bmatrix} -7 \\ -7 \end{bmatrix}$.

Substituting this parameterization for $(x, y)$ in $E_2$ we get the following:

$E_2(x, y) = (w_0 - 7)^2 + (-7)^2 = w_0^2 - 14w_0 + 49 + 49$

${d \over dw} E_2 = 2w_0 - 14 = 0 \Rightarrow w_0 = 7$

$\therefore z_2 = N_1\begin{bmatrix} 7 \end{bmatrix} + z_1 = \begin{bmatrix} 0 \\ -7 \end{bmatrix}$

$z_2$ is the minimal energy value for $E_2$ that is in the null space of $E_1$.
