# Lagrange Multiplier Method

Solve the multi-objective minimization problem:

\begin{equation}
min_x  {E1(x), E2(x), \dots , Ek(x)}
\end{equation}
where
\begin{equation}
E_i = 0.5 x^T H_i x + x^T f_i
\end{equation}

and $E_i$ is deemed "more important" than $E_{i+1}$ (lexicographical ordering).

## Formulation

\begin{equation}
C_0 = H_0
\end{equation}
\begin{equation}
d_0 = f_0
\end{equation}

\begin{equation}
C_{i+1} =
\begin{bmatrix}
{H_{i+1}} & C_i^T \\  
C_i       & 0
\end{bmatrix}
\end{equation}
\begin{equation}
d_i =
\begin{bmatrix}
f_{i+1} \\  
d_i
\end{bmatrix}
\end{equation}

Where $H_{i+1}$ is padded by zeros to match the size of $C_i$, and
$f_{i+1}$ is padded by zeros to match the size of $d_i$.

When there are no more Energies or $C_i$ is full-rank the equation
$C\vec{z} = \vec{d}$ is solved. The final solution is the first n elements of
$\vec{z}$.

\begin{equation}
\min_{x,\lambda_1,\lambda_2, ..., \lambda_{i-1}} {1 \over 2}x^T H_i x + x^T f_i +
0\cdot\lambda_1 + 0\cdot\lambda_2 + ... + 0\cdot\lambda_{i-1}
\end{equation}

such that $C_{i-1} \cdot [x^T \lambda_1 \lambda_2 ... \lambda_{i-1}]^T = D_{i-1}$
or

\begin{equation}
y = [x^T \lambda_1 \lambda_2 ... \lambda_{i-1}]^T
\end{equation}

\begin{equation}
\min_y {1 \over 2} y^T A y - y^T B
\end{equation}

such that $C_{i-1} \cdot [x^T \lambda_1 \lambda_2 ... \lambda_{i-1}]^T = D_{i-1}$.

## Example

For example, on the third iteration we have the following:

\begin{equation}
C_2 =
\begin{bmatrix}
H_2   & 0     & H_1 & H_0^T \\  
0     & 0     & H_0   & 0   \\
H_1^T & H_0^T & 0     & 0    \\  
H_0   & 0     & 0     & 0    \\
\end{bmatrix}
\end{equation}
where each $H_i$ and $0$ are $n \times n$ matrices
\begin{equation}
d_2 =
\begin{bmatrix}
{f_{2}} \\
0       \\  
{f_{1}} \\
{f_{0}} \\
\end{bmatrix}
\end{equation}
where each $f_i$ and $0$ are $n \times 1$ vectors.

On the fourth iteration we have the following

\begin{equation}
C_3 =
\begin{bmatrix}
H_3   & 0     & 0     & 0     & H_2^T & 0     & H_1   & H_0 \\
0     & 0     & 0     & 0     & 0     & 0     & H_0   & 0   \\
0     & 0     & 0     & 0     & H_1^T & H_0^T & 0     & 0   \\
0     & 0     & 0     & 0     & H_0   & 0     & 0     & 0   \\
H_2   & 0     & H_1   & H_0^T &0      &0      &0      &0\\
0     & 0     & H_0   & 0     &0      &0      &0      &0\\
H_1^T & H_0^T & 0     & 0     &0      &0      &0      &0\\
H_0   & 0     & 0     & 0     &0      &0      &0      &0\\
\end{bmatrix}
\end{equation}
\begin{equation}
d_3 =
\begin{bmatrix}
{f_{3}} \\
0       \\
0 \\
0 \\
{f_{2}} \\
0       \\
{f_{1}} \\
{f_{0}} \\
\end{bmatrix}
\end{equation}

As can been seen the $C$ matrix doubles in size each iteration. The size of
$C_i$ is $(n2^i) \times (n2^i)$.

## Limitations

QR factorization on the i<sup>th</sup> set of constraints will be very
expensive because the number of non-zeros will be $O(n2^i)$.

The eventual solve will not necessarily behave well because the constraints
are not full rank.
