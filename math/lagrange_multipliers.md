# Lagrange Multiplier Method

Solve the multi-objective minimization problem:

\begin{equation}
min_x  {E1(x), E2(x), \dots , Ek(x)}
\end{equation}
where
\begin{equation}
E_i = 0.5 * x.T * H_i * x + x.T * f_i
\end{equation}

and $E_i$ is deemed "more important" than $E_{i+1}$ (lexicographical ordering).

## Algorithm

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
C_i         & 0
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

## Example

For example, on the third iteration we have the following:

\begin{equation}
C_2 =
\begin{bmatrix}
H_2   & 0     & H_1^T & H_0^T \\  
0     & 0     & H_0   & 0   \\
H_1^T & H_0^T & 0     & 0    \\  
H_0   & 0     & 0     & 0    \\
\end{bmatrix}
\end{equation}
\begin{equation}
C_2 =
\begin{bmatrix}
{f_{2}} \\
0       \\  
{f_{1}} \\
{f_{0}} \\
\end{bmatrix}
\end{equation}

On the fourth iteration we have the following

\begin{equation}
C_2 =
\begin{bmatrix}
H_3   & 0     & 0     & 0     & H_2^T & 0     & H_1   & H_0^T \\  
0     & 0     & 0     & 0     & 0     & 0     & H_0   & 0   \\
0     & 0     & 0     & 0     & H_1   & H_0^T & 0     & 0    \\   
0     & 0     & 0     & 0     & H_0   & 0     & 0     & 0    \\
H_2   & 0     & H_1^T & H_0^T &0      &0      &0      &0\\  
0     & 0     & H_0   & 0     &0      &0      &0      &0\\
H_1^T & H_0^T & 0     & 0     &0      &0      &0      &0\\  
H_0   & 0     & 0     & 0     &0      &0      &0      &0\\
\end{bmatrix}
\end{equation}
\begin{equation}
C_3 =
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
