# Reviewer notes: internals synthesis

These public reference derivations support review; they are not a private or validated exam key.

For the rectangular map, the Jacobian is

\[
J(a,b)=\begin{bmatrix}b&a\\\cos a&2b\\2a&-1\end{bmatrix}.
\]

The Hessian of half the squared output norm is \(J^\mathsf{T}J+\sum_i f_iH_i\), where

\[
H_1=\begin{bmatrix}0&1\\1&0\end{bmatrix},\quad
H_2=\begin{bmatrix}-\sin a&0\\0&2\end{bmatrix},\quad
H_3=\begin{bmatrix}2&0\\0&0\end{bmatrix}.
\]

The input direction has two coordinates and output cotangent has three. The JVP has three coordinates, VJP two, and their paired dot products are scalars. Require an independent changed-point comparison, not only the adjoint identity, because paired wrong rules can satisfy that identity.

For the composed softplus loss, let \(t_i=w_ix_i+c_i\) and \(p_i=\sigma(t_i)\). The gradient is \(w_ip_i\) and the diagonal Hessian is \(w_i^2p_i(1-p_i)\). Off-diagonal entries are zero. A missing squared weight is an actual chain-rule error. Saturation affects slope differently at the negative and positive tails: softplus slope tends to zero at negative inputs and one at positive inputs; curvature tends to zero at both tails.

For the exponential extension, the local tangent rule is \(\dot y=e^z\dot z\). The whole-expression reference is

\[
DG(z)[v]=\sum_i\left(e^{z_i}+\sin z_i+z_i\cos z_i\right)v_i.
\]

Moderate test inputs avoid overflow without claiming the primitive remains finite for arbitrarily large arguments. Examine constant tangents, reduction axes, tuple output contracts, and actual lower/compile calls. A source search for an autodiff call alone is weak evidence; read the new rule and inspect changed-condition behavior.

The interpreter is intentionally incomplete. A clear rejection is correct outside its documented subset. StableHLO text and transformation structure may vary across JAX versions; numerical comparisons and semantic explanations should not depend on variable names or exact IR text length.
