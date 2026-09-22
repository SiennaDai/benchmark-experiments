# Exact two-mode polar response

Take (Sigma=\operatorname{diag}(\sigma_i,\sigma_j)), (sigma_i,sigma_j>0), and (E=\begin{pmatrix}0&e\\-e&0\end{pmatrix}). Then

\[
A(t)=\Sigma+tE=\begin{pmatrix}\sigma_i&te\\-te&\sigma_j\end{pmatrix},\qquad \det A(t)=\sigma_i\sigma_j+t^2e^2>0.
\]

Its polar factor is

\[
Q(t)=A(A^TA)^{-1/2}=\frac1{\sqrt{(\sigma_i+\sigma_j)^2+4t^2e^2}}
\begin{pmatrix}\sigma_i+\sigma_j&2te\\-2te&\sigma_i+\sigma_j\end{pmatrix}.
\]

Writing (Q=\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix}) gives (\tan\theta=2te/(\sigma_i+\sigma_j)). With (s=2|e|/(\sigma_i+\sigma_j)), the two-dimensional Frobenius cosine to (Q(0)=I) is (cos\theta), hence

\[
d(t)=1-\cos\theta=1-\frac1{\sqrt{1+t^2s^2}},\qquad \partial_s d=\frac{t^2s}{(1+t^2s^2)^{3/2}}>0
\]

for (s,t>0). The ordering in (s) is exactly monotone at every fixed positive (t), although the true angle is (\arctan(ts)), not its unbounded linear approximation (ts).

For diagonal perturbations and symmetric off-diagonal perturbations, the matrix remains symmetric positive definite over the reported smooth interval; its polar factor remains (I), so distortion is exactly zero. At a zero eigenvalue the polar map is not differentiable; after a sign change the factor changes branch. Those crossing points are recorded separately and excluded from smooth-branch monotonicity claims.
