# Local cosine expansion

Let (x=O_0=Phi(M)), (r=|x|_F>0), and (u=x/r). Write the perturbed output as a radial and orthogonal part:

\[
O(t)=\alpha(t)u+q(t),\qquad q(t)=P_x^\perp O(t).
\]

If (Phi) is twice differentiable near (M),

\[
O(t)=x+t v+\tfrac12t^2a+O(t^3),\quad v=D\Phi_M[E].
\]

Consequently (alpha(t)=r+t\langle u,v\rangle+O(t^2)) and (q(t)=t v_\perp+\tfrac12t^2a_\perp+O(t^3)), where (v_\perp=P_x^\perp v). For positive (alpha),

\[
\cos(x,O(t))=\frac{\alpha(t)}{\sqrt{\alpha(t)^2+\|q(t)\|_F^2}}
=\left(1+\frac{\|q(t)\|_F^2}{\alpha(t)^2}\right)^{-1/2}.
\]

Since (|q(t)|^2=t^2|v_\perp|^2+t^3\langle v_\perp,a_\perp\rangle+O(t^4)), expanding ((1+z)^{-1/2}=1-z/2+O(z^2)) gives

\[
1-\cos(x,O(t))=\frac{t^2}{2r^2}\|P_x^\perp D\Phi_M[E]\|_F^2+O(t^3).
\]

The second derivative (a) cannot enter the quadratic coefficient; it first enters the cubic term through its component orthogonal to (x). A radial second-order correction changes output norm but not the leading angle. If (v_\perp=0), the quadratic coefficient vanishes and the first nonzero angular term can be fourth order. This is a local expansion only; it says nothing by itself about monotone behavior for finite (t).

For exact polar, (Q(M)^TQ(M)=I) on the active square factor, and its derivative is tangent to the orthogonal/Stiefel manifold. Thus \(\langle Q,DQ[E]\rangle_F=0\) and (v_\perp=v); with rectangular partial isometries the same identity holds on the active polar factor when rank is locally constant.
