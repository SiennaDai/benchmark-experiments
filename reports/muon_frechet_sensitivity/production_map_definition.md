# Production Muon map definition

The canonical map is `src/optim/muon_reference.py::zeropower_newton_schulz`. It casts the 2-D input to FP32; if rows exceed columns, it transposes before iteration; it divides by `||M||_F + eps`; then it performs K=5 steps with `(a,b,c)=(3.4445,-4.7750,2.0315)`:

`X <- a X + b (X X^T) X + c (X X^T)^2 X`.

It transposes back for tall inputs. There is no post-scaling. Under an SVD this is `U q_5(Sigma/(||M||_F+eps)) V^T`. Thus the polynomial part is a spectral map, but its normalization is state-dependent. The production Fréchet derivative below includes `d(||M||_F)`; the frozen-normalization diagnostic omits that term and is reported separately. Tall/wide orientation is transpose-equivariant; the rectangular leakage term is taken from the left nullspace for tall matrices and right nullspace for wide matrices.

For normalized singular values `x_i`, scalar values `f_i=q_5(x_i)` and derivatives `f'_i=q'_5(x_i)`, the fixed-scale core uses diagonal `f'_i`, symmetric divided difference `(f_i-f_j)/(x_i-x_j)`, skew factor `(f_i+f_j)/(x_i+x_j)`, and leakage `f_i/x_i`. Production additionally has `dX=E/s - M <M,E>/(||M||_F s^2)`, `s=||M||_F+eps`, so the global radial normalization derivative contributes to the diagonal coordinate perturbation. At equal singular values the divided difference uses the average derivative limit. Exact polar uses `f=1`, zero diagonal and symmetric channels, skew factor `2/(sigma_i+sigma_j)`, and rectangular coefficient `1/sigma_i`.
