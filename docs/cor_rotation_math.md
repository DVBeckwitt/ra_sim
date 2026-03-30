# Center-of-rotation (CoR) axis math

RA-SIM applies the sample tilt `theta_initial` about a configurable
center-of-rotation axis rather than always about laboratory `+x`. The axis is
controlled by:

- `cor_angle`: pitch away from `+x` toward `+z`
- `psi_z`: yaw of that pitched axis about laboratory `z`

This note matches the exact sign convention used in
`ra_sim/simulation/diffraction.py`.

## Baseline sample rotation

Before the CoR tilt is applied, the code builds

\[
R_{ZY} = R_z(\psi)\,R_y(\chi),
\]

with

\[
R_y(\chi)=
\begin{bmatrix}
\cos\chi & 0 & \sin\chi \\
0 & 1 & 0 \\
-\sin\chi & 0 & \cos\chi
\end{bmatrix},
\qquad
R_z(\psi)=
\begin{bmatrix}
\cos\psi & \sin\psi & 0 \\
-\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}.
\]

With this `R_z` convention, positive `psi` rotates `+x` toward `-y`.

## CoR axis

The unyawed CoR axis lives in the `x-z` plane:

\[
\mathbf{a}_0 =
\begin{bmatrix}
\cos\varphi \\
0 \\
\sin\varphi
\end{bmatrix},
\qquad
\varphi = \mathrm{radians}(\text{cor\_angle}).
\]

Yawing that axis by `psi_z` gives

\[
\mathbf{a} = R_z(\psi_z)\,\mathbf{a}_0
=
\begin{bmatrix}
\cos\psi_z \cos\varphi \\
-\sin\psi_z \cos\varphi \\
\sin\varphi
\end{bmatrix}.
\]

So:

- `cor_angle = 0`, `psi_z = 0` gives `a = +x`
- positive `cor_angle` tips the axis toward `+z`
- positive `psi_z` rotates the `+x` direction toward `-y`

The implementation normalizes `a` before using it.

## Rodrigues rotation

Let

\[
\theta = \mathrm{radians}(\text{theta\_initial}).
\]

The CoR tilt is applied with Rodrigues' formula:

\[
R_{\mathrm{cor}}
= \cos\theta\,I
+ (1-\cos\theta)\,\mathbf{a}\mathbf{a}^\top
+ \sin\theta\,[\mathbf{a}]_\times,
\]

where

\[
[\mathbf{a}]_\times =
\begin{bmatrix}
0 & -a_z & a_y \\
a_z & 0 & -a_x \\
-a_y & a_x & 0
\end{bmatrix}.
\]

The final sample rotation used by the kernel is

\[
R_{\mathrm{sample}} = R_{\mathrm{cor}}\,R_{ZY}.
\]

The rotated sample normal is

\[
\mathbf{n}_{\mathrm{surf}} = R_{\mathrm{cor}}\,(R_{ZY}\,\hat{\mathbf{z}}).
\]

## Reference point

The sample reference point starts at

\[
P_0 = (0,\,0,\,-z_s).
\]

The code rotates it with `R_sample` and then explicitly sets `P0_rot[0] = 0`.
That keeps the reference point on the laboratory CoR plane used by the detector
intersection geometry.

## Where this appears

The same construction is used in:

- `_build_sample_rotation()` in `ra_sim/simulation/diffraction.py`
- the detector-path debug helper in the same module

That keeps the main simulation and the debug path on the same axis convention.
