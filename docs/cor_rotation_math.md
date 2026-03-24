# Center-of-rotation (CoR) axis math

The sample can rotate about an axis that is pitched away from the laboratory +x direction by the user-specified `cor_angle` (°) and yawed about laboratory `+z` by `psi_z` (°). The rotation is defined using Rodrigues' rotation formula so that the axis orientation and the applied tilt remain decoupled and numerically stable.

## Axis definition

The CoR axis is constructed by first pitching it inside its local x–z plane, then yawing that pitched axis about laboratory `+z`. When `cor_angle = 0` and `psi_z = 0`, the axis coincides with `+x`. Positive `cor_angle` pitches the axis toward `+z`. The pitched intermediate axis is

\[
\mathbf{a}_0 = (\cos\varphi,\; 0,\; \sin\varphi),\quad\varphi = \mathrm{radians}(\text{cor\_angle}).
\]

The final axis is

\[
\mathbf{a} = \mathbf{R}_z(\psi)\,\mathbf{a}_0,\quad \psi = \mathrm{radians}(\text{psi\_z}),
\]

using the same `R_z` convention as the sample-orientation matrices in the simulator. The vector is normalized defensively in the implementation so that even tiny angles keep a well-defined unit axis.

## Rotation matrix

The sample tilt `theta_initial` (°) is applied as a right-handed rotation of angle \(\theta\) about \(\mathbf{a}\) using Rodrigues' formula

\[
\mathbf{R}_{\text{CoR}} = \cos\theta\,\mathbf{I} + (1-\cos\theta)\, \mathbf{a}\,\mathbf{a}^\top + \sin\theta\, [\mathbf{a}]_\times,
\]

where

\[
[\mathbf{a}]_\times = \begin{bmatrix}
0 & -a_z & a_y\\
a_z & 0 & -a_x\\
-a_y & a_x & 0
\end{bmatrix}
\]

is the skew-symmetric cross-product matrix for \(\mathbf{a}\).  The resulting matrix rotates any lab-frame vector into the CoR-tilted frame.  We multiply this matrix into the existing \(R_z R_y\) orientation to obtain the overall sample rotation used during diffraction calculations.

## Usage in the code

Both the core diffraction kernel (`calculate_phi`) and the geometry debug helper (`trace_specular_debug`) build `R_cor` exactly from the expressions above, normalize the resulting surface normal, and rotate the incident-beam origin so that the simulation consistently respects the configurable CoR axis.
