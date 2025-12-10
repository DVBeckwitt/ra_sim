# Center-of-rotation (CoR) axis math

The sample can rotate about an axis that is pitched away from the laboratory +x direction by the user-specified `cor_angle` (°).  The rotation is defined using Rodrigues' rotation formula so that the axis angle and the applied tilt remain decoupled and numerically stable.

## Axis definition

The CoR axis lies in the x–z plane.  When `cor_angle = 0`, the axis coincides with +x; positive angles pitch the axis toward +z.  The unit axis vector is

\[
\mathbf{a} = (\cos\varphi,\; 0,\; \sin\varphi),\quad\varphi = \mathrm{radians}(\text{cor\_angle}).
\]

The vector is normalized defensively in the implementation so that even tiny angles keep a well-defined unit axis.

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
