# L-Axis Reflectivity Diagnostics

Material: Bi2Se3
Stack: air / Bi2Se3(100 nm) / SiO2
Bandwidth mode: resimulate
Bandwidth FWHM: 0.05
Wavelength samples: 241
Lambda0: 1.5418 A
Born-scaling Qz floor: 1e-08

The plotted HT-derived comparisons are:

- Born-scaled HT p=0: (Qc/2Qz)^4 * I_HT0/I_HT0(0)
- Fresnel-corrected HT p=0: R_F(Qz) * I_HT0/I_HT0(0)
- Parratt-envelope HT p=0: R_Parratt(Qz) * I_HT0/I_HT0(0)

Parratt-envelope HT p=0 is an optical-envelope comparison for the finite stack.
It does not prove total external reflection; it only shows whether a feature lies
on the surface-optical scale.

The Miceli-style replacement is:

(Qc/2Qz)^4 -> R_F(Qz)

or, for a finite stack:

(Qc/2Qz)^4 -> R_Parratt(Qz).

At large Qz/Qc, the exact Fresnel reflectivity approaches the Born asymptote.
Near Qc, the Born asymptote underestimates the bounded optical reflectivity.
