# MHD shell model
### About
A shell model for hydrodynamical (HD) and magnetohydrodynamical (MHD) turbulence. Based on the 'improved' shell model of [L'vov, V. S. et al. Phys. Rev. E 58, 1811 (1998)](https://doi.org/10.1103/PhysRevE.58.1811) extended to MHD.  Uses a fourth-order Runge-Kutta method for time-stepping. 

### Santiago J. Benavides 

(santiago.benavides@upm.es)

### Has two different modes
1. Full Navier Stokes.
2. Phase-only. See [Arguedas-Leiva et al. Phys. Rev. Research 4, L032035 (2022)](https://doi.org/10.1103/PhysRevResearch.4.L032035) for info on phase-only formulations.

Example directories include two full models (one HD and one MHD) and one phase-only HD model. The MHD directory creates multiple parallel runs (using MPI4Py), with varying magnetic energy injection rate. The two HD directories create multiple parallel runs, with varying control parameter $b$ (which controls what the second conserved quantity is). Note, MPI4Py is not needed to run the model itself.

### Capabilities: 
* Statistics of phases and triad phases.

### References
* [Biferale, L., Ann. Rev. Fluid Mech. Vol. 35:441-468 (2003)](https://doi.org/10.1146/annurev.fluid.35.101101.161122)
* [Plunian, F. et al., Phys. Rep. Volume 523, Issue 1 (2013)](https://doi.org/10.1016/j.physrep.2012.09.001)

[![DOI](https://zenodo.org/badge/1013186208.svg)](https://doi.org/10.5281/zenodo.15800546)
