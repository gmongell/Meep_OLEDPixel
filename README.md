# OLED optical simulations with Meep

Exploratory Python scripts for multilayer OLED optics, optical local density of states (LDOS), thickness sweeps, and spectral/color calculations. The stated goal is to investigate outcoupling and wavelength-dependent behavior; the models require physical validation before quantitative design use.

## Functions and engineering applications

| Source / interface | Observed function | Inferred application |
|---|---|---|
| `simulate_oled.py`: `build_multilayer(...)` | Builds layer geometry from nm thicknesses | Comparing electrode/emitter/transport-layer stacks |
| `simulate_oled.py`: `simulate_device(...)` | Returns wavelength grid, angle grid, and a two-dimensional flux-derived array | Screening thickness and spectral/angle trends [1] |
| `simulate_oled.py`: `cie_chromaticity(wavelengths_nm, spectrum)` | Integrates a spectrum against the included color-matching arrays | Exploratory color-shift comparisons |
| `extraction_eff_ldos.py`: `extraction_eff_cyl(dmat, h)`, `extraction_eff_3D(dmat, h)` | Cylindrical/3D extraction and LDOS examples | Comparing geometric representations and emission behavior [1] |
| `Multiwavelength_Multithickness_dispersiveLDOS_OLED.py`: `run_ldos_simulation(sweep_layer, new_thickness)` | Layer/thickness-dependent LDOS calculations | Locating optical-environment trends |
| `OLED_optimizer_tmm_meep*.py` | Combined transfer-matrix and Meep exploration | Candidate screening followed by electromagnetic checks [1,2] |
| `OLED_ldos_python*.py`, `DrivingVoltageCalculation.py` | Separate illustrative electronic DOS/energy-level and voltage calculations | Concept exploration; electronic DOS is distinct from optical LDOS |

## Requirements and example calls

Core scripts import Python, Meep, and NumPy; many also need Matplotlib. Optimizer variants additionally import pandas, `tmm`, and, depending on the file, opticalmaterialspy, requests, or PyYAML. Use an environment with the actual Meep electromagnetic Python bindings and verify it with:

```bash
python meep_install_verify.py
python simulate_oled.py --help
python simulate_oled.py --d_ito 105 --d_emitter 42 --d_etl 57 --d_ag 10 \
  --nangles 3 --nwav 5 --output oled_example.npz
```

This small-grid invocation matches the parser but was not run with Meep here. It is not a converged simulation. Thickness arguments are nm; returned wavelengths are nm and angles are radians. NPZ keys are `wavelengths`, `angles`, `transmittance`, `angle_avg_trans`, and `chromaticity`.

A direct function example, which imports Meep but does not itself launch an FDTD run:

```python
import numpy as np
from simulate_oled import cie_chromaticity
wl_nm = np.linspace(400.0, 700.0, 61)
spectrum = np.exp(-0.5 * ((wl_nm - 550.0) / 25.0)**2)
xy = cie_chromaticity(wl_nm, spectrum)
print(xy)
```

## Model limits and validation

The CLI assumes a flat intrinsic emission spectrum for its final color calculation. The included color-matching table and flux normalization require independent checks. A transmitted-flux quantity or plane-wave transmittance is not automatically a dipole-emission outcoupling fraction or external quantum efficiency. Verify source position, layer centering, absorbing-material dispersion, dimensionality/Bloch-angle treatment, and reference-power normalization. Use at least two angle samples for integration; increase resolution, run duration, PML spacing, and angular/spectral sampling to establish convergence. Thin metal layers may be poorly resolved by the default grid. Several scripts execute sweeps or plotting at import, and filename version suffixes do not establish correctness. Python syntax inspection is not Meep runtime or physics validation.

## Review scope and software citation

Documentation reviewed on 2026-10-08 against source commit [`2064a4052573`](https://github.com/gmongell/Meep_OLEDPixel/tree/2064a40525737590f4cce9a6480c07cf62c3e82f). “Observed” means supported by source inspection; engineering applications are reasoned possibilities unless explicitly demonstrated. Scholarly references provide methodological context and do not certify these implementations. Runtime validation is stated separately above.

For software attribution, cite Guy Francis Mongelli, *Meep_OLEDPixel*, the [repository](https://github.com/gmongell/Meep_OLEDPixel), the exact commit used, and your access date. Also cite the relevant method publications and any original third-party contributors. No unverified software DOI or release version is assigned by this documentation.

## Scholarly references

1. A. F. Oskooi et al. (2010). “Meep: A flexible free-software package for electromagnetic simulations by the FDTD method.” *Computer Physics Communications* 181, 687–702. [DOI: 10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008). Method/software reference for FDTD, dispersive media, sources, and boundary conditions.

2. S. J. Byrnes (2016; revised 2020). “Multilayer optical calculations.” *arXiv:1603.02720*, preprint. [DOI: 10.48550/arXiv.1603.02720](https://doi.org/10.48550/arXiv.1603.02720). Reference for planar transfer matrices, polarization, absorption, and power normalization; it is not a validation of the repository’s emitter models.

## Ownership and existing license notices

Copyright (c) 2025 Guy Francis Mongelli

The existing project notice declares Apache License 2.0 for project code. Documentation, prose, and figures are declared CC BY 4.0; notebook code cells are Apache-2.0 and narrative/figures CC BY 4.0. Preserve all file-level and third-party notices. This README update does not change ownership or licensing terms.


## Reproducibility, source verification, and contribution policy (2026-10-08)

- **Observed versus proposed:** The function inventory above describes inspected source where indicated. Engineering applications identified as *inferred* are potential uses, not verified features or validated performance claims.
- **Usage examples:** Treat the documented commands and calls as illustrative until the referenced source file, runtime version, dependencies, required input data, and working directory have been checked. Do not execute notebook fragments or batch-scheduler directives as standalone programs without adapting their context.
- **Scientific citations:** References above identify relevant governing methods and computational background; citing a publication does not imply that its algorithm is implemented in this repository or that the publication was authored by this repository owner.
- **Missing or null artifacts:** Empty, placeholder, missing, or non-executable source files must not be represented as functional implementations. Candidate restorations from personal archives require provenance, content comparison, license review, and explicit verification before committing code.
- **Access model:** This repository is publicly readable. Public visibility does not grant anonymous push rights; write access is controlled separately through repository collaborators, credentials, apps, deploy keys, and branch rules. This README is descriptive and does not itself enforce permissions.
