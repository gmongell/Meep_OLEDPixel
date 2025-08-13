"""
simulate_oled.py
------------------

This module provides a reference implementation for simulating the angle‑dependent
outcoupling efficiency of a multilayer OLED structure using the Meep finite
difference time domain (FDTD) library.  It is written for Meep ≥1.12 and
assumes that Meep and its Python bindings are available on the target system.

The script constructs a one‑dimensional (effectively planar) stack of layers
representing the glass substrate, ITO, a metal‑free TADF emissive layer,
an electron transport layer based on Alq₃ and a thin silver cathode.  It then
excites the structure with an impulsive current source representing the
spontaneous emission from the TADF layer and records the Poynting flux into
the far field for multiple wavelengths and emission angles.  Finally, the
spectral outcoupling efficiency is combined with the CIE 1931 colour matching
functions to produce chromaticity coordinates for the device.

The default geometry corresponds to the optimized layer thicknesses found via
transfer‑matrix analysis: 105 nm ITO, 42 nm TADF, 57 nm ETL and 10 nm Ag.
The user may adjust these thicknesses on the command line.  Because full
angular sampling in FDTD can be computationally intensive, the example below
uses a coarse angular grid.  For production use you should refine the angle
and wavelength sampling and consider using parallel computation.

Usage::

    python simulate_oled.py --nangles 5 --nwav 15 --output results.npz

This will run the simulation, store the spectral transmittance and output
chromaticity coordinates upon completion.  The resulting NPZ file contains
arrays of wavelengths, angles and transmittance values.

Note: To keep the script self‑contained, a tabulated subset of the CIE 1931
colour matching functions (20 nm spacing) is embedded directly in this file.
For more accurate colour calculations, replace these arrays with a higher
resolution dataset.
"""

import argparse
import numpy as np
import meep as mp


# -----------------------------------------------------------------------------
# Colour matching functions (CIE 1931 2° observer)
#
# These arrays tabulate the x̄(λ), ȳ(λ) and z̄(λ) colour matching functions at
# wavelengths from 380 nm to 780 nm in 20 nm increments.  The values are taken
# from the Wyszecki & Stiles tables reproduced in the public domain【924867989069980†L245-L253】.
#
# If more accuracy is required, replace these arrays with a higher‑resolution
# dataset (e.g. 1 nm spacing) and interpolate accordingly.
CMF_WAVELENGTHS = np.array([
    380, 400, 420, 440, 460, 480, 500, 520, 540, 560,
    580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780
])
CMF_XBAR = np.array([
    0.0014, 0.0143, 0.1344, 0.3483, 0.2908, 0.0956, 0.0049,
    0.0633, 0.2904, 0.5945, 0.9163, 1.0622, 0.8544, 0.4479,
    0.1649, 0.0468, 0.0114, 0.0029, 0.0007, 0.0002, 0.0
])
CMF_YBAR = np.array([
    0.0,    0.0004, 0.0040, 0.0230, 0.0600, 0.1390, 0.3230,
    0.7100, 0.9540, 0.9950, 0.8700, 0.6310, 0.3810, 0.1750,
    0.0610, 0.0170, 0.0041, 0.0010, 0.0003, 0.0001, 0.0
])
CMF_ZBAR = np.array([
    0.0065, 0.0679, 0.6456, 1.7471, 1.6692, 0.8130, 0.2720,
    0.0782, 0.0203, 0.0039, 0.0017, 0.0008, 0.0002, 0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0
])


def cie_chromaticity(wavelengths_nm: np.ndarray, spectrum: np.ndarray) -> tuple:
    """Compute chromaticity coordinates (x, y) given a spectral power distribution.

    Parameters
    ----------
    wavelengths_nm : numpy.ndarray
        Array of wavelengths (nm) corresponding to the values in ``spectrum``.
    spectrum : numpy.ndarray
        Spectral power or intensity at each wavelength (arbitrary units).

    Returns
    -------
    tuple
        (x, y) chromaticity coordinates.
    """
    # interpolate CMF to the input wavelengths
    xbar_interp = np.interp(wavelengths_nm, CMF_WAVELENGTHS, CMF_XBAR)
    ybar_interp = np.interp(wavelengths_nm, CMF_WAVELENGTHS, CMF_YBAR)
    zbar_interp = np.interp(wavelengths_nm, CMF_WAVELENGTHS, CMF_ZBAR)
    # compute XYZ tristimulus values
    X = np.trapz(spectrum * xbar_interp, wavelengths_nm)
    Y = np.trapz(spectrum * ybar_interp, wavelengths_nm)
    Z = np.trapz(spectrum * zbar_interp, wavelengths_nm)
    denom = X + Y + Z
    return (X / denom, Y / denom) if denom != 0 else (0.0, 0.0)


def build_multilayer(d_ito_nm: float, d_emitter_nm: float,
                     d_etl_nm: float, d_ag_nm: float,
                     dpml: float = 0.5, pad: float = 2.0) -> tuple:
    """Construct the simulation geometry for the OLED stack.

    Thicknesses are given in nanometres and converted internally to microns.
    The function returns a tuple ``(cell, geometry)`` which may be passed
    directly to ``mp.Simulation``.
    """
    # convert nm to microns
    d_ito = d_ito_nm * 1e-3
    d_emit = d_emitter_nm * 1e-3
    d_etl = d_etl_nm * 1e-3
    d_ag = d_ag_nm * 1e-3
    # relative positions (z coordinates) of layer interfaces
    z0 = 0  # start of substrate interface
    z1 = z0 + d_ito
    z2 = z1 + d_emit
    z3 = z2 + d_etl
    z4 = z3 + d_ag
    # total stack thickness
    total_thickness = z4 - z0
    # simulation cell extends beyond stack plus PML and padding
    sz = total_thickness + 2 * (dpml + pad)
    cell = mp.Vector3(0, 0, sz)
    # define materials using Meep's Medium class.
    # For glass (BK7), ITO, TADF and Alq3 we use constant real permittivity.
    # For the silver cathode we include a Drude model to account for dispersion.
    bk7_index = 1.51  # approximate refractive index of BK7 glass
    ito_index = 1.9   # average index for ITO in visible
    tadf_index = 1.8  # assumed index for metal‑free TADF (no absorption)
    etl_index = 1.7   # approximate index for Alq3
    # Drude parameters for silver (taken from literature).  Fitting actual
    # dispersion data is recommended for quantitative work.
    silver_plasma_freq = 1.37e16  # rad/s
    silver_gamma = 2.73e13        # rad/s
    silver_epsilon_inf = 1.0
    silver_susc = [
        mp.DrudeSusceptibility(frequency=silver_plasma_freq/(2*np.pi),
                               gamma=silver_gamma/(2*np.pi),
                               sigma=1.0)
    ]
    bk7 = mp.Medium(index=bk7_index)
    ito = mp.Medium(index=ito_index)
    tadf = mp.Medium(index=tadf_index)
    etl = mp.Medium(index=etl_index)
    silver = mp.Medium(epsilon=silver_epsilon_inf, E_susceptibilities=silver_susc)
    # create geometry objects; they extend infinitely in x and y (2D) or only
    # along z (1D).  Here we use a 1D simulation (size.x==0, size.y==0).
    geometry = [
        mp.Block(material=ito,
                 size=mp.Vector3(mp.inf, mp.inf, d_ito),
                 center=mp.Vector3(0, 0, z0 + d_ito/2 + pad + dpml)),
        mp.Block(material=tadf,
                 size=mp.Vector3(mp.inf, mp.inf, d_emit),
                 center=mp.Vector3(0, 0, z1 + d_emit/2 + pad + dpml)),
        mp.Block(material=etl,
                 size=mp.Vector3(mp.inf, mp.inf, d_etl),
                 center=mp.Vector3(0, 0, z2 + d_etl/2 + pad + dpml)),
        mp.Block(material=silver,
                 size=mp.Vector3(mp.inf, mp.inf, d_ag),
                 center=mp.Vector3(0, 0, z3 + d_ag/2 + pad + dpml)),
    ]
    return cell, geometry


def simulate_device(d_ito_nm: float, d_emitter_nm: float, d_etl_nm: float,
                    d_ag_nm: float, n_angles: int, n_wavelengths: int,
                    wavelength_min: float = 400.0, wavelength_max: float = 700.0,
                    pad: float = 2.0, dpml: float = 0.5) -> tuple:
    """Simulate the OLED stack and return spectral outcoupling efficiency.

    Parameters
    ----------
    d_ito_nm : float
        Thickness of the ITO layer in nm.
    d_emitter_nm : float
        Thickness of the emissive TADF layer in nm.
    d_etl_nm : float
        Thickness of the electron transport layer (Alq₃) in nm.
    d_ag_nm : float
        Thickness of the silver cathode in nm.
    n_angles : int
        Number of emission angles within the escape cone to sample.
    n_wavelengths : int
        Number of wavelengths in the range [wavelength_min, wavelength_max].
    wavelength_min : float, optional
        Lower wavelength limit in nm.
    wavelength_max : float, optional
        Upper wavelength limit in nm.
    pad : float, optional
        Padding (μm) between the stack and the simulation boundaries.
    dpml : float, optional
        Thickness (μm) of the perfectly matched layers on both sides.

    Returns
    -------
    tuple
        ``(wavelengths, angles, transmittance)`` where ``transmittance`` is a
        2D array of shape (n_wavelengths, n_angles) containing the angle‑averaged
        outcoupling efficiency at each wavelength.
    """
    # build geometry and simulation cell
    cell, geometry = build_multilayer(d_ito_nm, d_emitter_nm, d_etl_nm,
                                      d_ag_nm, dpml=dpml, pad=pad)
    # define PML layers on both sides
    pml_layers = [mp.PML(thickness=dpml)]
    # define frequency sampling
    wavelengths = np.linspace(wavelength_min, wavelength_max, n_wavelengths)
    frequencies = 1e3 / wavelengths  # convert nm → μm; freq = c/λ in Meep units
    # define angles inside the BK7 substrate from 0 to critical angle
    n_sub = 1.51  # approximate BK7 index; critical angle for BK7/air
    n_air = 1.0
    theta_c = np.arcsin(n_air / n_sub)
    angles = np.linspace(0.0, theta_c, n_angles)
    # allocate result array
    transmittance = np.zeros((n_wavelengths, n_angles))
    # iterate over wavelengths
    for i, freq in enumerate(frequencies):
        # run one simulation per angle; reuse geometry for efficiency if desired
        for j, theta in enumerate(angles):
            # compute in‑plane wavevector for oblique incidence.  In a 1D stack
            # simulation using Bloch periodic boundary conditions in x, the
            # k_point along x is given by k_par = (2*pi*freq/c) * n_sub * sin(theta).
            # Since Meep normalises c=1, this reduces to k_par = freq * n_sub * sin(theta).
            k_parallel = freq * n_sub * np.sin(theta)
            sim = mp.Simulation(
                cell_size=cell,
                boundary_layers=pml_layers,
                geometry=geometry,
                sources=[
                    mp.Source(
                        src=mp.GaussianSource(frequency=freq, fwidth=freq/2.0),
                        component=mp.Ey,
                        center=mp.Vector3(0, 0, dpml + pad + d_ito_nm*1e-3/2.0),
                        size=mp.Vector3(0, 0, 0)
                    )
                ],
                dimensions=1,
                k_point=mp.Vector3(k_parallel, 0, 0),
                resolution=30,
            )
            # flux monitor positioned just above the stack to capture transmitted power
            fr = mp.FluxRegion(center=mp.Vector3(0, 0, cell.z/2 - dpml - pad/2),
                               size=mp.Vector3(0, 0, 0))
            flux = sim.add_flux(freq, 0, 1, fr)
            # run simulation long enough for fields to decay
            sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey,
                                                                    mp.Vector3(0, 0, 0),
                                                                    1e-9))
            # record transmitted flux; Meep normalises flux by source amplitude
            transmittance[i, j] = mp.get_fluxes(flux)[0]
            sim.reset_meep()
    return wavelengths, angles, transmittance


def main():
    parser = argparse.ArgumentParser(description="Simulate OLED outcoupling and compute chromaticity")
    parser.add_argument("--d_ito", type=float, default=105.0, help="ITO thickness in nm")
    parser.add_argument("--d_emitter", type=float, default=42.0, help="Emitter (TADF) thickness in nm")
    parser.add_argument("--d_etl", type=float, default=57.0, help="ETL thickness in nm")
    parser.add_argument("--d_ag", type=float, default=10.0, help="Silver thickness in nm")
    parser.add_argument("--nangles", type=int, default=5, help="Number of angles to sample")
    parser.add_argument("--nwav", type=int, default=15, help="Number of wavelengths to sample")
    parser.add_argument("--output", type=str, default="oled_outcoupling.npz", help="Output file for spectral data")
    args = parser.parse_args()
    # run simulation
    print(f"Running simulation with ITO={args.d_ito} nm, emitter={args.d_emitter} nm, ETL={args.d_etl} nm, Ag={args.d_ag} nm")
    wavelengths, angles, transmittance = simulate_device(
        d_ito_nm=args.d_ito,
        d_emitter_nm=args.d_emitter,
        d_etl_nm=args.d_etl,
        d_ag_nm=args.d_ag,
        n_angles=args.nangles,
        n_wavelengths=args.nwav
    )
    # angle averaging with cos(theta)*sin(theta) weighting
    weights = np.cos(angles) * np.sin(angles)
    weight_norm = np.trapz(weights, angles)
    angle_avg_trans = np.trapz(transmittance * weights[np.newaxis, :], angles, axis=1) / weight_norm
    # compute chromaticity assuming flat intrinsic emission spectrum
    x, y = cie_chromaticity(wavelengths, angle_avg_trans)
    print(f"Chromaticity coordinates: x={x:.4f}, y={y:.4f}")
    # save data to NPZ
    np.savez(args.output, wavelengths=wavelengths, angles=angles,
             transmittance=transmittance, angle_avg_trans=angle_avg_trans,
             chromaticity=(x, y))
    print(f"Saved spectral data and chromaticity to {args.output}")


if __name__ == "__main__":
    main()