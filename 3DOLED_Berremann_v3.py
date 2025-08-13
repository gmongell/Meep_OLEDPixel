#!/usr/bin/env python3
"""
Minimal MEEP script demonstrating:
 - A metal layer with complex refractive index
 - An anisotropic organic layer using epsilon_diag
 - ITO and substrate layers, all with real thickness
 - A dipole source in the organic layer
 - A flux monitor in air above the device

Run via: python meep_oled_aniso_corrected.py
"""

import meep as mp

##############################################################################
# SIMULATION PARAMETERS
##############################################################################
resolution = 50  # grid points per micron (example)
dpml = 0.5       # thickness of the PML boundary layer
cell_height = 4.0  # total height of the simulation region (in microns)

# 2D simulation: infinite in x, extent in y
cell_size = mp.Vector3(0, cell_height, 0)

# PML boundaries at top and bottom along y
pml_layers = [
    mp.PML(thickness=dpml, direction=mp.Y, side=mp.High),
    mp.PML(thickness=dpml, direction=mp.Y, side=mp.Low)
]

##############################################################################
# MATERIAL DEFINITIONS
##############################################################################
# Example metal: approximate silver-like values at ~500 nm
metal_n, metal_k = 0.14, 4.29
metal_epsilon_complex = (metal_n + 1j * metal_k)**2  # complex value

# A "metal" Medium with complex permittivity
metal_material = mp.Medium(
    epsilon_diag=mp.Vector3(
        metal_epsilon_complex,
        metal_epsilon_complex,
        metal_epsilon_complex
    )
)

# Example anisotropic organic layer
# Suppose eps_x=2.89, eps_y=2.25, eps_z=2.89 => n_x=1.7, n_y=1.5, n_z=1.7
# No imaginary part => purely real, diagonal anisotropy
anisotropic_organic = mp.Medium(
    epsilon_diag=mp.Vector3(2.89, 2.25, 2.89)
)

# ITO (isotropic) with n=1.9 => eps=3.61
ito_material = mp.Medium(epsilon=3.61)

# Substrate (e.g., glass) with n=1.5 => eps=2.25
substrate_material = mp.Medium(epsilon=2.25)

##############################################################################
# LAYER THICKNESSES (all REAL, not complex!)
##############################################################################
metal_thickness = 0.03      # 30 nm
organic_thickness = 0.1     # 100 nm
ito_thickness = 1        # 50 nm
substrate_thickness = 0.5   # 500 nm

##############################################################################
# BUILD THE GEOMETRY STACK (top to bottom in y)
##############################################################################
# Let y=0 be the top of the metal, going negative downward.
y_top = 0.0

# Metal layer
metal_layer = mp.Block(
    material=metal_material,
    size=mp.Vector3(mp.inf, metal_thickness, mp.inf),
    center=mp.Vector3(0, y_top - metal_thickness / 2, 0)
)
y_top -= metal_thickness

# Organic (anisotropic) layer
organic_layer = mp.Block(
    material=anisotropic_organic,
    size=mp.Vector3(mp.inf, organic_thickness, mp.inf),
    center=mp.Vector3(0, y_top - organic_thickness / 2, 0)
)
y_top -= organic_thickness

# ITO layer
ito_layer = mp.Block(
    material=ito_material,
    size=mp.Vector3(mp.inf, ito_thickness, mp.inf),
    center=mp.Vector3(0, y_top - ito_thickness / 2, 0)
)
y_top -= ito_thickness

# Substrate layer
substrate_layer = mp.Block(
    material=substrate_material,
    size=mp.Vector3(mp.inf, substrate_thickness, mp.inf),
    center=mp.Vector3(0, y_top - substrate_thickness / 2, 0)
)
y_top -= substrate_thickness

geometry = [
    metal_layer,
    organic_layer,
    ito_layer,
    substrate_layer
]

##############################################################################
# SOURCE DEFINITION: A SIMPLE DIPOLE IN THE ORGANIC LAYER
##############################################################################
# The middle of the organic layer is at y_center_organic:
metal_top = 0.0
organic_top = metal_top - metal_thickness
y_center_organic = organic_top - (organic_thickness / 2)

# Choose a single frequency (for example, lambda ~ 0.5um => freq=2.0)
freq = 2.0

sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=freq),
        component=mp.Ez,  # or Ex/Ey for TE/TM in 2D
        center=mp.Vector3(0, y_center_organic, 0)
    )
]

##############################################################################
# FLUX MONITOR ABOVE THE DEVICE
##############################################################################
# Place it at y=+0.1 um (above the metal top at y=0)
flux_monitor_y = 0.2
trans_region = mp.FluxRegion(
    center=mp.Vector3(0, flux_monitor_y, 0),
    size=mp.Vector3(mp.inf, 0, mp.inf)
)

##############################################################################
# CREATE AND RUN THE SIMULATION
##############################################################################
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    boundary_layers=pml_layers,
    sources=sources,
    resolution=resolution
)

trans_flux_obj = sim.add_flux(freq, 0, 1, trans_region)

# Run until fields settle or sufficiently decay
sim.run(until=200)

# Get transmitted flux
trans_flux_val = mp.get_fluxes(trans_flux_obj)[0]
print(f"Transmitted flux above the device: {trans_flux_val:.5g} (arbitrary units)")

##############################################################################
# NOTES ON OUTCOUPLING EFFICIENCY
##############################################################################
# In a real OLED outcoupling scenario:
# 1. You would measure total power the dipole emits (maybe from a reference simulation 
#    in free space or with full sphere flux monitors).
# 2. You might run multiple frequencies or a broadband source.
# 3. Real device layers often have angle-dependent emission => 3D or multiple angle analysis.
# 4. This code simply shows how to define layers, anisotropy, complex metal,
#    and avoid the 'TypeError' about complex thicknesses.
