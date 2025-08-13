#!/usr/bin/env python3

import meep as mp
import numpy as np

##############################################################################
# PARAMETERS
##############################################################################
dpml = 0.2               # thickness of PML
device_extent = 1.0      # the block occupies [0..1] in x, y, z
resolution = 100         # grid points per micron

# We extend the simulation cell on both the low and high sides by 'dpml'
# so the domain goes from -dpml to (device_extent + dpml) in each dimension:
cell_size = mp.Vector3(
    device_extent + 2*dpml,  # x
    device_extent + 2*dpml,  # y
    device_extent + 2*dpml   # z
)

# PML on all faces (low and high) in x, y, z
pml_layers = [mp.PML(dpml)]

##############################################################################
# MATERIALS & GEOMETRY
##############################################################################
# Example dielectric with n ~ sqrt(1.75) ~ 1.32
# (Adjust epsilon as needed; 1.75 is just an example)
dielectric = mp.Medium(epsilon=1.75)

# A block occupying [0..1] in x, y, z.
# Because our cell extends from -dpml..(1+dpml),
# the block center is still at (0.5, 0.5, 0.5).
# (Coordinates are measured from the lower-left-front corner at x=-dpml,y=-dpml,z=-dpml.)
block = mp.Block(
    material=dielectric,
    size=mp.Vector3(1.0, 1.0, 1.0),
    center=mp.Vector3(0.5, 0.5, 0.5)
)

geometry = [block]

##############################################################################
# SOURCE
##############################################################################
# Continuous-wave source at frequency f=1 (λ=1 µm).
# Place it near the bottom of the block (z=~0.1).
fcen = 1.0
sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=fcen),
        component=mp.Ex,
        center=mp.Vector3(0.5, 0.5, 0.1)  # just inside the block
    )
]

##############################################################################
# FLUX MONITOR
##############################################################################
# We'll measure the upward flux near z=1.05 (above the block top at z=1.0).
# The top PML starts around z=1.0 + dpml = 1.2, so this plane is safely away
# from the PML, allowing outgoing waves to be measured.
flux_monitor_z = 1.05
flux_size_xy = 0.8  # smaller than the full domain so it fits in the interior

flux_region = mp.FluxRegion(
    center=mp.Vector3(0.5, 0.5, flux_monitor_z),
    size=mp.Vector3(flux_size_xy, flux_size_xy, 0)  # plane parallel to x-y
)

nfreq = 100  # number of frequency points
df    = 0.4  # bandwidth around fcen (i.e. fcen ± 0.2)

##############################################################################
# CREATE SIMULATION
##############################################################################
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    boundary_layers=pml_layers,
    sources=sources,
    resolution=resolution,
    default_material=mp.Medium(epsilon=1.0)  # air background
)

trans_flux_obj = sim.add_flux(fcen, df, nfreq, flux_region)

##############################################################################
# RUN
##############################################################################
# Run enough time for fields to propagate out the top.
# With a CW source, fields never die out, so we typically pick a time
# after which the net flux has converged. You might check field energies, etc.
sim.run(until=200)

##############################################################################
# POST-PROCESSING
##############################################################################
flux_spectrum = mp.get_fluxes(trans_flux_obj)  # array of length nfreq
frequencies   = np.linspace(fcen - 0.5*df, fcen + 0.5*df, nfreq)

print("freq, flux")
for f, flx in zip(frequencies, flux_spectrum):
    print(f"{f:.3f}, {flx:.6g}")

print("\nSimulation complete with PML on all sides!")
