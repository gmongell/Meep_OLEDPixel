#!/usr/bin/env python3

import meep as mp
import numpy as np

##############################################################################
# SIMULATION IN POSITIVE OCTANT (x>0, y>0, z>0)
##############################################################################

# 1) Define overall domain size:
#    Let's say we want a 1-micron device region plus 0.2 microns of PML in x, y, z.
dpml = 0.2
device_extent = 1.0  # the device occupies [0, 1] in x, y, z
cell_size = mp.Vector3(device_extent + dpml,  # x from 0 to 1+0.2
                       device_extent + dpml,  # y from 0 to 1+0.2
                       device_extent + dpml)  # z from 0 to 1+0.2

# 2) PML only on "high" sides in x, y, z => i.e., at x=Lx, y=Ly, z=Lz
pml_layers = [
    mp.PML(dpml, direction=mp.X, side=mp.High),
    mp.PML(dpml, direction=mp.Y, side=mp.High),
    mp.PML(dpml, direction=mp.Z, side=mp.High)
]

resolution = 100  # grid points per micron (example)

##############################################################################
# MATERIALS (ILLUSTRATIVE)
##############################################################################
# Let's define one simple material region (e.g., dielectric with n=1.5 => eps=2.25)
dielectric = mp.Medium(epsilon=1.75)

# And let air be the default background (epsilon=1)
# If you have more complex layers (metal, organics, etc.), define them similarly.

##############################################################################
# GEOMETRY
##############################################################################
# Suppose we want a single block from (0,0,0) to (1,1,1). In MEEP, we specify
# center plus size. A block with size=(1,1,1) and center=(0.5,0.5,0.5)
# will fill exactly the positive unit cube from 0..1 in x,y,z.

block = mp.Block(
    material=dielectric,
    size=mp.Vector3(1.0, 1.0, 1.0),
    center=mp.Vector3(0.5, 0.5, 0.5)
)

geometry = [block]

##############################################################################
# SOURCE
##############################################################################
# Place, for example, a continuous-wave point source at (0.5, 0.5, 0.1)
# near the bottom of the block in the z-direction. 
# We'll have it emit in Ex polarization, at some test frequency f=1.0 (λ=1.0 µm).
fcen = 1.0  # frequency = 1/µm => wavelength=1.0 µm
sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=fcen),
        component=mp.Ex,
        center=mp.Vector3(0.5, 0.5, 0.1)
    )
]

##############################################################################
# FLUX MONITOR
##############################################################################
# As an example, let's measure flux on a plane near the "top" (z=0.9),
# well before the PML at z=1.2. We'll see how much power propagates upward.

flux_monitor_z = 0.9
flux_region = mp.FluxRegion(
    center=mp.Vector3(0.5, 0.5, flux_monitor_z),
    size=mp.Vector3(0.8, 0.8, 0)  # e.g. an 0.8x0.8 plane in x-y
)
nfreq = 100  # number of frequency points to record
# We'll measure flux at fcen ± 0.2
df = 0.4

##############################################################################
# CREATE SIMULATION
##############################################################################
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    boundary_layers=pml_layers,
    sources=sources,
    resolution=resolution
)

trans_flux_obj = sim.add_flux(fcen, df, nfreq, flux_region)

##############################################################################
# RUN
##############################################################################
# Run for some time so fields can propagate.
sim.run(until=200)

##############################################################################
# POST-PROCESSING
##############################################################################
# Retrieve flux vs. frequency
flux_spectrum = mp.get_fluxes(trans_flux_obj)  # array of length nfreq
frequencies   = np.linspace(fcen - 0.5*df, fcen + 0.5*df, nfreq)

# Print the data
print("freq, flux")
for f, flx in zip(frequencies, flux_spectrum):
    print(f"{f:.3f}, {flx:.6g}")

print("\nSimulation complete in the positive octant!")
