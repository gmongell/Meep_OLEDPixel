#!/usr/bin/env python3

import meep as mp
import numpy as np
import math

##############################################################################
# SIMULATION PARAMETERS
##############################################################################
resolution = 60        # grid points per µm
dpml       = 0.5       # thickness of PML on all sides
sx         = 8.0       # total width (x dimension) of the 2D cell
sy         = 8.0       # total height (y dimension) of the 2D cell
cell_size  = mp.Vector3(sx, sy, 0)

pml_layers = [mp.PML(dpml)]  # apply PML on every boundary (x low/high, y low/high)

# Visible range: ~400–700 nm => frequencies ~1.43–2.5 µm^-1
lambda_min = 0.4
lambda_max = 0.7
fmin       = 1/lambda_max  # ~1.4286
fmax       = 1/lambda_min  # ~2.5
fcen       = 0.5*(fmin + fmax)
df         = fmax - fmin

##############################################################################
# MATERIAL PROPERTIES (SIMPLIFIED)
##############################################################################
# BK7 (n ~1.515 => eps ~2.30)
bk7 = mp.Medium(epsilon=2.3)

# ITO (real devices are dispersive; here, pick a constant as placeholder)
ito = mp.Medium(epsilon=3.8)

# Alq3 (organic emitter) (n ~1.7 => eps ~2.89); pick 2.9 as placeholder
alq3 = mp.Medium(epsilon=2.9)

# Ag (silver) - real silver is highly dispersive and lossy.
# For a quick approximation, you can use a large conductivity or Meep’s built-in:
#   from meep.materials import Ag
# but let's define a simple “fake silver” with large imaginary part for demonstration:
ag = mp.Medium(epsilon=1.0, D_conductivity=2000.0)

# Air is the default_material (epsilon=1.0).

##############################################################################
# LAYER THICKNESSES
##############################################################################
bk7_thick   = 0.5   # 0.5 µm
ito_thick   = 0.05  # 50 nm
alq3_thick  = 0.2   # 200 nm
ag_thick    = 0.05  # 50 nm
# Above Ag is air, extending to the top of the cell.

# For convenience, define the y coordinates of each layer boundary:
# Start from y=0 at the bottom of the BK7. The stack will extend upward.
bk7_bottom  = 0.0
bk7_top     = bk7_bottom + bk7_thick

ito_bottom  = bk7_top
ito_top     = ito_bottom + ito_thick

alq3_bottom = ito_top
alq3_top    = alq3_bottom + alq3_thick

ag_bottom   = alq3_top
ag_top      = ag_bottom + ag_thick
# Above ag_top is air until the top of the cell (sy - dpml).

##############################################################################
# GEOMETRY
##############################################################################
# We use mp.Block for each layer in a 2D plane (x,y). The Block is infinite (inf) in x
# but has thickness in y, with centers at the midpoint of each layer.

geometry = [
    # BK7
    mp.Block(
        material=bk7,
        size=mp.Vector3(mp.inf, bk7_thick),
        center=mp.Vector3(0, bk7_bottom + 0.5*bk7_thick)
    ),
    # ITO
    mp.Block(
        material=ito,
        size=mp.Vector3(mp.inf, ito_thick),
        center=mp.Vector3(0, ito_bottom + 0.5*ito_thick)
    ),
    # Alq3
    mp.Block(
        material=alq3,
        size=mp.Vector3(mp.inf, alq3_thick),
        center=mp.Vector3(0, alq3_bottom + 0.5*alq3_thick)
    ),
    # Ag
    mp.Block(
        material=ag,
        size=mp.Vector3(mp.inf, ag_thick),
        center=mp.Vector3(0, ag_bottom + 0.5*ag_thick)
    ),
    # Air above => no block needed (default_material = air).
]

##############################################################################
# SOURCE (DIPOLE IN ALQ3)
##############################################################################
# Place a broadband dipole near the middle of the Alq3 layer:
source_y = alq3_bottom + 0.5*alq3_thick  # center
sources = [
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,     # TE-like polarization in 2D
        center=mp.Vector3(0, source_y),
        amplitude=1.0
    )
]

##############################################################################
# CREATE SIMULATION
##############################################################################
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    default_material=mp.Medium(epsilon=1.0)  # air
)

##############################################################################
# NEAR-TO-FAR TRANSFORM
##############################################################################
# We'll capture fields in a line above the entire stack (above the Ag),
# then transform to the far field at various angles.

# Place near2far line above the Ag layer, say at y=ag_top + 0.1 = ~0.35
n2f_y = ag_top + 0.1
n2f_length = 6.0  # covers a region in x from -3..+3, for example

n2f_obj = sim.add_near2far(
    fcen, df, 301,  # center freq, bandwidth, # freq points
    mp.Near2FarRegion(
        center=mp.Vector3(0, n2f_y),
        size=mp.Vector3(n2f_length, 0),  # line along x
        direction=mp.Y  # normal direction is +y
    )
)

##############################################################################
# RUN SIMULATION
##############################################################################
# Run until fields substantially leave the domain or decay.
# For a broadband dipole, 300-500 time units is a typical guess. 
sim.run(until=300)

##############################################################################
# FAR-FIELD ANGULAR INTEGRATION
##############################################################################
# In 2D, we can sample the far field at various angles from 0..90
# (normal to grazing in the +y half-space).

n_angles = 91
thetas = np.linspace(0, 90, n_angles)

rad = 10.0  # radius at which to compute far fields (arbitrary > cell)
total_flux = 0.0

print("# theta_deg, S_radial (approx)")

for theta_deg in thetas:
    theta_rad = math.radians(theta_deg)
    # Convert so that 0° is along +y, increasing to 90° is along +x in the top half-plane:
    phi_rad = 0.5*math.pi - theta_rad
    rx = rad * math.cos(phi_rad)
    ry = rad * math.sin(phi_rad)

    Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz, Bx, By, Bz = sim.get_farfield(n2f_obj,mp.Vector3(rx, ry))

    # Rough radial Poynting calculation in 2D: S = 0.5 * Re(E x H*)
    Ex_c = np.conjugate(Ex)
    Ey_c = np.conjugate(Ey)
    Ez_c = np.conjugate(Ez)

    Hx_c = np.conjugate(Hx)
    Hy_c = np.conjugate(Hy)
    Hz_c = np.conjugate(Hz)

    # Compute cross product (E × H*) => S
    Sx = 0.5 * (Ey * Hz_c - Ez * Hy_c).real
    Sy = 0.5 * (Ez * Hx_c - Ex * Hz_c).real

    # Project onto the radial direction
    r_hat = np.array([math.cos(phi_rad), math.sin(phi_rad)])
    S_vec = np.array([Sx, Sy])
    S_radial = np.dot(S_vec, r_hat)

    print(f"{theta_deg:6.1f}, {S_radial:.6g}")
    total_flux += S_radial  # simple sum; for better accuracy, do an integral in radians

print(f"\nApprox. total flux (sum over angles) = {total_flux:.6g} (arb units)\n")
print("Finished 5-layer OLED stack simulation!")
