#!/root/miniconda3/envs/meep_env/bin/python

import meep as mp
import numpy as np

##############################################################################
# SIMULATION PARAMETERS
##############################################################################
resolution = 50       # grid points per µm (example)
dpml = 0.5            # thickness of PML
sx = 7.0              # total size in x-direction
sy = 7.0              # total size in y-direction
cell_size = mp.Vector3(sx, sy, 0)  # 2D: z=0

# Add PML on all sides for a scattering/TFSF setup
pml_layers = [mp.PML(dpml)]

##############################################################################
# FREQUENCY / WAVELENGTH RANGE (WHITE LIGHT)
##############################################################################
# Suppose we cover 400-700 nm => 0.4-0.7 µm
lambda_min = 0.4
lambda_max = 0.7
fmin = 1/lambda_max  # ~1.4286
fmax = 1/lambda_min  # ~2.5
fcen = 0.5*(fmin + fmax)  # center frequency
df   = (fmax - fmin)      # bandwidth
nfreq = 60                # number of frequency points to record

##############################################################################
# ANGLE OF INCIDENCE
##############################################################################
# We'll choose, for example, 30-degree incidence in 2D.
# The wave vector in free space: k = 2π/λ. In dimensionless MEEP units,
# we specify a normalized vector = (kx, ky).
# Let's define k_mag ~ fcen * 2π = 2π * (somewhere ~2.0).
# Then kx = k_mag * sin(theta), ky = -k_mag * cos(theta) if we want
# the wave to propagate downward at angle theta from x-axis.
# We'll define a function that returns k_point.

import math

theta_deg = 30.0
theta_rad = math.radians(theta_deg)

# In 2D, let's define k to have:
#   kx = fcen * 2π * sin(theta),
#   ky = -fcen * 2π * cos(theta)  (negative if we want it going downward).
k_mag = 2*math.pi*fcen
kx = k_mag * math.sin(theta_rad)
ky = -k_mag * math.cos(theta_rad)

k_point = mp.Vector3(kx, ky, 0)  # wave vector in 2D

##############################################################################
# GEOMETRY (EXAMPLE)
##############################################################################
# For an OLED-like simulation, you might have:
#   - air region
#   - a thin organic layer at y=0..d
#   - metallic electrode, etc.
# Here, let's just put a single dielectric block as a placeholder
# so we can see reflection/transmission at non-normal incidence.

block_thickness = 1.0
block_center_y  = 0.5*block_thickness

# Suppose n=1.7 => eps=2.89
oled_material = mp.Medium(epsilon=2.89)

geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, block_thickness, mp.inf),
        center=mp.Vector3(0, block_center_y),
        material=oled_material
    )
]

##############################################################################
# TFSF SOURCE SETUP
##############################################################################
# The TFSF region is typically a rectangle that encloses the geometry
# in y but is wide in x. We'll define it so it encloses the block around y=0..1,
# leaving some space for air above/below.

tfsf_ymin = -1.0
tfsf_ymax = 2.0
tfsf_height = tfsf_ymax - tfsf_ymin

tfsf_xmin = -2.0
tfsf_xmax = 2.0
tfsf_width = tfsf_xmax - tfsf_xmin

tfsf_center = mp.Vector3((tfsf_xmin + tfsf_xmax)/2,
                         (tfsf_ymin + tfsf_ymax)/2)
tfsf_size   = mp.Vector3(tfsf_width, tfsf_height)

# Make sure the TFSF region is inside the cell: e.g., cell is 6x6 => range ~±3 in x,y
# We'll set plane-wave incidence from top-left, for instance.

# The TFSF source injects fields with a wave vector "k_point".
tfsf_source = mp.TFSFSource(
    src=mp.GaussianSource(frequency=fcen, fwidth=df),
    center=tfsf_center,
    size=tfsf_size,
    direction=mp.AUTOMATIC,  # MEEP will handle orientation
    k_point=k_point,         # important: sets angle of incidence
    # You can also specify polarization with component, if needed (Ex/Ey).
    # But TFSF typically includes both E and H as needed. In 2D, we might specify polarization explicitly.
    # e.g., polarization=mp.HEZ_POL, etc.
)

sources = [tfsf_source]

##############################################################################
# FLUX MONITORS (MEASURING OUT-COUPLING)
##############################################################################
# Typically, to find "outcoupling" or "transmission," place a flux region
# in the air region below or above the device. Let’s place one in the lower region (y=-2),
# and one in the upper region (y=+2). Then TFSF cancels the incident wave outside,
# so flux outside the TFSF region is purely scattered or transmitted field.

# Lower flux plane:
lower_flux_y = -2.5  # below the TFSF region
lower_flux = mp.FluxRegion(
    center=mp.Vector3(0, lower_flux_y),
    size=mp.Vector3(sx-2*dpml, 0)  # extends across the cell in x
)
# Upper flux plane:
upper_flux_y = 2.5   # above TFSF region
upper_flux = mp.FluxRegion(
    center=mp.Vector3(0, upper_flux_y),
    size=mp.Vector3(sx-2*dpml, 0)
)

##############################################################################
# CREATE SIMULATION
##############################################################################
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution,
    default_material=mp.Medium(epsilon=1.0)  # air background
)

f_lower = sim.add_flux(fcen, df, nfreq, lower_flux)
f_upper = sim.add_flux(fcen, df, nfreq, upper_flux)

##############################################################################
# RUN
##############################################################################
sim.run(until=300)  # run long enough for broadband to propagate

##############################################################################
# ANALYSIS: REFLECTION / TRANSMISSION / OUTCOUPLING
##############################################################################
# In TFSF, the flux below the source region can be interpreted as reflection (if wave
# is coming from above). The flux above might be transmission. You can also do a reference
# run with no device to measure the "incident" flux.

flux_lower = mp.get_fluxes(f_lower)  # array of length nfreq
flux_upper = mp.get_fluxes(f_upper)

freqs = np.linspace(fcen - 0.5*df, fcen + 0.5*df, nfreq)
wavelengths = 1.0 / freqs

# Typically, you do a "reference" simulation with the device removed
# (just air) to get incident flux arrays, then you compute:
#   R(λ) = flux_lower_dev / flux_lower_ref
#   T(λ) = flux_upper_dev / flux_upper_ref
#   outcoupling or absorption can be derived accordingly.
#
# For demonstration, we'll just print the raw flux values here.

print("# lambda (um)\tFlux_lower\tFlux_upper")
for wl, flxL, flxU in zip(wavelengths, flux_lower, flux_upper):
    print(f"{wl:.6f}\t{flxL:.6g}\t{flxU:.6g}")

print("\nDone. To get reflection/transmission, run a 'reference' simulation with no block and divide.")

