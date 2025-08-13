import meep as mp

##############################################################################
# SIMULATION PARAMETERS
##############################################################################
resolution = 100           # grid points per micron (example)
dpml = 0.5                # thickness of PML
cell_height = 4.0         # total height of simulation region in microns
cell = mp.Vector3(0, cell_height, 0)   # 2D sim: infinite in x, extent in y

# PML layers on top and bottom in y
pml_layers = [
    mp.PML(dpml, direction=mp.Y, side=mp.High),
    mp.PML(dpml, direction=mp.Y, side=mp.Low)
]

##############################################################################
# MATERIAL DEFINITIONS
##############################################################################
# For demonstration, we define:
#   - a metal layer (using a placeholder complex epsilon)
#   - an anisotropic organic layer
#   - an ITO layer (isotropic)
#   - a substrate layer (e.g., glass)
#   - top region is air by default

# -- Metal (e.g., Silver-ish, but made-up values for example) --
metal_n, metal_k = 0.14, 4.29  # example at ~500 nm
metal_epsilon = (metal_n + 1j*metal_k)**2

metal_material = mp.Medium(epsilon_diag=mp.Vector3(metal_epsilon,
                                                   metal_epsilon,
                                                   metal_epsilon))

# -- Anisotropic organic layer --
# For instance, let eps_x=2.89 (n=1.7), eps_y=2.25 (n=1.5), eps_z=2.89.
# You can change these values as needed.
anisotropic_organic = mp.Medium(
    epsilon_diag=mp.Vector3(2.89, 2.25, 2.89)
)

# -- ITO (isotropic, example n=1.9 => eps=3.61) --
ito_material = mp.Medium(epsilon=3.61)

# -- Substrate (e.g., glass, n=1.5 => eps=2.25) --
substrate_material = mp.Medium(epsilon=2.25)

##############################################################################
# GEOMETRY (stack definition)
##############################################################################
# Let's place the stack along the y-axis, with air above (y>0).
# We'll define layer thicknesses from top to bottom:
metal_thickness      = 0.03
organic_thickness    = 0.1
ito_thickness        = 1
substrate_thickness  = 0.5

# The top of the metal is at y=0 (interface with air).
# Then we go downward for each layer. For convenience:
y0 = 0.0

# Metal layer
metal_layer = mp.Block(
    material=metal_material,
    size=mp.Vector3(mp.inf, metal_thickness, mp.inf),
    center=mp.Vector3(0, y0 - metal_thickness/2, 0)
)
y0 -= metal_thickness

# Organic (anisotropic) layer
organic_layer = mp.Block(
    material=anisotropic_organic,
    size=mp.Vector3(mp.inf, organic_thickness, mp.inf),
    center=mp.Vector3(0, y0 - organic_thickness/2, 0)
)
y0 -= organic_thickness

# ITO layer
ito_layer = mp.Block(
    material=ito_material,
    size=mp.Vector3(mp.inf, ito_thickness, mp.inf),
    center=mp.Vector3(0, y0 - ito_thickness/2, 0)
)
y0 -= ito_thickness

# Substrate layer
substrate_layer = mp.Block(
    material=substrate_material,
    size=mp.Vector3(mp.inf, substrate_thickness, mp.inf),
    center=mp.Vector3(0, y0 - substrate_thickness/2, 0)
)
y0 -= substrate_thickness

geometry = [metal_layer, organic_layer, ito_layer, substrate_layer]

##############################################################################
# SOURCE DEFINITION
##############################################################################
# We'll place a dipole source in the middle of the organic layer
# The middle of the organic layer is at y_center_organic = top_of_organic - (organic_thickness/2).
# In our definition, the top_of_organic is y0+ organic_thickness from the block definition, 
# but we already computed it. Let's do it directly:

metal_top    = 0.0
organic_top  = metal_top - metal_thickness
y_center_organic = organic_top - (organic_thickness/2)  # halfway down

# Frequency/wavelength
# Suppose we pick a single wavelength ~ 0.5 um => freq=2.0 (since freq=1/lambda in Meep units).
freq = 2.0

sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=freq),
        component=mp.Ez,  # or Ex/Ey depending on polarization of interest
        center=mp.Vector3(0, y_center_organic, 0)
    )
]

##############################################################################
# FLUX MONITOR IN AIR
##############################################################################
# We place a flux plane above the metal to measure how much light gets out.
# Let’s put it at y=+0.1, well above the metal (which ends at y=0).
flux_monitor_y = 0.2
trans_region = mp.FluxRegion(
    center=mp.Vector3(0, flux_monitor_y, 0),
    size=mp.Vector3(mp.inf, 0, mp.inf)
)

##############################################################################
# CREATE AND RUN SIMULATION
##############################################################################
sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    boundary_layers=pml_layers,
    sources=sources,
    resolution=resolution
)

# Add flux object
trans_flux_obj = sim.add_flux(freq, 0, 1, trans_region)

# Run until fields have mostly decayed or stabilized
sim.run(until=200)

# Extract transmitted flux (in arbitrary units)
trans_flux_val = mp.get_fluxes(trans_flux_obj)[0]

print(f"Transmitted flux above the device = {trans_flux_val:.6g} (arbitrary units)")

##############################################################################
# NOTES ON NORMALIZATION & "OUT-COUPLING EFFICIENCY"
##############################################################################
# - In a real outcoupling calculation, you must also:
#   1. Account for total dipole emission power. Typically, you'd measure 
#      dipole emission in free space or an infinite homogeneous medium 
#      as a reference, or place flux monitors all around (sides, bottom).
#   2. Possibly run multiple frequencies or a broadband source and integrate.
#   3. Possibly run multiple angles or 3D if the device is not uniform out of plane.
#
# The ratio of transmitted power above the device to total emitted power 
# would give you a rough outcoupling fraction. 
# This script simply demonstrates setting up an anisotropic layer 
# and measuring flux in the air region.
