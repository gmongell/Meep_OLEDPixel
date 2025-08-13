import meep as mp
import numpy as np

# ------------------------------------------------------------------------------
# 1. SIMULATION PARAMETERS
# ------------------------------------------------------------------------------
resolution = 50        # pixels per micron (example)
dpml = 0.5             # thickness of PML layer (in microns)
cell_size_x = 0        # 2D sim, no extent in x
cell_size_y = 4        # total size in y, microns (example)

# We define a cell that extends from -cell_size_y/2 to +cell_size_y/2 in y
pml_layers = [mp.PML(thickness=dpml, direction=mp.Y, side=mp.High),
              mp.PML(thickness=dpml, direction=mp.Y, side=mp.Low)]

# ------------------------------------------------------------------------------
# 2. MATERIAL DEFINITIONS (with anisotropy if needed)
# ------------------------------------------------------------------------------
# Example: define an anisotropic organic layer by specifying a permittivity tensor.
# We assume uniaxial anisotropy, epsilon_x != epsilon_y, etc.
# NOTE: MEEP uses complex relative permittivity (epsilon) or conductivity. 
# Real device data must be inserted here.

# For a simple demonstration, let's define:
#   e_x = 2.89 (n = 1.7),  e_y = 2.25 (n = 1.5),  e_z = 2.89 (like uniaxial)
# You can also add imaginary parts for absorption if needed.

organic_epsilon_diag = mp.Matrix3(
    2.89, 0,    0,
    0,    2.25, 0,
    0,    0,    2.89
)

organic_material = mp.Medium(epsilon_tensor=organic_epsilon_diag)

# Define other layers (ITO, metal, substrate) as isotropic or also complex.
# Example isotropic:
ito_material = mp.Medium(epsilon=3.61)  # n=1.9^2
substrate_material = mp.Medium(epsilon=2.25) # n=1.5^2, e.g., glass
metal_material = mp.Medium(epsilon= -18 + 1j*1.0)  # Fake example for Ag at some freq

air_material = mp.Medium(epsilon=1.0)

# ------------------------------------------------------------------------------
# 3. GEOMETRY
# ------------------------------------------------------------------------------
# Let's define a vertical stack:
#
#   air (top)
#   metal layer (thin)
#   organic layer
#   ITO layer
#   substrate (bottom)
#
# We'll place them along y. For demonstration, we define approximate thicknesses:
metal_thickness = 0.05
organic_thickness = 0.1
ito_thickness = 0.05
substrate_thickness = 0.5

# Positioning in y: Let y=0 be the top of the metal, going downward.
# So the metal goes from y=0 to y=-metal_thickness, etc.

geometry = [
    mp.Block(
        material=metal_material,
        size=mp.Vector3(mp.inf, metal_thickness, mp.inf),
        center=mp.Vector3(0, -metal_thickness/2)
    ),
    mp.Block(
        material=organic_material,
        size=mp.Vector3(mp.inf, organic_thickness, mp.inf),
        center=mp.Vector3(0, -(metal_thickness + organic_thickness/2))
    ),
    mp.Block(
        material=ito_material,
        size=mp.Vector3(mp.inf, ito_thickness, mp.inf),
        center=mp.Vector3(0, -(metal_thickness + organic_thickness + ito_thickness/2))
    ),
    mp.Block(
        material=substrate_material,
        size=mp.Vector3(mp.inf, substrate_thickness, mp.inf),
        center=mp.Vector3(0, -(metal_thickness + organic_thickness + ito_thickness
                               + substrate_thickness/2))
    )
]

# The simulation region will extend up in air (above y=0) and below into the substrate,
# with PML boundaries at top and bottom. Adjust cell_size_y accordingly.
cell = mp.Vector3(cell_size_x, cell_size_y, 0)

# ------------------------------------------------------------------------------
# 4. SOURCE DEFINITION (dipole in organic to mimic emission)
# ------------------------------------------------------------------------------
# For an OLED, you might want a dipole inside the organic layer.
# We'll define a simple continuous-wave point dipole source at some wavelength/freq.

# center of the organic layer is at y_center_organic = -(metal_thickness + organic_thickness/2)
y_center_organic = -(metal_thickness + organic_thickness/2)
freq = 2.0  # a chosen frequency in Meep units (1/um). 
            # If you prefer a wavelength ~ 0.5 um, freq = 1/0.5 = 2.

source = [mp.Source(
    src=mp.ContinuousSource(frequency=freq),
    component=mp.Ez,   # or Ex, Ey, etc., or use a loop for different polarizations
    center=mp.Vector3(0, y_center_organic, 0)
)]

# ------------------------------------------------------------------------------
# 5. FLUX MEASUREMENTS (for outcoupling)
# ------------------------------------------------------------------------------
# We place a flux region in the air, above the metal (say at y=+0.2um), 
# to catch transmitted upward radiation (the "light outcoupled to air").
# Another flux region can measure downward flux (reflected/absorbed).
flux_monitor_y = 0.2
trans_flux = mp.FluxRegion(
    center=mp.Vector3(0, flux_monitor_y, 0),
    size=mp.Vector3(mp.inf, 0, mp.inf)
)

# ------------------------------------------------------------------------------
# 6. SIMULATION OBJECT
# ------------------------------------------------------------------------------
sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    resolution=resolution
)

# Add the flux monitor:
trans_flux_obj = sim.add_flux(freq, 0, 1, trans_flux)

# ------------------------------------------------------------------------------
# 7. RUN THE SIMULATION
# ------------------------------------------------------------------------------
sim.run(until=200)  # run for some time until fields decay or converge

# ------------------------------------------------------------------------------
# 8. EXTRACT TRANSMITTED POWER & CALCULATE EFFICIENCY
# ------------------------------------------------------------------------------
# In FDTD, the "power" is integrated flux. If you want the fraction:
transmitted_power = mp.get_fluxes(trans_flux_obj)[0]  # sum over frequencies if broadband

# The question: "What do we normalize against?" 
# For an internal source, you can:
#  (a) measure total emitted power in all directions by placing multiple flux monitors 
#      or using the source power normalization approach,
#  (b) or compare to a reference scenario.
# 
# One approach is:
#   outcoupling_efficiency = transmitted_power / total_source_power
# 
# But total_source_power can be tricky to define for a dipole in FDTD. 
# Typically, you'd do:
#   - Simulate the dipole in "free space" or a known environment to measure 
#     how much power it emits in all directions,
#   - Then compare that to the fraction that emerges here.
# 
# For demonstration, let's just print out the transmitted_flux as a "figure of merit."

print("Transmitted flux (arbitrary units) =", transmitted_power)

# ------------------------------------------------------------------------------
# 9. POST-PROCESSING FOR OUT-COUPLING
# ------------------------------------------------------------------------------
# If you have another flux monitor capturing downward flux, you can do:
#   reflection_power = mp.get_fluxes(ref_flux_obj)[0]
# Then "trapped" or absorbed is whatever's not in (transmitted + reflected).
# 
# Real outcoupling efficiency from a dipole typically requires 
# angle/wavelength integration or a distribution of dipoles/orientations.
# 
# But the above script demonstrates the *basic approach* for anisotropic 
# MEEP simulation with a dipole source and flux measurement.
