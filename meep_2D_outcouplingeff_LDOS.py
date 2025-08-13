import meep as mp
import numpy as np
import matplotlib.pyplot as plt

#############################
# Simulation parameters
#############################
resolution = 50  # pixels/µm

# Define layer thicknesses (in nm) and convert to µm:
thicknesses_nm = [5, 10, 10, 42.2, 133.8]
thicknesses = [t * 1e-3 for t in thicknesses_nm]  # 5 nm -> 0.005 µm, etc.

# Total device thickness (in µm)
device_thickness = sum(thicknesses)

# Use a central wavelength of 0.55 µm (emission band of interest).
central_wl = 0.55  # µm
central_freq = 1 / central_wl  # frequency in 1/µm
freq_width = 0.5 * central_freq  # adjust as needed

# Define frequency resolution for DFT monitors.
nfreq = 200

# Choose simulation cell size.
dpml = 1.0  # thickness of PML layers in µm
sy = device_thickness + 2 * dpml + 2.0  # extra space top/bottom (µm)
sx = 6.0  # cell width (µm)
cell_size = mp.Vector3(sx, sy, 0)

#############################
# Define dispersive materials for each layer.
#############################
# For layer 1 we assume a non-dispersive material with n=1.5:
material_layer1 = mp.Medium(epsilon=1.5**2)

# For the dispersive layers (2–5) we need to mimic the wavelength-dependent behavior.
# Here we provide an example using one Lorentzian susceptibility.
# (Replace the following placeholder parameters with those obtained by fitting your n/k functions.)
# For example, for layer 2 with average n=2.1: epsilon_static ~ 2.1^2 = 4.41.
lorentz_params = dict(frequency=central_freq, gamma=0.1 * central_freq, sigma=0.5)
# Note: sigma controls the oscillator strength; gamma gives the linewidth.

material_layer2 = mp.Medium(epsilon=4.41, 
                            E_susceptibilities=[mp.LorentzianSusceptibility(**lorentz_params)])
# For layers 3, 4, 5 (average n ≈ 1.75, so epsilon_static ~ 3.0625)
material_layer3 = mp.Medium(epsilon=3.0625, 
                            E_susceptibilities=[mp.LorentzianSusceptibility(**lorentz_params)])
material_layer4 = mp.Medium(epsilon=3.0625, 
                            E_susceptibilities=[mp.LorentzianSusceptibility(**lorentz_params)])
material_layer5 = mp.Medium(epsilon=3.0625, 
                            E_susceptibilities=[mp.LorentzianSusceptibility(**lorentz_params)])

materials = [material_layer1, material_layer2, material_layer3, material_layer4, material_layer5]

#############################
# Build the multilayer geometry.
#############################
geometry = []
# Let the layers stack in the y-direction.
# We'll center the device at y=0 (so y extends from -device_thickness/2 to +device_thickness/2).
y_bottom = -device_thickness / 2.0
current_y = y_bottom
for mat, t in zip(materials, thicknesses):
    slab = mp.Block(size=mp.Vector3(mp.inf, t, mp.inf),
                    center=mp.Vector3(0, current_y + t/2.0, 0),
                    material=mat)
    geometry.append(slab)
    current_y += t

#############################
# Define sources and monitors
#############################
# Place a point dipole source for LDOS calculations.
# For instance, place the source at the middle of layer 3.
source_layer_index = 2  # layer 3 (0-indexed)
# Compute the y-center of layer 3:
y_center_layer3 = y_bottom + sum(thicknesses[:source_layer_index]) + thicknesses[source_layer_index]/2.0
source_position = mp.Vector3(0, y_center_layer3, 0)

sources = [mp.Source(src=mp.ContinuousSource(frequency=central_freq, fwidth=freq_width),
                     component=mp.Ez,
                     center=source_position)]

# Add a flux monitor (flux region) to capture outcoupled power.
# Here we put a flux monitor near the top boundary.
flux_region_top = mp.FluxRegion(center=mp.Vector3(0, sy/2 - dpml - 0.1, 0),
                                size=mp.Vector3(sx, 0, 0))
flux_top = mp.add_flux(sim=None,  # temporary placeholder; will be added when the simulation object is created.
                       frequency=central_freq,
                       df=freq_width,
                       nfreq=nfreq,
                       flux_region=flux_region_top)

# In addition, add a near-to-far-field (NTF) transform monitor so that the angular distribution
# (i.e. outcoupling efficiency vs. angle) may be computed.
# (For example, get the far fields on the top side.)
ntf_top = mp.Near2Far(frequency=central_freq, df=freq_width, nfreq=nfreq,
                      flux_region=flux_region_top, direction=mp.North)

# For LDOS, we add a very small DFT volume (essentially at the source point).
dft_point = mp.Volume(center=source_position, size=mp.Vector3(0,0,0))
dft_fields = mp.add_dft_fields([mp.Ez], dft_point,
                               frequency=central_freq, df=freq_width, nfreq=nfreq)

#############################
# Set up simulation
#############################
pml_layers = [mp.PML(dpml)]
sim = mp.Simulation(cell_size=cell_size,
                    geometry=geometry,
                    sources=sources,
                    boundary_layers=pml_layers,
                    resolution=resolution)

# Now add the monitors to the simulation.
sim.add_flux(flux_top.frequency, flux_top.df, flux_top.nfreq, flux_region_top)
sim.add_near2far(ntf_top.frequency, ntf_top.df, ntf_top.nfreq, flux_region_top, direction=mp.North)

#############################
# Run simulation
#############################
simulation_time = 200  # adjust as needed (in time units of µm)
sim.run(until=simulation_time)

#############################
# Extract and post-process data
#############################

# 1. Outcoupling Efficiency as Function of Wavelength
flux_data = mp.get_fluxes(flux_top)
freqs = mp.get_flux_freqs(flux_top)
wavelengths = 1.0 / np.array(freqs)  # since frequency = 1/λ (µm)

plt.figure()
plt.plot(wavelengths, flux_data, 'bo-')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Outcoupled Flux")
plt.title("Outcoupling Efficiency vs. Wavelength")
plt.grid(True)
plt.savefig("outcoupling_vs_wavelength.png", dpi=300)
plt.show()

# 2. Angular Distribution from the Near-to-Far Field transform
# Here we compute far-field flux at different angles.
angles = np.linspace(-80, 80, 161)  # in degrees; adjust range and step as needed
angular_flux = []
for angle in angles:
    # Convert angle from degrees to radians.
    ang_rad = np.radians(angle)
    # Compute far-field amplitude for the given angle.
    flux_at_angle = sim.get_farfield(ntf_top, ang_rad)
    angular_flux.append(np.abs(flux_at_angle)**2)  # intensity ~ amplitude squared

plt.figure()
plt.plot(angles, angular_flux, 'r-')
plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized Far-Field Intensity")
plt.title("Outcoupling Efficiency vs. Angle")
plt.grid(True)
plt.savefig("outcoupling_vs_angle.png", dpi=300)
plt.show()

# 3. Retrieve DFT data at the source location for LDOS calculation.
# (Post-processing to extract the local density of states will require normalization
#  with the corresponding free-space simulation.)
ldos_data = sim.get_dft_array(dft_fields, mp.Ez, 0)
np.save("ldos_Ez_at_source.npy", ldos_data)

# (Further analysis is required to convert the recorded field DFT data into the LDOS.)
