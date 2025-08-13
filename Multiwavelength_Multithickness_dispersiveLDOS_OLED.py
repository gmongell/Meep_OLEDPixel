import meep as mp
import numpy as np
import matplotlib.pyplot as plt

###########################################################################
# USER-DEFINED PARAMETERS
###########################################################################
# Pixel lateral size (simulate a cross-sectional width appropriate for an OLED pixel)
pixel_width = 20.0  # in µm (adjust this value if you wish)

# PML and additional margins (in µm)
dpml = 1.0       # thickness of the PML boundary layers (top/bottom)
margin_y = 2.0   # extra vertical margin added to the simulation cell

# Simulation frequency parameters
central_wl = 0.55           # central wavelength in µm
central_freq = 1 / central_wl # central frequency in 1/µm
freq_width = 0.3 * central_freq  # frequency width for the broadband source
nfreq = 1                   # number of frequency points (using only the central frequency)

# Nominal thicknesses for the five layers (converted from nm to µm)
default_thicknesses = [0.005, 0.01, 0.01, 0.0422, 0.1338]  # µm

# Sweep ranges (µm) for each layer.
# These ranges are chosen near the nominal values (e.g., layer 1 sweeps from 3 to 10 nm)
sweep_ranges = {
    0: np.linspace(0.003, 0.010, 5),  # Layer 1: 3–10 nm
    1: np.linspace(0.005, 0.030, 5),  # Layer 2: 5–30 nm
    2: np.linspace(0.008, 0.040, 5),  # Layer 3: 8–40 nm
    3: np.linspace(0.030, 0.100, 5),  # Layer 4: 30–100 nm
    4: np.linspace(0.100, 0.250, 5)   # Layer 5: 100–250 nm
}

# Resolution: because the layers are extremely thin (few nm) while the pixel width is tens of µm,
# a high resolution is needed. (Be aware that a very high resolution can be computationally expensive.)
resolution = 2000  # pixels/µm (adjust based on available resources)

###########################################################################
# MATERIAL DEFINITIONS
###########################################################################
# For simplicity we use non-dispersive media here.
# (If required, you can embed dispersive models, e.g., via LorentzianSusceptibility.)
medium_layer1 = mp.Medium(epsilon=1.5**2)
medium_layer2 = mp.Medium(epsilon=2.1**2)
medium_layer3 = mp.Medium(epsilon=1.75**2)
medium_layer4 = mp.Medium(epsilon=1.75**2)
medium_layer5 = mp.Medium(epsilon=1.75**2)
mediums = [medium_layer1, medium_layer2, medium_layer3, medium_layer4, medium_layer5]

###########################################################################
# FUNCTION TO RUN LDOS SIMULATION FOR A GIVEN LAYER THICKNESS
###########################################################################
def run_ldos_simulation(sweep_layer, new_thickness):
    """
    Run a MEEP simulation for the OLED multilayer stack where the layer 'sweep_layer'
    has thickness 'new_thickness' (in µm) and the other layers have their default thicknesses.
    A dipole source is placed at the center of the swept layer.
    Returns an LDOS proxy value (squared magnitude of the DFT field at the central frequency).
    """
    # Create a new list of thicknesses, updating the swept layer.
    thicknesses = default_thicknesses.copy()
    thicknesses[sweep_layer] = new_thickness
    total_thickness = sum(thicknesses)
    
    # Build the geometry (device centered at y = 0)
    geometry = []
    y_bottom = -total_thickness / 2.0
    cumulative = 0.0  # cumulative thickness from the bottom
    layer_centers = []
    for i, t in enumerate(thicknesses):
        center_y = y_bottom + cumulative + t/2.0
        layer_centers.append(center_y)
        block = mp.Block(size=mp.Vector3(mp.inf, t, mp.inf),
                         center=mp.Vector3(0, center_y, 0),
                         material=mediums[i])
        geometry.append(block)
        cumulative += t

    # Define simulation cell (width = pixel_width; height = device thickness plus margins)
    cell_y = total_thickness + 2 * dpml + margin_y
    cell_size = mp.Vector3(pixel_width, cell_y, 0)
    
    # Define PML layers (along y direction)
    pml_layers = [mp.PML(dpml, direction=mp.Y)]
    
    # Position the dipole source at the center of the swept layer
    dipole_y = layer_centers[sweep_layer]
    source_position = mp.Vector3(0, dipole_y, 0)
    sources = [mp.Source(src=mp.ContinuousSource(frequency=central_freq, fwidth=freq_width),
                         component=mp.Ez,
                         center=source_position)]
    
    # Create the simulation
    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)
    
    # Add a DFT monitor at the location of the dipole (a near-zero volume)
    dft_vol = mp.Volume(center=source_position, size=mp.Vector3(0, 0, 0))
    dft_fields = sim.add_dft_fields([mp.Ez], dft_vol,
                                    frequency=central_freq, df=freq_width, nfreq=nfreq)
    
    # Run the simulation for a fixed duration (adjust simulation time as needed)
    sim_time = 200  # time units (µm, since c = 1 in MEEP units)
    sim.run(until=sim_time)
    
    # Extract the DFT data (assuming only one frequency point is stored)
    field_dft = sim.get_dft_array(dft_fields, mp.Ez, 0)
    
    # Use the squared magnitude of the DFT field as an LDOS proxy.
    ldos = np.abs(field_dft)**2
    return ldos

###########################################################################
# SWEEP OVER LAYER THICKNESSES AND COLLECT LDOS DATA
###########################################################################
ldos_results = {}  # dictionary for storing sweep data: key = layer index, value = (thicknesses, LDOS)
for layer in range(5):
    thickness_values = sweep_ranges[layer]
    ldos_values = []
    print(f"Running LDOS sweep for Layer {layer+1}...")
    for th in thickness_values:
        print(f"  Thickness = {th*1e3:.2f} nm")
        ldos_val = run_ldos_simulation(layer, th)
        ldos_values.append(ldos_val)
    ldos_results[layer] = (thickness_values, ldos_values)

###########################################################################
# PLOT RESULTS
###########################################################################
fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=False)
for layer in range(5):
    th_vals, ldos_vals = ldos_results[layer]
    # Convert thickness to nm for plotting
    thickness_nm = th_vals * 1e3  
    axs[layer].plot(thickness_nm, ldos_vals, 'o-', label=f'Layer {layer+1}')
    axs[layer].set_xlabel("Layer Thickness (nm)")
    axs[layer].set_ylabel("LDOS (a.u.)")
    axs[layer].set_title(f"LDOS vs. Thickness for Layer {layer+1}")
    axs[layer].grid(True)
    axs[layer].legend()

plt.tight_layout()
plt.savefig("LDOS_vs_thickness_each_layer.png", dpi=300)
plt.show()
