import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
n_layers = 5
layer_thickness = 20  # nm, thickness of each layer
total_thickness = layer_thickness * n_layers  # Total device thickness
spatial_positions = np.linspace(0, total_thickness, 1000)  # Spatial axis (nm)
energies = np.linspace(-5, 5, 500)  # Energy axis (eV)

# Define the bandgaps and HOMO/LUMO levels for each layer
layers = [
    {'name': 'HIL', 'E_HOMO': -3.5, 'E_LUMO': -2.0},  # Low bandgap layer
    {'name': 'HTL', 'E_HOMO': -5.0, 'E_LUMO': -2.8},  # Moderate bandgap layer
    {'name': 'EML', 'E_HOMO': -4.8, 'E_LUMO': -3.0},  # Emissive layer
    {'name': 'ETL', 'E_HOMO': -5.0, 'E_LUMO': -2.8},  # Moderate bandgap layer
    {'name': 'EIL', 'E_HOMO': -3.5, 'E_LUMO': -2.0}   # Low bandgap layer
]

# Spatial regions for each layer
layer_boundaries = np.linspace(0, total_thickness, n_layers + 1)

# Standard deviation for the DOS Gaussian function (disorder)
sigma = 0.2  # eV

# Function to calculate the Gaussian DOS
def gaussian_dos(E, E0, sigma):
    return np.exp(-((E - E0)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# Initialize LDOS array (Energy, Space)
LDOS = np.zeros((len(energies), len(spatial_positions)))

# Loop over layers and calculate LDOS in each region
for i, layer in enumerate(layers):
    # Spatial range for this layer
    spatial_mask = (spatial_positions >= layer_boundaries[i]) & (spatial_positions < layer_boundaries[i + 1])
    
    # Calculate LDOS for HOMO and LUMO within this spatial region
    for j, energy in enumerate(energies):
        # HOMO and LUMO contributions
        LDOS[j, spatial_mask] += gaussian_dos(energy, layer['E_HOMO'], sigma) + gaussian_dos(energy, layer['E_LUMO'], sigma)

# Plot the LDOS as a 2D heatmap
plt.figure(figsize=(10, 6))
plt.imshow(LDOS, extent=[0, total_thickness, energies[0], energies[-1]], aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='LDOS (a.u.)')
plt.title('Spatial-Dependent Local Density of States in OLED Structure')
plt.xlabel('Position (nm)')
plt.ylabel('Energy (eV)')
plt.axvline(x=layer_boundaries[1], color='white', linestyle='--', label=layers[0]['name'])
plt.axvline(x=layer_boundaries[2], color='white', linestyle='--', label=layers[1]['name'])
plt.axvline(x=layer_boundaries[3], color='white', linestyle='--', label=layers[2]['name'])
plt.axvline(x=layer_boundaries[4], color='white', linestyle='--', label=layers[3]['name'])
plt.legend(loc='upper right')
plt.show()
