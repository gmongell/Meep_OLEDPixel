import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
n_layers = 5
layer_thickness = 20  # nm, thickness of each layer
total_thickness = layer_thickness * n_layers  # Total device thickness
spatial_positions = np.linspace(0, total_thickness, 1000)  # Spatial axis (nm)
energies = np.linspace(-7, 0, 500)  # Energy axis (eV)

# Define the bandgaps and HOMO/LUMO levels for each layer
layers = [
    {'name': 'PEDOT', 'E_HOMO': -5.0, 'E_LUMO': -3.0, 'function': 'HIM'},  # Hole Injection Material
    {'name': 'NPB', 'E_HOMO': -5.4, 'E_LUMO': -2.3, 'function': 'HTM'},   # Hole Transport Material
    {'name': 'PFO', 'E_HOMO': -5.8, 'E_LUMO': -2.2, 'function': 'Emissive (Blue)'},  # Emissive Layer
    {'name': 'Alq₃', 'E_HOMO': -5.9, 'E_LUMO': -3.0, 'function': 'Emissive + ETM'},  # Emissive and Electron Transport
    {'name': 'BPhen', 'E_HOMO': -6.5, 'E_LUMO': -3.0, 'function': 'ETM'},  # Electron Transport Material
]

# Spatial regions for each layer
layer_boundaries = np.linspace(0, total_thickness, n_layers + 1)

# Standard deviation for the DOS Gaussian function (disorder)
sigma = 0.2  # eV

# Function to calculate the Gaussian DOS
def gaussian_dos(E, E0, sigma):
    return np.exp(-((E - E0)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# Function to calculate the parabolic DOS (simple quadratic model near band edges)
def parabolic_dos(E, E_edge, coeff=1):
    return coeff * np.maximum(E - E_edge, 0)**2  # Only valid for E > E_edge

# Initialize LDOS arrays (Energy, Space)
LDOS_gaussian = np.zeros((len(energies), len(spatial_positions)))
LDOS_parabolic = np.zeros((len(energies), len(spatial_positions)))

# Loop over layers and calculate LDOS in each region
for i, layer in enumerate(layers):
    # Spatial range for this layer
    spatial_mask = (spatial_positions >= layer_boundaries[i]) & (spatial_positions < layer_boundaries[i + 1])
    
    # Calculate Gaussian LDOS for HOMO and LUMO within this spatial region
    for j, energy in enumerate(energies):
        # HOMO and LUMO contributions for Gaussian LDOS
        LDOS_gaussian[j, spatial_mask] += gaussian_dos(energy, layer['E_HOMO'], sigma) + gaussian_dos(energy, layer['E_LUMO'], sigma)
        
        # Parabolic DOS for HOMO and LUMO
        LDOS_parabolic[j, spatial_mask] += parabolic_dos(energy, layer['E_HOMO']) + parabolic_dos(energy, layer['E_LUMO'])

# Plot the Gaussian LDOS as a 2D heatmap
plt.figure(figsize=(10, 6))
plt.imshow(LDOS_gaussian, extent=[0, total_thickness, energies[0], energies[-1]], aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='Gaussian LDOS (a.u.)')
plt.title('Gaussian and Parabolic LDOS Overlay for OLED Materials')
plt.xlabel('Position (nm)')
plt.ylabel('Energy (eV)')
plt.axvline(x=layer_boundaries[1], color='white', linestyle='--', label=layers[0]['name'])
plt.axvline(x=layer_boundaries[2], color='white', linestyle='--', label=layers[1]['name'])
plt.axvline(x=layer_boundaries[3], color='white', linestyle='--', label=layers[2]['name'])
plt.axvline(x=layer_boundaries[4], color='white', linestyle='--', label=layers[3]['name'])
plt.legend(loc='upper right')

# Overlay the parabolic LDOS in contour plot
plt.contour(spatial_positions, energies, LDOS_parabolic, levels=10, colors='white', linestyles='dashed')
plt.show()
