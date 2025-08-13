import numpy as np
import matplotlib.pyplot as plt

# Constants for the DOS calculation
band_gap = 2.0  # eV, energy difference between LUMO and HOMO
sigma = 0.2     # eV, standard deviation for Gaussian DOS (disorder)
E_HOMO = -3.0   # eV, center of HOMO level
E_LUMO = E_HOMO + band_gap  # eV, center of LUMO level

# Energy range
E = np.linspace(E_HOMO - 2, E_LUMO + 2, 1000)  # Energy axis (eV)

# Gaussian function for DOS
def gaussian_dos(E, E0, sigma):
    return np.exp(-((E - E0)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# Density of states for HOMO and LUMO
DOS_HOMO = gaussian_dos(E, E_HOMO, sigma)
DOS_LUMO = gaussian_dos(E, E_LUMO, sigma)

# Plotting the DOS
plt.figure(figsize=(8, 6))
plt.plot(E, DOS_HOMO, label='HOMO', color='blue')
plt.plot(E, DOS_LUMO, label='LUMO', color='red')
plt.fill_between(E, DOS_HOMO, color='blue', alpha=0.3)
plt.fill_between(E, DOS_LUMO, color='red', alpha=0.3)

# Labels and title
plt.title('Density of States (DOS) vs Energy for OLED with Constant Band Gap', fontsize=14)
plt.xlabel('Energy (eV)', fontsize=12)
plt.ylabel('Density of States (a.u.)', fontsize=12)
plt.axvline(x=E_HOMO, color='blue', linestyle='--', label='HOMO Level')
plt.axvline(x=E_LUMO, color='red', linestyle='--', label='LUMO Level')
plt.legend()
plt.grid(True)
plt.show()
