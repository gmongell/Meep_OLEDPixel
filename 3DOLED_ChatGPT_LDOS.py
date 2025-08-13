import meep as mp
import numpy as np
import matplotlib.pyplot as plt  # NEW: import matplotlib

# 1. Define simulation cell and resolution
cell_size = mp.Vector3(10, 10, 0)     # 2D example
resolution = 50

# 2. Define geometry
#    For example, a substrate block of some material:
geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, mp.inf),   # fill entire cell in z=0 plane
        center=mp.Vector3(),
        material=mp.Medium(epsilon=1.5)
    )
    # You can add more mp.Block objects to this list
]

# 3. Define boundary layers (PML)
pml_layers = [mp.PML(thickness=1.0)]

# 4. Define sources
#    Example: a continuous source at some frequency
fcen = 0.45  # in Meep’s unit system
sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=fcen),
        component=mp.Ez,
        center=mp.Vector3(-4, 0, 0),
        size=mp.Vector3(0, 10, 0)
    )
]

# 5. Create the simulation object
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    boundary_layers=pml_layers,
    sources=sources,
    resolution=resolution
)

# 6. Add flux regions to measure reflection / transmission
#    For instance, place one flux region to the left, another to the right
nfreq = 500  # number of frequency points to record
df = 0.35    # frequency width around fcen

refl_fr = mp.FluxRegion(center=mp.Vector3(-3.5, 0, 0), size=mp.Vector3(0,10,0))
tran_fr = mp.FluxRegion(center=mp.Vector3( 3.5, 0, 0), size=mp.Vector3(0,10,0))

refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# 7. Run the simulation
sim.run(
    until=200
)

# 8. Retrieve flux data
refl_flux = mp.get_fluxes(refl)
tran_flux = mp.get_fluxes(tran)

# Convert these fluxes into reflection/transmission coefficients, etc.
# ...
print("Reflection flux:", refl_flux)
print("Transmission flux:", tran_flux)

freqs = mp.get_flux_freqs(refl)  # same frequencies used for reflection & transmission

plt.figure(figsize=(6,4))
plt.plot(freqs, refl_flux, 'r-', label='Reflection flux')
plt.plot(freqs, tran_flux, 'b-', label='Transmission flux')
plt.xlabel("Frequency (Meep units)")
plt.ylabel("Flux")
plt.title("Reflection & Transmission Flux")
plt.legend()
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig("flux_vs_frequency.png", dpi=150)

# Optionally, show the plot if running interactively:
plt.show()