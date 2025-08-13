import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def compute_ldos_at_point(x, y, fstart, fstop, nfreq, resolution=50):
    """
    Create a simulation and compute the LDOS for a dipole source placed at (x,y).
    Returns the *integrated* LDOS over the specified frequency range.
    """

    # 1. Define the simulation cell (2D example).
    cell_size = mp.Vector3(10, 10, 0)

    # 2. Simple geometry (a uniform medium with epsilon=1.5)
    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, mp.inf),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=1.5)
        )
    ]

    # 3. PML layers
    pml_layers = [mp.PML(1.0)]

    # 4. Frequency sampling for LDOS
    freqs = np.linspace(fstart, fstop, nfreq)

    # 5. Define a single dipole source at (x,y) spanning the entire freq range
    #    We choose a GaussianSource centered at the midpoint of [fstart,fstop]
    #    with a fwidth that covers the entire band.
    src = mp.Source(
        src=mp.GaussianSource(
            frequency=0.5*(fstart + fstop),
            fwidth=(fstop - fstart)
        ),
        component=mp.Ez,         # or Ex, Ey, etc. for other polarizations
        center=mp.Vector3(x, y)  # location of dipole
    )

    # 6. Build the Simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=[src],
        resolution=resolution
    )

    # 7. Define the LDOS object
    ldos = mp.DFT-LDOS(frequencies=freqs)

    # 8. Run until fields have time to decay (important!)
    sim.run(mp.ldos(ldos), until_after_sources=200)

    # 9. The LDOS values at each frequency are stored in ldos.data (length = nfreq).
    #    Sum them up to get an "integrated" LDOS in that frequency band.
    return np.sum(ldos.data)

def main():
    # Frequency range
    fstart, fstop = 0.35, 0.85
    nfreq = 501

    # Grid points (be cautious with large grids => many simulations!)
    Nx = 5
    Ny = 5

    # Physical extent in x,y
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3

    xs = np.linspace(x_min, x_max, Nx)
    ys = np.linspace(y_min, y_max, Ny)

    # 2D array to store the integrated LDOS
    ldos_map = np.zeros((Ny, Nx), dtype=float)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            print(f"Computing LDOS at (x={x:.2f}, y={y:.2f}) ...")
            val = compute_ldos_at_point(x, y, fstart, fstop, nfreq, resolution=10)
            ldos_map[iy, ix] = val

    # Plotting
    plt.figure(figsize=(6,5))
    # 'extent' sets axis tick labels to real coordinates
    # 'origin=lower' puts (y_min) at the bottom
    plt.imshow(ldos_map, extent=[x_min, x_max, y_min, y_max],
               origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(label="Integrated LDOS")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D LDOS Map (Integrated over f=0.10 to 0.20)")
    plt.tight_layout()

    # Save the plot to a PNG file
    plt.savefig("ldos_map.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
