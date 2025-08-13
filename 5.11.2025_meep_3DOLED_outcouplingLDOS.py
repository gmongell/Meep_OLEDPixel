import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os

###############################################################################
# Simulation Parameters
###############################################################################

resolution = 20  # pixels/µm

wl_min = 0.4  # microns
wl_max = 0.7
fmin = 1 / wl_max
fmax = 1 / wl_min
fcen = 0.5 * (fmin + fmax)
df = fmax - fmin

nfreq = 100
frequencies = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, nfreq)
wavelengths_nm = 1 / frequencies * 1e3

# Layer stack thicknesses (µm)
t_ag = 0.03
t_npb = 0.06
t_alq3 = 0.06
t_ito = 0.06
t_glass = 0.1
t_air_top = 0.3
t_air_bot = 0.3

sy = t_air_bot + t_glass + t_ito + t_alq3 + t_npb + t_ag + t_air_top
cell_size = mp.Vector3(1, sy, 1)

pml_layers = [mp.PML(thickness=0.2, direction=mp.Y)]

###############################################################################
# Materials (simplified)
###############################################################################

n_air = 1.0
n_ag = 0.14 + 3.98j
n_npb = 1.7
n_alq3 = 1.74
n_ito = 1.8
n_glass = 1.52

materials = {
    "Air": mp.Medium(index=n_air),
    "Ag": mp.Medium(index=n_ag),
    "NPB": mp.Medium(index=n_npb),
    "Alq3": mp.Medium(index=n_alq3),
    "ITO": mp.Medium(index=n_ito),
    "Glass": mp.Medium(index=n_glass),
}

###############################################################################
# Geometry (bottom to top)
###############################################################################

geometry = []
y0 = -0.5 * sy + t_air_bot

def add_layer(material, thickness):
    global y0
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, thickness, mp.inf),
        center=mp.Vector3(0, y0 + 0.5 * thickness),
        material=materials[material]
    ))
    y0 += thickness

add_layer("Glass", t_glass)
add_layer("ITO", t_ito)
alq3_start = y0
add_layer("Alq3", t_alq3)
alq3_end = y0
add_layer("NPB", t_npb)
add_layer("Ag", t_ag)

# Output directory
outdir = "ldos_profiles"
os.makedirs(outdir, exist_ok=True)

###############################################################################
# Depth sampling through Alq3
###############################################################################

npoints = 20
y_positions = np.linspace(alq3_start, alq3_end, npoints)

###############################################################################
# Orientation sampling: 0° to 90° in 5° increments
###############################################################################

angles_deg = np.arange(0, 91, 5)

for angle in angles_deg:
    theta = np.radians(angle)
    orientation = mp.Vector3(np.sin(theta), 0, np.cos(theta))

    print(f"Simulating angle {angle}° ...")

    ldos_profile = []

    for y in y_positions:
        source = mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            center=mp.Vector3(0, y, 0),
            size=mp.Vector3(0, 0, 0),
            component=mp.Vector3Field(orientation)
        )

        sim = mp.Simulation(
            cell_size=cell_size,
            geometry=geometry,
            sources=[source],
            boundary_layers=pml_layers,
            resolution=resolution,
            dimensions=3
        )

        sim.run(until=200)
        ldos = sim.ldos(fcen, df, nfreq)
        ldos_profile.append(ldos)

    # Convert to numpy array: shape (npoints, nfreq)
    ldos_profile = np.array(ldos_profile)

    # Save data to CSV (depth vs. LDOS spectrum)
    csv_file = os.path.join(outdir, f"ldos_angle_{angle:02d}deg.csv")
    with open(csv_file, "w") as f:
        f.write("Depth(um)," + ",".join(f"{w:.2f}nm" for w in wavelengths_nm) + "\n")
        for yval, ldos_row in zip(y_positions, ldos_profile):
            f.write(f"{yval}," + ",".join(str(v) for v in ldos_row) + "\n")

    # Plot LDOS vs depth at select wavelengths
    plt.figure()
    idxs = [0, nfreq//4, nfreq//2, 3*nfreq//4, nfreq-1]
    for i in idxs:
        plt.plot(y_positions, ldos_profile[:, i], label=f"{wavelengths_nm[i]:.0f} nm")
    plt.xlabel("Position in Alq3 (μm)")
    plt.ylabel("LDOS (arb. units)")
    plt.title(f"LDOS vs Depth, Angle {angle}°")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"ldos_profile_{angle:02d}deg.png"), dpi=300)
    plt.close()
