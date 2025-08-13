#!/usr/bin/env python3

"""
OLED Outcoupling Efficiency Optimization Script

This script performs a comprehensive, multi-stage optimization of Organic Light-
Emitting Diode (OLED) outcoupling efficiency using a combination of the Transfer
Matrix Method (TMM) for rapid planar analysis and the Finite-Difference Time-
Domain (FDTD) method for rigorous scattering analysis.

Author: Award-Winning Writer & Computational Physicist
Date: August 7, 2025

Dependencies:
- meep
- tmm
- numpy
- pandas
- scipy
- matplotlib
- opticalmaterialspy

Installation (Fedora Linux):
---------------------------
# 1. Install system dependencies for Meep
sudo dnf install -y git-core autoconf automake libtool libgsl-devel \
    libpng-devel libjpeg-devel swig hdf5-devel libgfortran-static \
    openmpi-devel lapack-devel blas-devel fftw-devel

# 2. Install Python dependencies
pip install --upgrade pip
pip install numpy scipy pandas matplotlib
pip install tmm opticalmaterialspy

# 3. Install Meep from source (recommended for MPI support)
# Follow the official Meep installation guide for Linux.
# Alternatively, for a simpler non-MPI build:
pip install meep

---------------------------
"""

import meep as mp
import numpy as np
import pandas as pd
import tmm
import opticalmaterialspy as mat
import itertools
import random
import os
import time
from datetime import timedelta

# --- SCRIPT CONFIGURATION ---

# Define the parameter space for the optimization sweep.
# The script will iterate through all combinations of these values.
PARAMETER_SPACE = {
    'htl_thickness': [20, 30, 40],  # Thickness of the Hole Transport Layer (nm)
    'etl_thickness': [20, 30, 40],  # Thickness of the Electron Transport Layer (nm)
    'np_concentration': [0.0, 0.1, 0.2], # Volume fraction of nanoparticles in the scattering layer
    'np_radius': [30, 40, 50], # Radius of nanoparticles (nm)
}

# Define the base OLED device structure.
# Thicknesses will be overridden by the PARAMETER_SPACE values during the sweep.
# Format: [name, material_key, thickness (nm), layer_type]
# layer_type can be 'stack' or 'scattering'
OLED_STACK_BASE = [
    ['ITO', 'ITO', 150, 'stack'],
    ['HIL', 'PEDOT:PSS', 40, 'stack'],
    ['HTL', 'NPB', 30, 'stack'],
    ['EML', 'Alq3', 20, 'stack'], # Emissive Layer
    ['ETL', 'Alq3', 30, 'stack'], # Electron Transport Layer
    ['EIL', 'LiF', 1, 'stack'],
    # Cathode is semi-infinite, defined in simulators
]

# Define the scattering layer to be inserted if np_concentration > 0
SCATTERING_LAYER = {
    'name': 'ScatteringLayer',
    'host_material': 'SOG', # Spin-on-glass host matrix
    'np_material': 'YSZ',   # YSZ nanoparticles
    'thickness': 2500,      # Total thickness of the layer in nm
    'position_after': 'ITO' # Insert this layer after the ITO layer
}


# Define materials by mapping a key to its refractiveindex.info URL [4]
# This allows for dynamic fetching of optical constants.
MATERIAL_URLS = {
    'Air': None, # Special case
    'Glass': 'https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson',
    'ITO': 'https://refractiveindex.info/?shelf=other&book=In2O3-SnO2&page=Moerland',
    'PEDOT:PSS': 'https://refractiveindex.info/?shelf=other&book=PEDOT-PSS&page=Chen',
    'NPB': 'https://refractiveindex.info/?shelf=organic&book=NPB&page=Ghosh',
    'Alq3': 'https://refractiveindex.info/?shelf=organic&book=Alq3&page=Ghosh',
    'LiF': 'https://refractiveindex.info/?shelf=main&book=LiF&page=Li',
    'Al': 'https://refractiveindex.info/?shelf=main&book=Al&page=Rakic',
    'SOG': 'https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson', # Approximating SOG with Fused Silica
    'YSZ': 'https://refractiveindex.info/?shelf=other&book=ZrO2-Y2O3&page=Wood'
}

# General simulation settings
SIMULATION_SETTINGS = {
    'wavelength_min_nm': 400,
    'wavelength_max_nm': 700,
    'wavelength_points': 31,
    'tmm_threshold': 0.20, # Only run Meep if TMM avg. transmission is above this
    'meep_resolution': 20, # pixels/um. Increase for accuracy, decrease for speed.
    'results_filename': 'oled_optimization_results.csv'
}

# --- MODULE 1: MATERIAL DATA HANDLER ---

class MaterialHandler:
    """
    Handles fetching and caching of material optical constants.
    """
    def __init__(self, urls):
        self._urls = urls
        self._cache = {}
        self.wavelengths_nm = np.linspace(
            SIMULATION_SETTINGS['wavelength_min_nm'],
            SIMULATION_SETTINGS['wavelength_max_nm'],
            SIMULATION_SETTINGS['wavelength_points']
        )

    def get_nk_array(self, material_key):
        """
        Fetches complex refractive index over the simulation wavelength range.
        """
        if material_key in self._cache:
            return self._cache[material_key]

        print(f"  Fetching optical data for {material_key}...")
        if material_key == 'Air':
            nk_array = np.ones_like(self.wavelengths_nm, dtype=complex)
        else:
            url = self._urls.get(material_key)
            if not url:
                raise ValueError(f"URL for material '{material_key}' not found.")
            
            try:
                material_obj = mat.RefractiveIndexWeb(url)
                n_values = material_obj.n(self.wavelengths_nm)
                k_values = material_obj.k(self.wavelengths_nm)
                nk_array = n_values + 1j * k_values
            except Exception as e:
                print(f"    ERROR: Could not fetch data for {material_key}. {e}")
                # Fallback to a default dielectric if fetching fails
                nk_array = 1.5 * np.ones_like(self.wavelengths_nm, dtype=complex)


        self._cache[material_key] = nk_array
        return nk_array

# --- MODULE 2: TMM SIMULATOR (PLANAR STACK) ---

class TMM_Simulator:
    """
    Performs rapid optical simulation of planar OLED stacks using TMM. [5, 6, 7]
    """
    def __init__(self, wavelengths_nm):
        self.wavelengths_nm = wavelengths_nm

    def run(self, params, material_handler, oled_stack):
        """
        Calculates the average transmission of a planar stack.
        """
        d_list = [np.inf] # Start with semi-infinite Air/Glass
        
        # Build thickness list from the stack definition
        for layer in oled_stack:
            layer_name, _, default_thickness, _ = layer
            # Use thickness from parameter sweep if available, else use default
            thickness = params.get(f'{layer_name.lower()}_thickness', default_thickness)
            d_list.append(thickness)
        
        d_list.append(np.inf) # End with semi-infinite Cathode

        total_transmission = 0
        
        for i, lam_vac in enumerate(self.wavelengths_nm):
            n_list = [material_handler.get_nk_array('Glass')[i]]
            for layer in oled_stack:
                _, material_key, _, _ = layer
                n_list.append(material_handler.get_nk_array(material_key)[i])
            n_list.append(material_handler.get_nk_array('Al')[i])

            # Calculate for unpolarized light at normal incidence
            try:
                T = tmm.unpolarized_RT(n_list, d_list, 0, lam_vac)['T']
                total_transmission += T
            except Exception as e:
                # Handle potential numerical issues in tmm
                print(f"    TMM calculation failed for wavelength {lam_vac} nm: {e}")
                continue

        return total_transmission / len(self.wavelengths_nm)


# --- MODULE 3: MEEP SIMULATOR (SCATTERING STRUCTURES) ---

class Meep_Simulator:
    """
    Performs rigorous FDTD simulation of OLEDs with scattering nanoparticles. [8, 9, 10, 11]
    """
    def __init__(self, wavelengths_nm):
        self.wavelengths_nm = wavelengths_nm
        self.freqs = 1 / (self.wavelengths_nm * 1e-3) # Convert nm to um for Meep
        self.fcen = np.mean(self.freqs)
        self.df = np.max(self.freqs) - np.min(self.freqs)

    def run(self, params, material_handler, oled_stack):
        """
        Runs a full 2D FDTD simulation and returns the outcoupling efficiency.
        """
        # --- 1. Define Geometry and Materials ---
        resolution = SIMULATION_SETTINGS['meep_resolution']
        
        # Get layer thicknesses from params
        stack_layers = []
        current_z = 0
        
        # Substrate (Glass)
        t_glass = 1000 # 1 um thick glass substrate in simulation
        
        # Build the device stack
        for layer_info in oled_stack:
            name, material_key, default_thickness, _ = layer_info
            thickness = params.get(f'{name.lower()}_thickness', default_thickness) * 1e-3 # nm to um
            
            # Find the position of the emissive layer
            if name == 'EML':
                eml_pos_z = current_z + thickness / 2
                eml_thickness = thickness

            mat = mp.Medium(index=np.real(material_handler.get_nk_array(material_key)[len(self.wavelengths_nm)//2]))
            stack_layers.append(mp.Block(material=mat, size=mp.Vector3(mp.inf, mp.inf, thickness), center=mp.Vector3(0, 0, current_z + thickness / 2)))
            current_z += thickness

        t_stack = current_z
        t_cathode = 100 * 1e-3 # 100 nm Al cathode
        
        # Simulation cell size
        sx = 4 # um
        sy = 4 # um
        dpml = 1.0 # PML thickness
        sz = dpml + t_glass*1e-3 + t_stack + t_cathode + dpml
        cell_size = mp.Vector3(sx, sy, sz)
        
        # Shift entire geometry to center it in the cell
        z_shift = -sz/2 + dpml
        
        geometry = [mp.Block(material=mp.Medium(index=np.real(material_handler.get_nk_array('Glass')[len(self.wavelengths_nm)//2])),
                             size=mp.Vector3(mp.inf, mp.inf, t_glass*1e-3),
                             center=mp.Vector3(0, 0, z_shift + t_glass*1e-3/2))]
        
        for layer in stack_layers:
            layer.center.z += z_shift + t_glass*1e-3
            geometry.append(layer)
        
        geometry.append(mp.Block(material=mp.Medium(index=np.real(material_handler.get_nk_array('Al')[len(self.wavelengths_nm)//2])),
                                 size=mp.Vector3(mp.inf, mp.inf, t_cathode),
                                 center=mp.Vector3(0, 0, z_shift + t_glass*1e-3 + t_stack + t_cathode/2)))

        # Add nanoparticles if concentration > 0
        if params['np_concentration'] > 0:
            np_radius_um = params['np_radius'] * 1e-3
            host_mat_key = SCATTERING_LAYER['host_material']
            np_mat_key = SCATTERING_LAYER['np_material']
            scat_layer_thickness_um = SCATTERING_LAYER['thickness'] * 1e-3
            
            # Find the z-position for the scattering layer
            z_pos_scat = 0
            for layer in oled_stack:
                name, _, default_thickness, _ = layer
                thickness = params.get(f'{name.lower()}_thickness', default_thickness) * 1e-3
                z_pos_scat += thickness
                if name == SCATTERING_LAYER['position_after']:
                    break
            
            z_center_scat = z_shift + t_glass*1e-3 + z_pos_scat + scat_layer_thickness_um / 2

            # Add host material block
            geometry.append(mp.Block(material=mp.Medium(index=np.real(material_handler.get_nk_array(host_mat_key)[len(self.wavelengths_nm)//2])),
                                     size=mp.Vector3(mp.inf, mp.inf, scat_layer_thickness_um),
                                     center=mp.Vector3(0, 0, z_center_scat)))

            # Add nanoparticles
            np_material = mp.Medium(index=np.real(material_handler.get_nk_array(np_mat_key)[len(self.wavelengths_nm)//2]))
            
            # Calculate number of NPs to add for the given concentration
            vol_layer = sx * sy * scat_layer_thickness_um
            vol_np = 4/3 * np.pi * np_radius_um**3
            num_nps = int(params['np_concentration'] * vol_layer / vol_np)
            
            for _ in range(num_nps):
                x = random.uniform(-sx/2, sx/2)
                y = random.uniform(-sy/2, sy/2)
                z = random.uniform(z_center_scat - scat_layer_thickness_um/2, z_center_scat + scat_layer_thickness_um/2)
                geometry.append(mp.Sphere(radius=np_radius_um, center=mp.Vector3(x,y,z), material=np_material))


        # --- 2. Define Sources and Monitors ---
        # Source is a dipole in the EML
        src_pos_z = z_shift + t_glass*1e-3 + eml_pos_z
        sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.df),
                              component=mp.Ez, # Default component, will be overwritten
                              center=mp.Vector3(0,0,src_pos_z))] # This will be changed in the loop

        # Flux monitors
        # Box around source to measure total power
        box_z_half = eml_thickness / 2 + 0.01 # 10nm buffer
        flux_box_center = mp.Vector3(0, 0, src_pos_z)
        fr_source_box_z = mp.FluxRegion(center=flux_box_center-mp.Vector3(0,0,box_z_half), size=mp.Vector3(sx-2*dpml, sy-2*dpml, 0))
        fr_source_box_z_rev = mp.FluxRegion(center=flux_box_center+mp.Vector3(0,0,box_z_half), size=mp.Vector3(sx-2*dpml, sy-2*dpml, 0))
        
        # Monitor in glass to measure outcoupled power
        fr_out = mp.FluxRegion(center=mp.Vector3(0, 0, z_shift + 0.1), size=mp.Vector3(sx-2*dpml, sy-2*dpml, 0))

        # --- 3. Run Simulations for each Polarization ---
        total_flux_out = np.zeros(len(self.freqs))
        total_flux_source = np.zeros(len(self.freqs))

        for pol in [mp.Ex, mp.Ey, mp.Ez]:
            sim = mp.Simulation(cell_size=cell_size,
                                geometry=geometry,
                                sources=sources,
                                boundary_layers=[mp.PML(dpml)],
                                resolution=resolution)
            
            sim.sources[0].component = pol

            # Add flux monitors
            source_flux_z = sim.add_flux(self.fcen, self.df, len(self.freqs), fr_source_box_z)
            source_flux_z_rev = sim.add_flux(self.fcen, self.df, len(self.freqs), fr_source_box_z_rev)
            out_flux = sim.add_flux(self.fcen, self.df, len(self.freqs), fr_out)

            # Run simulation
            sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0,0,src_pos_z), 1e-8))

            # Accumulate flux data
            # Total power is flux leaving the box around the source
            total_flux_source += np.array(mp.get_fluxes(source_flux_z_rev)) - np.array(mp.get_fluxes(source_flux_z))
            total_flux_out += -np.array(mp.get_fluxes(out_flux)) # Flux is negative in -z direction
            
            sim.reset_meep()

        # --- 4. Calculate Efficiency ---
        # Integrate flux over the frequency range
        avg_source_power = np.trapz(total_flux_source, self.freqs)
        avg_out_power = np.trapz(total_flux_out, self.freqs)

        if avg_source_power == 0:
            return 0.0
            
        outcoupling_efficiency = avg_out_power / avg_source_power
        return outcoupling_efficiency


# --- MODULE 4: MAIN OPTIMIZATION ORCHESTRATOR ---

class Optimizer:
    """
    Manages the optimization workflow, iterates through parameters,
    runs simulations, and analyzes results.
    """
    def __init__(self, param_space, stack_base, sim_settings):
        self.param_space = param_space
        self.stack_base = stack_base
        self.sim_settings = sim_settings
        self.results_df = pd.DataFrame()

    def run_optimization(self):
        """
        Executes the main optimization loop.
        """
        start_time = time.time()
        
        # Generate all parameter combinations
        keys, values = zip(*self.param_space.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        total_runs = len(param_combinations)
        
        print(f"--- Starting OLED Optimization ---")
        print(f"Total parameter combinations to test: {total_runs}")
        
        material_handler = MaterialHandler(MATERIAL_URLS)
        tmm_sim = TMM_Simulator(material_handler.wavelengths_nm)
        meep_sim = Meep_Simulator(material_handler.wavelengths_nm)

        for i, params in enumerate(param_combinations):
            run_start_time = time.time()
            print(f"\n--- Running Simulation {i+1}/{total_runs} ---")
            print(f"Parameters: {params}")

            # Construct the full OLED stack for this run
            current_oled_stack = list(self.stack_base)
            if params.get('np_concentration', 0) > 0:
                # Find insertion index
                insert_idx = next((i + 1 for i, layer in enumerate(current_oled_stack) if layer[0] == SCATTERING_LAYER['position_after']), 1)
                current_oled_stack.insert(insert_idx, 
                   [SCATTERING_LAYER['name'], SCATTERING_LAYER['host_material'], SCATTERING_LAYER['thickness'], 'scattering']
                )

            # --- Stage 1: TMM Simulation ---
            print("Running TMM simulation for planar equivalent...")
            tmm_score = tmm_sim.run(params, material_handler, self.stack_base)
            print(f"  TMM Score (Avg. Transmission): {tmm_score:.4f}")

            # --- Stage 2: Meep Simulation (Conditional) ---
            meep_score = -1.0 # Default value if not run
            if params['np_concentration'] > 0 or tmm_score > self.sim_settings['tmm_threshold']:
                print("TMM score above threshold or scattering layer present. Running Meep simulation...")
                meep_score = meep_sim.run(params, material_handler, current_oled_stack)
                print(f"  Meep Score (Outcoupling Efficiency): {meep_score:.4f}")
            else:
                print("TMM score below threshold. Skipping Meep simulation.")

            # --- Record Results ---
            result_data = params.copy()
            result_data['tmm_avg_transmission'] = tmm_score
            result_data['meep_outcoupling_eff'] = meep_score
            
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
            
            # Save progress
            self.results_df.to_csv(self.sim_settings['results_filename'], index=False)
            
            run_time = time.time() - run_start_time
            print(f"Run {i+1} finished in {timedelta(seconds=run_time)}.")

        total_time = time.time() - start_time
        print(f"\n--- Optimization Finished ---")
        print(f"Total execution time: {timedelta(seconds=total_time)}")

    def analyze_results(self):
        """
        Analyzes the final results and prints the optimal configuration.
        """
        if self.results_df.empty:
            print("No results to analyze.")
            return

        print("\n--- Optimization Results Summary ---")
        print(self.results_df)

        # Find best result based on Meep score, fallback to TMM score
        if 'meep_outcoupling_eff' in self.results_df.columns and self.results_df['meep_outcoupling_eff'].max() > 0:
            best_run = self.results_df.loc[self.results_df['meep_outcoupling_eff'].idxmax()]
            print("\n--- Best Configuration (based on Meep Outcoupling Efficiency) ---")
        else:
            best_run = self.results_df.loc[self.results_df['tmm_avg_transmission'].idxmax()]
            print("\n--- Best Configuration (based on TMM Transmission) ---")
        
        print(best_run)
        print(f"\nResults saved to '{self.sim_settings['results_filename']}'")


# --- SCRIPT EXECUTION ---

if __name__ == "__main__":
    # Check for Meep installation
    try:
        mp.Simulation(cell_size=mp.Vector3(1,1,1), resolution=10)
    except Exception as e:
        print("Error: Meep does not seem to be installed correctly.")
        print("Please follow the installation instructions in the script header.")
        print(f"Details: {e}")
        exit()

    optimizer = Optimizer(PARAMETER_SPACE, OLED_STACK_BASE, SIMULATION_SETTINGS)
    optimizer.run_optimization()
    optimizer.analyze_results()
