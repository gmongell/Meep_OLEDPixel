#!/usr/bin/env python3

"""
OLED Outcoupling Efficiency Optimization Script (Parallel Version)

This script performs a comprehensive, multi-stage optimization of Organic Light-
Emitting Diode (OLED) outcoupling efficiency using a combination of the Transfer
Matrix Method (TMM) and parallelized Finite-Difference Time-Domain (FDTD) method.

This version is modified to divide the parameter sweep across multiple CPU cores
using MPI, significantly reducing the total runtime.

Author: Award-Winning Writer & Computational Physicist
Date: August 7, 2025

Dependencies:
- mpi4py
- meep (compiled with MPI support)
- tmm
- numpy
- pandas
- scipy
- requests
- PyYAML

Installation (Fedora Linux with MPI):
-------------------------------------
# 1. Install system dependencies for Meep
sudo dnf install -y git-core autoconf automake libtool libgsl-devel \
    libpng-devel libjpeg-devel swig hdf5-devel libgfortran-static \
    openmpi-devel lapack-devel blas-devel fftw-devel

# 2. Install Python dependencies
pip install --upgrade pip
pip install mpi4py numpy scipy pandas matplotlib tmm requests PyYAML

# 3. Install Meep from source with MPI support
# (Follow the official Meep installation guide or the GitHub discussion link)

-------------------------------------
"""

import meep as mp
import numpy as np
import pandas as pd
import tmm
import itertools
import random
import os
import time
from datetime import timedelta
import requests
import yaml

# --- SCRIPT CONFIGURATION ---

# Define the parameter space for the optimization sweep.
PARAMETER_SPACE = {
    'htl_thickness': [20, 30, 40],
    'etl_thickness': [20, 30, 40],
    'np_concentration': [0.0, 0.1, 0.2],
    'np_radius': [30, 40, 50],
}

# Define the base OLED device structure.
OLED_STACK_BASE = [
    ['ITO', 'ITO', 150, 'stack'],
    ['HIL', 'PEDOT:PSS', 40, 'stack'],
    ['HTL', 'NPB', 30, 'stack'],
    ['EML', 'Alq3', 20, 'stack'],
    ['ETL', 'Alq3', 30, 'stack'],
    ['EIL', 'LiF', 1, 'stack'],
]

# Define the scattering layer details.
SCATTERING_LAYER = {
    'name': 'ScatteringLayer',
    'host_material': 'SOG',
    'np_material': 'YSZ',
    'thickness': 2500,
    'position_after': 'ITO'
}

# Define material refractive index data sources.
# CORRECTED URLS to point to the stable GitHub database.
MATERIAL_URLS = {
    'Air': None,
    'Glass': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/glass/fused_silica/Malitson.yml',
    'ITO': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/other/In2O3-SnO2/Moerland.yml',
    'PEDOT:PSS': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/other/PEDOT-PSS/Chen.yml',
    'NPB': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/organic/NPB/Ghosh.yml',
    'Alq3': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/organic/Alq3/Ghosh.yml',
    'LiF': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/main/LiF/Li.yml',
    'Al': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/main/Al/Rakic.yml',
    'SOG': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/glass/fused_silica/Malitson.yml',
    'YSZ': 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/other/ZrO2-Y2O3/Wood.yml'
}


# General simulation settings.
SIMULATION_SETTINGS = {
    'wavelength_min_nm': 400,
    'wavelength_max_nm': 700,
    'wavelength_points': 31,
    'tmm_threshold': 0.20,
    'meep_resolution': 20,
    'results_filename': 'oled_optimization_results.csv'
}

# --- MODULE 1: MATERIAL DATA HANDLER (Robust Version) ---
class MaterialHandler:
    """
    Handles fetching and caching of material optical constants by directly
    parsing the YAML data files from the refractiveindex.info GitHub database.
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
                response = requests.get(url)
                response.raise_for_status()
                
                full_data = yaml.safe_load(response.text)
                data_lines = full_data['data'].strip().split('\n')
                
                yaml_wavelengths_um, yaml_n, yaml_k = [], [], []
                for line in data_lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        yaml_wavelengths_um.append(float(parts[0]))
                        yaml_n.append(float(parts[1]))
                        yaml_k.append(float(parts[2]) if len(parts) >= 3 else 0.0)
                
                yaml_wavelengths_nm = np.array(yaml_wavelengths_um) * 1000
                
                n_interp = np.interp(self.wavelengths_nm, yaml_wavelengths_nm, np.array(yaml_n))
                k_interp = np.interp(self.wavelengths_nm, yaml_wavelengths_nm, np.array(yaml_k))
                
                nk_array = n_interp + 1j * k_interp

            except Exception as e:
                print(f"    ERROR: Could not fetch or parse data for {material_key}. {e}")
                nk_array = 1.5 * np.ones_like(self.wavelengths_nm, dtype=complex)

        self._cache[material_key] = nk_array
        return nk_array

# --- MODULE 2: TMM SIMULATOR (PLANAR STACK) ---
class TMM_Simulator:
    def __init__(self, wavelengths_nm):
        self.wavelengths_nm = wavelengths_nm
    def run(self, params, material_handler, oled_stack):
        d_list = [np.inf]
        for layer in oled_stack:
            layer_name, _, default_thickness, _ = layer
            thickness = params.get(f'{layer_name.lower()}_thickness', default_thickness)
            d_list.append(thickness)
        d_list.append(np.inf)
        total_transmission = 0
        for i, lam_vac in enumerate(self.wavelengths_nm):
            n_list = [material_handler.get_nk_array('Glass')[i]]
            for layer in oled_stack:
                _, material_key, _, _ = layer
                n_list.append(material_handler.get_nk_array(material_key)[i])
            n_list.append(material_handler.get_nk_array('Al')[i])
            try:
                T = tmm.unpolarized_RT(n_list, d_list, 0, lam_vac)['T']
                total_transmission += T
            except Exception as e:
                if mp.am_master(): print(f"    TMM calculation failed for wavelength {lam_vac} nm: {e}")
                continue
        return total_transmission / len(self.wavelengths_nm) if self.wavelengths_nm.size > 0 else 0

# --- MODULE 3: MEEP SIMULATOR (SCATTERING STRUCTURES) ---
class Meep_Simulator:
    def __init__(self, wavelengths_nm):
        self.wavelengths_nm = wavelengths_nm
        self.freqs = 1 / (self.wavelengths_nm * 1e-3) if wavelengths_nm.size > 0 else np.array([])
        self.fcen = np.mean(self.freqs) if self.freqs.size > 0 else 0
        self.df = np.max(self.freqs) - np.min(self.freqs) if self.freqs.size > 0 else 0
    def run(self, params, material_handler, oled_stack):
        resolution = SIMULATION_SETTINGS['meep_resolution']
        stack_layers, current_z = [], 0
        t_glass = 1000
        for layer_info in oled_stack:
            name, material_key, default_thickness, _ = layer_info
            thickness = params.get(f'{name.lower()}_thickness', default_thickness) * 1e-3
            if name == 'EML':
                eml_pos_z, eml_thickness = current_z + thickness / 2, thickness
            mat_index = np.real(material_handler.get_nk_array(material_key)[len(self.wavelengths_nm)//2])
            stack_layers.append(mp.Block(material=mp.Medium(index=mat_index), size=mp.Vector3(mp.inf, mp.inf, thickness), center=mp.Vector3(0, 0, current_z + thickness / 2)))
            current_z += thickness
        t_stack, t_cathode = current_z, 100 * 1e-3
        sx, sy, dpml = 4, 4, 1.0
        sz = dpml + t_glass*1e-3 + t_stack + t_cathode + dpml
        cell_size = mp.Vector3(sx, sy, sz)
        z_shift = -sz/2 + dpml
        geometry = [mp.Block(material=mp.Medium(index=np.real(material_handler.get_nk_array('Glass')[len(self.wavelengths_nm)//2])), size=mp.Vector3(mp.inf, mp.inf, t_glass*1e-3), center=mp.Vector3(0, 0, z_shift + t_glass*1e-3/2))]
        for layer in stack_layers:
            layer.center.z += z_shift + t_glass*1e-3
            geometry.append(layer)
        geometry.append(mp.Block(material=mp.Medium(index=np.real(material_handler.get_nk_array('Al')[len(self.wavelengths_nm)//2])), size=mp.Vector3(mp.inf, mp.inf, t_cathode), center=mp.Vector3(0, 0, z_shift + t_glass*1e-3 + t_stack + t_cathode/2)))
        if params['np_concentration'] > 0:
            np_radius_um = params['np_radius'] * 1e-3
            host_mat_key, np_mat_key = SCATTERING_LAYER['host_material'], SCATTERING_LAYER['np_material']
            scat_layer_thickness_um = SCATTERING_LAYER['thickness'] * 1e-3
            z_pos_scat = sum(params.get(f'{l[0].lower()}_thickness', l[2]) * 1e-3 for l in oled_stack[:oled_stack.index(next(l for l in oled_stack if l[0] == SCATTERING_LAYER['position_after']))+1])
            z_center_scat = z_shift + t_glass*1e-3 + z_pos_scat - scat_layer_thickness_um / 2
            geometry.append(mp.Block(material=mp.Medium(index=np.real(material_handler.get_nk_array(host_mat_key)[len(self.wavelengths_nm)//2])), size=mp.Vector3(mp.inf, mp.inf, scat_layer_thickness_um), center=mp.Vector3(0, 0, z_center_scat)))
            np_material = mp.Medium(index=np.real(material_handler.get_nk_array(np_mat_key)[len(self.wavelengths_nm)//2]))
            num_nps = int(params['np_concentration'] * (sx * sy * scat_layer_thickness_um) / (4/3 * np.pi * np_radius_um**3))
            for _ in range(num_nps):
                x, y, z = random.uniform(-sx/2, sx/2), random.uniform(-sy/2, sy/2), random.uniform(z_center_scat - scat_layer_thickness_um/2, z_center_scat + scat_layer_thickness_um/2)
                geometry.append(mp.Sphere(radius=np_radius_um, center=mp.Vector3(x,y,z), material=np_material))
        src_pos_z = z_shift + t_glass*1e-3 + eml_pos_z
        sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.df), component=mp.Ez, center=mp.Vector3(0,0,src_pos_z))]
        box_z_half = eml_thickness / 2 + 0.01
        fr_source_box_z = mp.FluxRegion(center=mp.Vector3(0,0,src_pos_z)-mp.Vector3(0,0,box_z_half), size=mp.Vector3(sx-2*dpml, sy-2*dpml, 0))
        fr_source_box_z_rev = mp.FluxRegion(center=mp.Vector3(0,0,src_pos_z)+mp.Vector3(0,0,box_z_half), size=mp.Vector3(sx-2*dpml, sy-2*dpml, 0))
        fr_out = mp.FluxRegion(center=mp.Vector3(0, 0, z_shift + 0.1), size=mp.Vector3(sx-2*dpml, sy-2*dpml, 0))
        total_flux_out, total_flux_source = np.zeros(len(self.freqs)), np.zeros(len(self.freqs))
        for pol in [mp.Ex, mp.Ey, mp.Ez]:
            sim = mp.Simulation(cell_size=cell_size, geometry=geometry, sources=sources, boundary_layers=[mp.PML(dpml)], resolution=resolution)
            sim.sources[0].component = pol
            source_flux_z = sim.add_flux(self.fcen, self.df, len(self.freqs), fr_source_box_z)
            source_flux_z_rev = sim.add_flux(self.fcen, self.df, len(self.freqs), fr_source_box_z_rev)
            out_flux = sim.add_flux(self.fcen, self.df, len(self.freqs), fr_out)
            sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0,0,src_pos_z), 1e-8))
            total_flux_source += np.array(mp.get_fluxes(source_flux_z_rev)) - np.array(mp.get_fluxes(source_flux_z))
            total_flux_out += -np.array(mp.get_fluxes(out_flux))
            sim.reset_meep()
        avg_source_power, avg_out_power = np.trapz(total_flux_source, self.freqs), np.trapz(total_flux_out, self.freqs)
        return avg_out_power / avg_source_power if avg_source_power != 0 else 0.0

# --- MODULE 4: MAIN OPTIMIZATION ORCHESTRATOR (Modified for Parallelism) ---
class Optimizer:
    def __init__(self, param_space, stack_base, sim_settings, comm):
        self.param_space = param_space
        self.stack_base = stack_base
        self.sim_settings = sim_settings
        self.results_df = pd.DataFrame()
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

    def prepare_and_distribute_materials(self):
        """
        Handles fetching material data on the master process (rank 0) and
        distributing it to all other processes to avoid race conditions.
        """
        if self.rank == 0:
            print("Rank 0: Pre-fetching all required material data...")
            handler = MaterialHandler(MATERIAL_URLS)
            
            unique_materials = set(layer[1] for layer in self.stack_base)
            unique_materials.update({
                SCATTERING_LAYER['host_material'],
                SCATTERING_LAYER['np_material'],
                'Glass', 'Al'
            })
            
            for mat_key in unique_materials:
                handler.get_nk_array(mat_key)
            print("Rank 0: Material data fetching complete.")
        else:
            handler = None

        handler = self.comm.bcast(handler, root=0)
        return handler

    def run_optimization(self):
        start_time = time.time()
        
        material_handler = self.prepare_and_distribute_materials()
        
        keys, values = zip(*self.param_space.items())
        all_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        total_runs = len(all_param_combinations)

        my_param_combinations = all_param_combinations[self.rank::self.size]
        my_total_runs = len(my_param_combinations)

        if self.rank == 0:
            print(f"--- Starting Parallel OLED Optimization ---")
            print(f"Total parameter combinations: {total_runs}")
            print(f"Running on {self.size} parallel processes.")

        tmm_sim = TMM_Simulator(material_handler.wavelengths_nm)
        meep_sim = Meep_Simulator(material_handler.wavelengths_nm)

        for i, params in enumerate(my_param_combinations):
            run_start_time = time.time()
            global_index = all_param_combinations.index(params)
            print(f"[Rank {self.rank:02d}] Running my sim {i+1}/{my_total_runs} (Global index {global_index}) | Params: {params}")

            current_oled_stack = list(self.stack_base)
            if params.get('np_concentration', 0) > 0:
                insert_idx = next((j + 1 for j, layer in enumerate(current_oled_stack) if layer[0] == SCATTERING_LAYER['position_after']), 1)
                current_oled_stack.insert(insert_idx, [SCATTERING_LAYER['name'], SCATTERING_LAYER['host_material'], SCATTERING_LAYER['thickness'], 'scattering'])
            
            tmm_score = tmm_sim.run(params, material_handler, self.stack_base)
            meep_score = -1.0
            if params['np_concentration'] > 0 or tmm_score > self.sim_settings['tmm_threshold']:
                meep_score = meep_sim.run(params, material_handler, current_oled_stack)
            
            result_data = params.copy()
            result_data['tmm_avg_transmission'] = tmm_score
            result_data['meep_outcoupling_eff'] = meep_score
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
            
            run_time = time.time() - run_start_time
            print(f"[Rank {self.rank:02d}] Finished sim {i+1}/{my_total_runs} in {timedelta(seconds=run_time)}. TMM: {tmm_score:.4f}, Meep: {meep_score:.4f}")

        all_results_dfs = self.comm.gather(self.results_df, root=0)

        if self.rank == 0:
            final_df = pd.concat(all_results_dfs, ignore_index=True) if all_results_dfs else pd.DataFrame()
            final_df.to_csv(self.sim_settings['results_filename'], index=False)
            self.results_df = final_df
            
            total_time = time.time() - start_time
            print(f"\n--- Optimization Finished on Rank 0 ---")
            print(f"Total execution time: {timedelta(seconds=total_time)}")

    def analyze_results(self):
        if self.rank == 0:
            if self.results_df.empty:
                print("No results to analyze.")
                return
            print("\n--- Optimization Results Summary ---")
            print(self.results_df.to_string())
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
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        try:
            mp.Simulation(cell_size=mp.Vector3(1,1,1), resolution=10)
        except Exception as e:
            print("Error: Meep does not seem to be installed correctly.")
            print(f"Details: {e}")
            comm.Abort()

    optimizer = Optimizer(PARAMETER_SPACE, OLED_STACK_BASE, SIMULATION_SETTINGS, comm)
    optimizer.run_optimization()
    optimizer.analyze_results()
