#!/usr/bin/env python3
"""
Test Set Energy-Volume Comparison Plot Script

This script runs LAMMPS workflow on test.xyz and generates energy-volume comparison plots
similar to plot_energy_volume_comparison.py, but for test dataset predictions.

Key Features:
- Automatic energy baseline alignment for tabGAP (NEP → TabGAP zero-point correction)
- Energy-volume curve comparison
- Scatter plot with statistics (MAE, RMSE, R²)

Usage:
    uv run python scripts/plot_test_energy_volume.py -f <forcefield> [-n <name>] [options]

Examples:
    # Use NEP potential (no alignment needed)
    uv run python scripts/plot_test_energy_volume.py -f forcefield/nep/3.3.0.txt
    
    # Use tabGAP potential (automatic energy alignment applied)
    uv run python scripts/plot_test_energy_volume.py -f forcefield/tabgap

    # Choose 2022 LAMMPS version
    uv run python scripts/plot_test_energy_volume.py -f forcefield/nep/3.3.0.txt --lammps-version 2022
    
    # Custom test name
    uv run python scripts/plot_test_energy_volume.py -f forcefield/nep/3.3.0.txt -n my_test
    
    # Skip LAMMPS run (use existing results)
    uv run python scripts/plot_test_energy_volume.py -f forcefield/nep/3.3.0.txt --skip-run
"""

import argparse
import os
import sys
import subprocess
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict


class EnergyAligner:
    """Energy baseline alignment tool for TabGAP"""
    def __init__(self):
        # TabGAP zero-point energies (eV)
        self.tabgap_zpe = {
            'Ga': -0.0244486,
            'O': -0.0350174
        }
        
        # NEP zero-point energies (eV)
        self.nep_zpe = {
            'Ga': -1.68768,
            'O': -3.19589
        }
    
    def count_atoms_from_lines(self, lines, start_idx) -> Dict[str, int]:
        """Count atoms from xyz structure lines"""
        try:
            num_atoms = int(lines[start_idx].strip())
            atom_counts = {'Ga': 0, 'O': 0}
            
            for i in range(start_idx + 2, start_idx + 2 + num_atoms):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) >= 1:
                    element = parts[0]
                    if element in atom_counts:
                        atom_counts[element] += 1
            
            return atom_counts
        except Exception:
            return {'Ga': 0, 'O': 0}
    
    def calculate_offset_per_atom(self, atom_counts: Dict[str, int]) -> float:
        """
        Calculate per-atom energy offset from NEP baseline to TabGAP baseline
        
        Args:
            atom_counts: Atom counts {'Ga': n_Ga, 'O': n_O}
        
        Returns:
            per-atom energy offset (eV/atom)
        """
        n_total = atom_counts['Ga'] + atom_counts['O']
        if n_total == 0:
            return 0.0
        
        # Calculate total offset (from NEP to TabGAP)
        offset_total = (
            atom_counts['Ga'] * (self.tabgap_zpe['Ga'] - self.nep_zpe['Ga']) +
            atom_counts['O'] * (self.tabgap_zpe['O'] - self.nep_zpe['O'])
        )
        
        # Convert to per-atom offset
        offset_per_atom = offset_total / n_total
        
        return offset_per_atom


def parse_lattice_string(lattice_str):
    """Parse Lattice string and return 3x3 matrix"""
    match = re.search(r'"([^"]*)"', lattice_str)
    if not match:
        raise ValueError(f"Cannot parse Lattice string: {lattice_str}")
    
    values = list(map(float, match.group(1).split()))
    if len(values) != 9:
        raise ValueError(f"Wrong number of Lattice parameters: {len(values)}, expected 9")
    
    lattice = np.array(values).reshape(3, 3)
    return lattice


def calculate_volume(lattice_matrix):
    """Calculate cell volume"""
    return abs(np.linalg.det(lattice_matrix))


def parse_xyz_structure(lines, start_idx, count_atoms=False, aligner=None):
    """Parse a single XYZ structure"""
    if start_idx >= len(lines):
        return None, start_idx
    
    try:
        # Read number of atoms
        num_atoms = int(lines[start_idx].strip())
        
        # Read properties line
        properties_line = lines[start_idx + 1].strip()
        
        # Extract config_type
        config_type_match = re.search(r'Config_type=(\w+)', properties_line)
        config_type = config_type_match.group(1) if config_type_match else "unknown"
        
        # Extract energy (DFT reference)
        energy_match = re.search(r'Energy=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', properties_line)
        dft_energy = float(energy_match.group(1)) if energy_match else 0.0
        
        # Parse Lattice
        lattice_matrix = parse_lattice_string(properties_line)
        volume = calculate_volume(lattice_matrix)
        # Calculate volume per atom (Å³/atom)
        volume_per_atom = volume / num_atoms
        
        structure = {
            'num_atoms': num_atoms,
            'config_type': config_type,
            'dft_energy': dft_energy,
            'dft_energy_per_atom': dft_energy / num_atoms,
            'volume': volume,
            'volume_per_atom': volume_per_atom,
        }
        
        # Count atoms and calculate alignment offset if requested
        if count_atoms and aligner:
            atom_counts = aligner.count_atoms_from_lines(lines, start_idx)
            structure['atom_counts'] = atom_counts
            structure['alignment_offset'] = aligner.calculate_offset_per_atom(atom_counts)
        
        return structure, start_idx + 2 + num_atoms
        
    except Exception as e:
        print(f"Error parsing structure at line {start_idx}: {e}")
        return None, start_idx + 1


def read_test_xyz(filename, align_for_tabgap=False):
    """
    Read test.xyz file and parse all structures
    
    Args:
        filename: Path to xyz file
        align_for_tabgap: If True, align DFT energies from NEP baseline to TabGAP baseline
                         Also count atoms for potential later use
    """
    print(f"Reading test set file: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    structures = []
    idx = 0
    
    # Initialize aligner if needed
    aligner = EnergyAligner() if align_for_tabgap else None
    
    if align_for_tabgap:
        print("  Energy baseline alignment: NEP → TabGAP")
        print(f"  NEP zero-point: Ga={aligner.nep_zpe['Ga']:.6f} eV, O={aligner.nep_zpe['O']:.6f} eV")
        print(f"  TabGAP zero-point: Ga={aligner.tabgap_zpe['Ga']:.6f} eV, O={aligner.tabgap_zpe['O']:.6f} eV")
    
    while idx < len(lines):
        # Always count atoms when align_for_tabgap is True (needed for both DFT and NEP alignment)
        structure, idx = parse_xyz_structure(lines, idx, count_atoms=align_for_tabgap, aligner=aligner)
        if structure:
            # Apply alignment offset if needed
            if align_for_tabgap and 'alignment_offset' in structure:
                offset = structure['alignment_offset']
                structure['dft_energy_per_atom'] += offset
                structure['dft_energy'] = structure['dft_energy_per_atom'] * structure['num_atoms']
            structures.append(structure)
    
    print(f"Successfully parsed {len(structures)} structures")
    
    if align_for_tabgap and structures:
        # Show alignment example
        example = structures[0]
        if 'alignment_offset' in example:
            print(f"\n  Alignment example (structure 0):")
            print(f"    Atom counts: Ga={example['atom_counts']['Ga']}, O={example['atom_counts']['O']}")
            print(f"    Offset: {example['alignment_offset']:.10f} eV/atom")
    
    return structures


def read_xyz_frames(xyz_file):
    """Read xyz file and return all frames"""
    frames = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            break
        
        frame_lines = lines[i:i + n_atoms + 2]
        
        if len(frame_lines) < n_atoms + 2:
            break
        
        frames.append(frame_lines)
        i += n_atoms + 2
    
    return frames


def save_frames_to_folders(frames, output_dir, start_index=0):
    """Save frames to numbered subdirectories"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, frame in enumerate(frames):
        frame_num = start_index + idx
        frame_dir = output_path / f"{frame_num:06d}"
        frame_dir.mkdir(exist_ok=True)
        
        output_file = frame_dir / "structure.xyz"
        with open(output_file, 'w') as f:
            f.writelines(frame)


class XyzToLammpsConverter:
    """Converter from GPUMD xyz format to LAMMPS data file"""
    
    def __init__(self):
        self.element_to_mass = {
            'O': 15.9994,
            'Ga': 69.723,
        }
    
    def parse_xyz_file(self, filename: str):
        """Parse GPUMD xyz file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            raise ValueError(f"File {filename} has incorrect format")
        
        n_atoms = int(lines[0].strip())
        header_line = lines[1].strip()
        
        lattice_match = re.search(r'Lattice="([^"]*)"', header_line)
        if not lattice_match:
            raise ValueError("Lattice information not found")
        
        lattice_str = lattice_match.group(1)
        lattice_values = [float(x) for x in lattice_str.split()]
        
        if len(lattice_values) != 9:
            raise ValueError(f"Lattice parameters should be 9 values")
        
        lattice_vectors = np.array(lattice_values).reshape(3, 3)
        
        atoms = []
        for i in range(2, 2 + n_atoms):
            if i >= len(lines):
                raise ValueError(f"Insufficient atom data in file")
                
            parts = lines[i].strip().split()
            if len(parts) < 4:
                raise ValueError(f"Line {i+1} has incomplete data")
            
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            atoms.append({'element': element, 'x': x, 'y': y, 'z': z})
        
        return {
            'n_atoms': n_atoms,
            'lattice_vectors': lattice_vectors,
            'atoms': atoms,
        }
    
    def get_lammps_box_and_transform(self, lattice_vectors: np.ndarray):
        """Calculate LAMMPS box parameters and coordinate transformation matrix"""
        a = lattice_vectors[0]
        b = lattice_vectors[1]
        c = lattice_vectors[2]
        
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        lc = np.linalg.norm(c)
        
        cos_alpha = np.dot(b, c) / (lb * lc)
        cos_beta = np.dot(a, c) / (la * lc)
        cos_gamma = np.dot(a, b) / (la * lb)
        
        lx = la
        xy = lb * cos_gamma
        ly = np.sqrt(lb**2 - xy**2)
        xz = lc * cos_beta
        yz = (lb * lc * cos_alpha - xy * xz) / ly
        lz = np.sqrt(lc**2 - xz**2 - yz**2)

        eps = 1e-8
        if abs(xy) >= 0.5 * lx:
            xy = np.sign(xy) * (0.5 * lx - eps)
        if abs(xz) >= 0.5 * lx:
            xz = np.sign(xz) * (0.5 * lx - eps)
        if abs(yz) >= 0.5 * ly:
            yz = np.sign(yz) * (0.5 * ly - eps)
        
        is_triclinic = (abs(xy) > 1e-6 or abs(xz) > 1e-6 or abs(yz) > 1e-6)
        
        box_params = {
            'xlo': 0.0, 'xhi': lx,
            'ylo': 0.0, 'yhi': ly,
            'zlo': 0.0, 'zhi': lz,
            'xy': xy, 'xz': xz, 'yz': yz,
            'is_triclinic': is_triclinic
        }
        
        new_lattice = np.array([
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz]
        ])
        
        transform_matrix = np.linalg.inv(lattice_vectors) @ new_lattice
        
        return box_params, transform_matrix
    
    def convert_to_lammps(self, xyz_file: str, output_file: str):
        """Convert xyz file to LAMMPS data format"""
        data = self.parse_xyz_file(xyz_file)
        box_params, transform_matrix = self.get_lammps_box_and_transform(data['lattice_vectors'])
        
        # Fixed atom type mapping: O=1, Ga=2
        atom_type_map = {'O': 1, 'Ga': 2}
        
        comment = f"LAMMPS data file converted from {Path(xyz_file).name}"
        
        with open(output_file, 'w') as f:
            f.write(f"# {comment}\n")
            f.write(f"{data['n_atoms']} atoms\n")
            f.write(f"2 atom types\n")
            
            f.write(f"{box_params['xlo']:.12f} {box_params['xhi']:.12f} xlo xhi\n")
            f.write(f"{box_params['ylo']:.12f} {box_params['yhi']:.12f} ylo yhi\n")
            f.write(f"{box_params['zlo']:.12f} {box_params['zhi']:.12f} zlo zhi\n")
            
            if box_params['is_triclinic']:
                f.write(f"{box_params['xy']:.12f} {box_params['xz']:.12f} {box_params['yz']:.12f} xy xz yz\n")
            
            f.write("\n")
            f.write("Masses\n\n")
            f.write(f"1 {self.element_to_mass['O']:.4f}  # O\n")
            f.write(f"2 {self.element_to_mass['Ga']:.4f}  # Ga\n")
            f.write("\n")
            f.write("Atoms  # atomic\n\n")
            
            for i, atom in enumerate(data['atoms']):
                atom_id = i + 1
                atom_type = atom_type_map[atom['element']]
                r_old = np.array([atom['x'], atom['y'], atom['z']])
                r_new = r_old @ transform_matrix
                x, y, z = r_new[0], r_new[1], r_new[2]
                f.write(f"{atom_id} {atom_type} {x:.12f} {y:.12f} {z:.12f}\n")
        
        return output_file


def convert_structures_to_lammps(root_dir: str):
    """Convert all structure.xyz in subdirectories to model.data"""
    root_path = Path(root_dir)
    converter = XyzToLammpsConverter()
    
    subdirs = [d for d in root_path.iterdir() if d.is_dir() and (d / "structure.xyz").exists()]
    
    success_count = 0
    fail_count = 0
    
    for subdir in sorted(subdirs):
        input_file = subdir / "structure.xyz"
        output_file = subdir / "model.data"
        
        try:
            converter.convert_to_lammps(str(input_file), str(output_file))
            success_count += 1
        except Exception as e:
            print(f"  ✗ {subdir.name}: Conversion failed - {str(e)}")
            fail_count += 1
    
    return success_count, fail_count


def create_symlinks_forcefield(forcefield_dir: str, target_root_dir: str, potential_type: str = "tabgap"):
    """Create symlinks for forcefield files in all subdirectories"""
    forcefield_path = Path(forcefield_dir).resolve()
    target_root_path = Path(target_root_dir).resolve()
    
    if forcefield_path.is_file():
        forcefield_files = [forcefield_path]
    elif forcefield_path.is_dir():
        forcefield_files = [f for f in forcefield_path.iterdir() if f.is_file()]
    else:
        print(f"  Warning: Forcefield path {forcefield_dir} does not exist")
        return 0
    
    subdirs = [d for d in target_root_path.iterdir() if d.is_dir()]
    
    success_count = 0
    
    for subdir in sorted(subdirs):
        for ff_file in forcefield_files:
            if potential_type == "nep":
                link_name = "nep.txt"
            else:
                link_name = ff_file.name
            
            link_path = subdir / link_name
            
            if link_path.exists() or link_path.is_symlink():
                continue
            
            try:
                rel_path = os.path.relpath(ff_file, subdir)
                os.symlink(rel_path, link_path)
                success_count += 1
            except Exception:
                pass
    
    return success_count


def create_symlinks_run_script(source_file: str, target_root_dir: str):
    """Create symlinks for run.in in all subdirectories"""
    source_path = Path(source_file).resolve()
    target_root_path = Path(target_root_dir).resolve()
    
    subdirs = [d for d in target_root_path.iterdir() if d.is_dir()]
    
    success_count = 0
    
    for subdir in sorted(subdirs):
        link_path = subdir / "run.in"
        
        if link_path.exists() or link_path.is_symlink():
            continue
        
        try:
            rel_path = os.path.relpath(source_path, subdir)
            os.symlink(rel_path, link_path)
            success_count += 1
        except Exception:
            pass
    
    return success_count


def run_lammps_in_directory(lammps_exe: Path, work_dir: Path):
    """Run LAMMPS in the specified directory"""
    input_path = work_dir / "run.in"
    
    if not input_path.exists():
        return False, "run.in does not exist"
    
    cmd = [str(lammps_exe), "-in", "run.in", "-log", "lammps.log"]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, f"Failed (return code {result.returncode})"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def run_lammps_wrapper(args_tuple):
    """Wrapper function for parallel execution"""
    lammps_exe, work_dir, index, total = args_tuple
    dir_name = work_dir.name
    success, message = run_lammps_in_directory(lammps_exe, work_dir)
    return index, dir_name, success, message


def extract_potential_energy_from_log(log_file: Path):
    """Extract potential energy from LAMMPS log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for the thermo output section
        in_thermo = False
        pe_col_index = None
        
        for line in lines:
            line = line.strip()
            
            # Find thermo header
            if line.startswith('Step') and 'PotEng' in line:
                in_thermo = True
                headers = line.split()
                try:
                    pe_col_index = headers.index('PotEng')
                except ValueError:
                    continue
                continue
            
            # End of thermo block
            if in_thermo and (line.startswith('Loop') or line.startswith('---')):
                break
            
            # Extract energy from last thermo line
            if in_thermo and pe_col_index is not None:
                parts = line.split()
                if len(parts) > pe_col_index:
                    try:
                        # This is the final energy value
                        energy = float(parts[pe_col_index])
                        return energy
                    except ValueError:
                        continue
        
        return None
        
    except Exception as e:
        print(f"  Warning: Failed to extract energy from {log_file}: {e}")
        return None


def collect_predicted_energies(raw_data_dir: Path):
    """Collect predicted energies from LAMMPS output"""
    print(f"Collecting predicted energies from LAMMPS output...")
    
    subdirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])
    
    predicted_energies = []
    
    for subdir in subdirs:
        log_file = subdir / "lammps.log"
        
        if not log_file.exists():
            print(f"  Warning: {subdir.name} - lammps.log not found")
            predicted_energies.append(None)
            continue
        
        energy = extract_potential_energy_from_log(log_file)
        predicted_energies.append(energy)
    
    valid_count = sum(1 for e in predicted_energies if e is not None)
    print(f"  Successfully collected {valid_count}/{len(predicted_energies)} energy values")
    
    return predicted_energies


def combine_data(structures, predicted_energies):
    """Combine structure information and predicted energies"""
    if len(structures) != len(predicted_energies):
        print(f"Warning: Data length mismatch!")
        print(f"  Structures: {len(structures)}")
        print(f"  Predicted energies: {len(predicted_energies)}")
        
        min_len = min(len(structures), len(predicted_energies))
        structures = structures[:min_len]
        predicted_energies = predicted_energies[:min_len]
        print(f"  Using first {min_len} data points")
    
    combined_data = []
    for i, struct in enumerate(structures):
        if predicted_energies[i] is None:
            continue
        
        data_point = {
            'config_type': struct['config_type'],
            'volume_per_atom': struct['volume_per_atom'],
            'predicted_energy_per_atom': predicted_energies[i] / struct['num_atoms'],
            'dft_energy_per_atom': struct['dft_energy_per_atom']
        }
        combined_data.append(data_point)
    
    print(f"  Combined {len(combined_data)} valid data points")
    return combined_data


def plot_energy_volume_comparison(data, output_filename, model_name="Model"):
    """Plot Model vs DFT energy-volume comparison"""
    from matplotlib.gridspec import GridSpec
    
    # Group data by config_type
    data_by_type = defaultdict(list)
    
    for point in data:
        config_type = point['config_type']
        volume_per_atom = point['volume_per_atom']
        predicted_energy = point['predicted_energy_per_atom']
        dft_energy = point['dft_energy_per_atom']
        
        data_by_type[config_type].append((volume_per_atom, predicted_energy, dft_energy))
    
    # Set up figure using GridSpec for ultra-fine control
    figsize = 10
    fontsize = 10
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Use ultra-fine grid division
    n = 100
    x0 = 10 * n
    y0 = 8 * n
    
    # Define margins and total size
    # Add some margins around the plot
    margin_x = int(1.5 * n)
    margin_y = int(1.5 * n)
    
    M = x0 + 2 * margin_x
    N = y0 + 2 * margin_y
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=(figsize, N/(M/figsize)))
    gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))
    
    # Create subplot centered in the grid
    ax1 = fig.add_subplot(gs[margin_y:margin_y+y0, margin_x:margin_x+x0])
    
    # Define colors and marker styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Mapping table for config types
    mapping_table = {'bulk_beta_phase': r'$\beta$ phase',
                'bulk_gamma_phase': r'$\gamma$ phase',
                'bulk_alpha_phase': r'$\alpha$ phase',
                'bulk_delta_phase': r'$\delta$ phase',
                'bulk_epsilon_phase': r'$\epsilon$ phase',
                'bulk_kappa_phase': r'$\kappa$ phase',
                'bulk_bixbyite_phase': r'$\text{hex}^{*}$ phase',
                'bulk_Pmc21_phase': r'$Pmc2_{1}$ phase',
                'bulk_P-1_phase': r'$P\overline{1}$ phase',
                'non_stoichiometry_GaO': r'GaO',
                'non_stoichiometry_GaO2': r'GaO$_2$',
                'non_stoichiometry_GaO3': r'GaO$_3$',
                'non_stoichiometry_Ga3O5': r'Ga$_3$O$_5$',
                'non_stoichiometry_Ga4O5': r'Ga$_4$O$_5$',
                'twobody': r'dimer Ga-Ga/Ga-O/O-O',
                'Ga_bulk': r'pure Ga',
                'Otrimer': r'trimer O$_3$',
                'RSS': r'random structure search',
                'active_training': r'O clusters',
                'melted_phase': r'melted phase',
                'isolated_atom': r'isolated Ga/O atoms',
                'close_3b_phase': r'close-3b phase',
                'amorphous_phase': r'amorphous phase',
                }

    # Calculate statistics for all data
    all_predicted = []
    all_dft = []
    
    for config_type, data_points in sorted(data_by_type.items()):
        if not data_points:
            continue
        
        predicted_energies = [point[1] for point in data_points]
        dft_energies = [point[2] for point in data_points]
        
        all_predicted.extend(predicted_energies)
        all_dft.extend(dft_energies)
    
    # Plot: Model vs DFT energy-volume curves
    for i, (config_type, data_points) in enumerate(sorted(data_by_type.items())):
        if not data_points:
            continue
        
        # Sort by volume
        data_points.sort(key=lambda x: x[0])
        
        volumes = [point[0] for point in data_points]
        predicted_energies = [point[1] for point in data_points]
        dft_energies = [point[2] for point in data_points]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Get display name from mapping table
        display_name = mapping_table.get(config_type, config_type)
        
        # Model data points and lines
        ax1.scatter(volumes, predicted_energies, 
                   color=color, marker=marker, s=60, alpha=0.8,
                   label=f'{display_name} {model_name} ({len(data_points)})')
        if len(data_points) > 1:
            ax1.plot(volumes, predicted_energies, color=color, alpha=0.6, linewidth=1.5, linestyle='-')
        
        # DFT data points and lines
        ax1.scatter(volumes, dft_energies, 
                   color=color, marker=marker, s=60, alpha=0.5, facecolors='none', edgecolors=color,
                   label=f'{display_name} DFT ({len(data_points)})')
        if len(data_points) > 1:
            ax1.plot(volumes, dft_energies, color=color, alpha=0.4, linewidth=1.5, linestyle='--')
    
    # Add statistics text box
    if len(all_dft) > 0 and len(all_predicted) > 0:
        all_predicted = np.array(all_predicted)
        all_dft = np.array(all_dft)
        
        mae = np.mean(np.abs(all_predicted - all_dft))
        rmse = np.sqrt(np.mean((all_predicted - all_dft)**2))
        r2 = np.corrcoef(all_predicted, all_dft)[0, 1]**2
        
        # Display statistics on plot
        stats_text = f'MAE: {mae:.4f} eV/atom\nRMSE: {rmse:.4f} eV/atom\nR²: {r2:.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=fontsize)
    
    ax1.set_xlabel('Volume per atom (Å³/atom)', fontsize=fontsize, fontweight='bold')
    ax1.set_ylabel('Energy per atom (eV/atom)', fontsize=fontsize, fontweight='bold')
    ax1.set_title(f'{model_name} vs DFT: Energy-Volume Curves', fontsize=fontsize+5, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=fontsize)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize)
    
    # Adjust layout - no longer needed with GridSpec as we controlled margins manually
    # but bbox_inches='tight' in savefig will help trim extra white space

    
    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_filename}")
    
    # Display statistics
    print("\nData Statistics:")
    print("=" * 80)
    print(f"{'Config Type':<20} {'Count':<8} {'Volume Range (Å³/atom)':<25} {'Energy Range (eV/atom)'}")
    print("=" * 80)
    
    for config_type, data_points in sorted(data_by_type.items()):
        if not data_points:
            continue
        
        volumes = [point[0] for point in data_points]
        dft_energies = [point[2] for point in data_points]
        
        vol_min, vol_max = min(volumes), max(volumes)
        energy_min, energy_max = min(dft_energies), max(dft_energies)
        
        print(f"{config_type:<20} {len(data_points):<8} "
              f"{vol_min:.2f} - {vol_max:.2f}        "
              f"{energy_min:.3f} - {energy_max:.3f}")
    
    if len(all_dft) > 0 and len(all_predicted) > 0:
        all_predicted_np = np.array(all_predicted)
        all_dft_np = np.array(all_dft)
        
        mae = np.mean(np.abs(all_predicted_np - all_dft_np))
        rmse = np.sqrt(np.mean((all_predicted_np - all_dft_np)**2))
        r2 = np.corrcoef(all_predicted_np, all_dft_np)[0, 1]**2
        
        print(f"\nOverall Prediction Accuracy:")
        print(f"  MAE:  {mae:.4f} eV/atom")
        print(f"  RMSE: {rmse:.4f} eV/atom") 
        print(f"  R²:   {r2:.4f}")
    
    return plt


def plot_multi_model_comparison(multi_model_data, output_filename, model_names=None):
    """Plot multiple models vs DFT energy-volume comparison
    
    Args:
        multi_model_data: List of data lists, each for one model
        output_filename: Output file path
        model_names: List of model names (optional)
    """
    from matplotlib.gridspec import GridSpec

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(multi_model_data))]
    
    # Group data by config_type for each model
    models_by_config = {}  # {config_type: {vol_key: {'volume': ..., 'dft': ..., 'models': {model_idx: energy}}}}
    
    # First pass: collect all config types and organize data
    for model_idx, data in enumerate(multi_model_data):
        for point in data:
            config_type = point['config_type']
            volume_per_atom = point['volume_per_atom']
            predicted_energy = point['predicted_energy_per_atom']
            dft_energy = point['dft_energy_per_atom']
            
            if config_type not in models_by_config:
                models_by_config[config_type] = {}
            
            # Use volume as key to match across models
            vol_key = round(volume_per_atom, 6)
            if vol_key not in models_by_config[config_type]:
                models_by_config[config_type][vol_key] = {
                    'volume': volume_per_atom,
                    'dft': dft_energy,
                    'models': {}
                }
            
            models_by_config[config_type][vol_key]['models'][model_idx] = predicted_energy
    
    # Set up figure using GridSpec
    figsize = 10
    fontsize = 14
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Use ultra-fine grid division
    n = 100
    x0 = 10 * n
    y0 = 7 * n
    
    # Define total size
   
    M = x0 
    N = y0
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=(figsize, N/(M/figsize)))
    gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))
    
    # Create subplot
    ax = fig.add_subplot(gs[0:y0, 0:x0])
    
    # Define colors for config types (consistent across all models)
    config_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Define markers for different models (same for all config types)
    model_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Define line styles for different models
    model_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # Mapping table for config types
    mapping_table = {'bulk_beta_phase': r'$\beta$ phase',
                'bulk_gamma_phase': r'$\gamma$ phase',
                'bulk_alpha_phase': r'$\alpha$ phase',
                'bulk_delta_phase': r'$\delta$ phase',
                'bulk_epsilon_phase': r'$\epsilon$ phase',
                'bulk_kappa_phase': r'$\kappa$ phase',
                'bulk_bixbyite_phase': r'$\text{hex}^{*}$ phase',
                'bulk_Pmc21_phase': r'$Pmc2_{1}$ phase',
                'bulk_P-1_phase': r'$P\overline{1}$ phase',
                'non_stoichiometry_GaO': r'GaO',
                'non_stoichiometry_GaO2': r'GaO$_2$',
                'non_stoichiometry_GaO3': r'GaO$_3$',
                'non_stoichiometry_Ga3O5': r'Ga$_3$O$_5$',
                'non_stoichiometry_Ga4O5': r'Ga$_4$O$_5$',
                'twobody': r'dimer Ga-Ga/Ga-O/O-O',
                'Ga_bulk': r'pure Ga',
                'Otrimer': r'trimer O$_3$',
                'RSS': r'random structure search',
                'active_training': r'O clusters',
                'melted_phase': r'melted phase',
                'isolated_atom': r'isolated Ga/O atoms',
                'close_3b_phase': r'close-3b phase',
                'amorphous_phase': r'amorphous phase',
                }
    
    # Statistics collection
    all_stats = {i: {'predicted': [], 'dft': []} for i in range(len(multi_model_data))}
    
    # Plot data by config type
    for config_idx, (config_type, vol_dict) in enumerate(sorted(models_by_config.items())):
        color = config_colors[config_idx % len(config_colors)]
        
        # Get display name
        display_name = mapping_table.get(config_type, config_type)
        
        # Sort by volume
        sorted_data = sorted(vol_dict.items(), key=lambda x: x[1]['volume'])
        
        volumes = [item[1]['volume'] for item in sorted_data]
        dft_energies = [item[1]['dft'] for item in sorted_data]
        
        # Plot DFT reference (hollow circles, dotted line)
        ax.scatter(volumes, dft_energies, 
                  color=color, marker='o', s=80, alpha=0.6, 
                  facecolors='none', edgecolors=color, linewidths=2,
                  label=f'{display_name} DFT', zorder=10)
        if len(volumes) > 1:
            ax.plot(volumes, dft_energies, color=color, alpha=0.3, 
                   linewidth=2, linestyle=':', zorder=5)
        
        # Plot each model's predictions
        for model_idx in range(len(multi_model_data)):
            model_energies = []
            model_volumes = []
            
            for item in sorted_data:
                if model_idx in item[1]['models']:
                    model_energies.append(item[1]['models'][model_idx])
                    model_volumes.append(item[1]['volume'])
                    
                    # Collect for statistics
                    all_stats[model_idx]['predicted'].append(item[1]['models'][model_idx])
                    all_stats[model_idx]['dft'].append(item[1]['dft'])
            
            if not model_energies:
                continue
            
            marker = model_markers[model_idx % len(model_markers)]
            linestyle = model_linestyles[model_idx % len(model_linestyles)]
            
            # Add label for each config_type and model combination
            label = f'{display_name} {model_names[model_idx]}'
            
            ax.scatter(model_volumes, model_energies, 
                      color=color, marker=marker, s=60, alpha=0.8,
                      label=label, zorder=8)
            if len(model_volumes) > 1:
                ax.plot(model_volumes, model_energies, color=color, 
                       alpha=0.7, linewidth=1.5, linestyle=linestyle, zorder=7)
    
    ax.set_xlabel('Volume per atom (Å³/atom)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('Energy per atom (eV/atom)', fontsize=fontsize, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize-6)

    # plt.tight_layout()

    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"\nMulti-model comparison plot saved to: {output_filename}")
    
    # Print detailed statistics
    print("\n" + "=" * 100)
    print("Multi-Model Prediction Accuracy Comparison")
    print("=" * 100)
    print(f"{'Model':<20} {'MAE (eV/atom)':<20} {'RMSE (eV/atom)':<20} {'R²':<15} {'N_points':<10}")
    print("=" * 100)
    
    for model_idx, model_name in enumerate(model_names):
        if len(all_stats[model_idx]['predicted']) > 0:
            pred = np.array(all_stats[model_idx]['predicted'])
            dft = np.array(all_stats[model_idx]['dft'])
            
            mae = np.mean(np.abs(pred - dft))
            rmse = np.sqrt(np.mean((pred - dft)**2))
            r2 = np.corrcoef(pred, dft)[0, 1]**2
            n_points = len(pred)
            
            print(f"{model_name:<20} {mae:<20.6f} {rmse:<20.6f} {r2:<15.6f} {n_points:<10}")
    
    print("=" * 100)
    
    return plt


def generate_test_name(forcefield: str) -> str:
    """Generate test name based on forcefield"""
    ff_path = Path(forcefield)
    if ff_path.is_file() and ff_path.suffix == '.txt':
        ff_name = ff_path.stem
    elif ff_path.is_dir() or ff_path.name == 'tabgap':
        ff_name = 'tabgap'
    else:
        ff_name = ff_path.stem if ff_path.is_file() else ff_path.name
    
    return f"{ff_name}_test"


def main():
    parser = argparse.ArgumentParser(
        description="Test Set Energy-Volume Comparison Plot (Single or Multi-Model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model: Use NEP potential (auto-generate name: 3.3.0_test)
  uv run python draw_pics/plot_test_energy_volume.py -f forcefield/nep/3.3.0.txt
  
  # Single model: Use tabGAP potential (auto-generate name: tabgap_test)
  uv run python draw_pics/plot_test_energy_volume.py -f forcefield/tabgap
  
  # Multi-model comparison: Compare 3 models
  uv run python draw_pics/plot_test_energy_volume.py \\
      --multi-model \\
      -f forcefield/nep/3.3.0.txt forcefield/nep/4.5.0.txt forcefield/tabgap \\
      -n 3.3.0 4.5.0 tabGAP \\
      -o multi_model_comparison
  
  # Skip LAMMPS run (use existing results)
  uv run python draw_pics/plot_test_energy_volume.py -f forcefield/nep/3.3.0.txt --skip-run
  
  # Limit to first 10 structures (testing)
  uv run python draw_pics/plot_test_energy_volume.py -f forcefield/nep/3.3.0.txt --max-jobs 10
        """
    )
    
    parser.add_argument(
        "-f", "--forcefield",
        type=str,
        nargs='+',
        required=True,
        help="Forcefield file(s) or directory (.txt=NEP, dir=tabGAP). Multiple values for multi-model mode."
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        nargs='*',
        default=None,
        help="Model name(s) for display. For multi-model: provide name for each model."
    )
    parser.add_argument(
        "-o", "--output-name",
        type=str,
        default=None,
        help="Output name for multi-model comparison (e.g., 'multi_comparison')"
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Enable multi-model comparison mode"
    )
    parser.add_argument(
        "--test-xyz",
        type=str,
        default="/home/abel/workspace/Ga2O3_2026_revision/model/test.xyz",
        help="Test xyz file path (default: train_dataset/nep_baseline/test.xyz)"
    )
    parser.add_argument(
        "--run-script",
        type=str,
        default=None,
        help="run.in script path (default: auto-select based on potential type)"
    )
    parser.add_argument(
        "--lammps",
        type=str,
        default=None,
        help="LAMMPS executable or command (default: selected by version tag)"
    )
    parser.add_argument(
        "--lammps-version",
        type=str,
        choices=["2025", "2022"],
        default="2025",
        help="LAMMPS version tag (default: 2025)"
    )
    parser.add_argument(
        "--lammps-2025",
        type=str,
        default="/home/abel/opt/lammps-nep-gap/build/lmp",
        help="LAMMPS 2025 executable path"
    )
    parser.add_argument(
        "--lammps-2022",
        type=str,
        default="/home/abel/opt/lammps-23Jun2022-intelmpi-2021.17.2/bin/lmp",
        help="LAMMPS 2022 executable path"
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum number of jobs to run (default: unlimited)"
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip LAMMPS run (use existing results)"
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count)"
    )
    
    args = parser.parse_args()
    if args.lammps is None:
        args.lammps = args.lammps_2025 if args.lammps_version == "2025" else args.lammps_2022
    
    # Validate multi-model mode
    if args.multi_model:
        if len(args.forcefield) < 2:
            print("Error: Multi-model mode requires at least 2 forcefields")
            return 1
        
        if args.name and len(args.name) != len(args.forcefield):
            print(f"Error: Number of names ({len(args.name)}) must match number of forcefields ({len(args.forcefield)})")
            return 1
        
        if not args.output_name:
            print("Error: Multi-model mode requires --output-name")
            return 1
        
        return main_multi_model(args)
    else:
        # Single model mode - use first forcefield only
        if len(args.forcefield) > 1:
            print("Warning: Multiple forcefields provided but not in multi-model mode. Using first one only.")
            print("         Use --multi-model flag for multi-model comparison.")
        
        args.forcefield = args.forcefield[0]
        if args.name:
            args.name = args.name[0] if args.name else None
        
        return main_single_model(args)


def main_single_model(args):
    """Single model workflow"""
    # Auto-detect potential type
    ff_path = Path(args.forcefield)
    if ff_path.is_file() and ff_path.suffix == '.txt':
        potential_type = "nep"
    elif ff_path.is_dir():
        potential_type = "tabgap"
    else:
        print(f"Warning: Cannot identify forcefield type, defaulting to tabgap")
        potential_type = "tabgap"
    
    # Path management
    workspace_root = Path(__file__).parent.parent
    
    # Auto-generate test name if not specified
    if args.name is None:
        test_name = generate_test_name(args.forcefield)
        print(f"Auto-generated test name: {test_name}")
    else:
        test_name = args.name
    
    raw_data_dir = workspace_root / "run" / "raw_data" / test_name
    output_dir = raw_data_dir
    plot_output = workspace_root / "run" / "analysis" / test_name / "energy_volume_comparison.png"
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Set default run script based on potential type
    if args.run_script is None:
        if potential_type == "nep":
            args.run_script = "scripts/run_nep.in"
        else:
            args.run_script = "scripts/run_gap.in"
    
    # Check input file
    test_xyz_path = Path(args.test_xyz)
    if not test_xyz_path.exists():
        print(f"Error: Test xyz file {args.test_xyz} does not exist")
        return 1
    
    print("=" * 80)
    print("Test Set Energy-Volume Comparison Workflow")
    print("=" * 80)
    print(f"Test name: {test_name}")
    print(f"Test xyz: {test_xyz_path}")
    print(f"Potential type: {potential_type.upper()}")
    print(f"Forcefield: {args.forcefield}")
    print(f"Run script: {args.run_script}")
    print(f"LAMMPS: {args.lammps} (version: {args.lammps_version})")
    print(f"Output directory: {output_dir}")
    print(f"Plot output: {plot_output}")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Step 1: Split xyz file
    print("\n[Step 1/6] Splitting xyz file into separate frames...")
    frames = read_xyz_frames(str(test_xyz_path))
    print(f"  Read {len(frames)} frames")
    
    if not args.skip_run:
        save_frames_to_folders(frames, str(output_dir))
        print(f"  ✓ Saved to {output_dir}")
        
        # Step 2: Convert to LAMMPS format
        print("\n[Step 2/6] Converting structure.xyz to LAMMPS format...")
        success, fail = convert_structures_to_lammps(str(output_dir))
        print(f"  ✓ Conversion complete: {success} successful, {fail} failed")
        
        # Step 3: Link forcefield files
        print("\n[Step 3/6] Linking forcefield files...")
        ff_count = create_symlinks_forcefield(args.forcefield, str(output_dir), potential_type)
        print(f"  ✓ Created {ff_count} forcefield symlinks")
        
        # Step 4: Link run.in script
        print("\n[Step 4/6] Linking run.in script...")
        run_count = create_symlinks_run_script(args.run_script, str(output_dir))
        print(f"  ✓ Created {run_count} run.in symlinks")
        
        # Step 5: Run LAMMPS in batch
        print("\n[Step 5/6] Running LAMMPS calculations in batch...")
        
        lammps_candidate = args.lammps
        found = shutil.which(lammps_candidate)
        lammps_path = Path(found) if found else Path(lammps_candidate).resolve()
        if not lammps_path.exists() or not os.access(str(lammps_path), os.X_OK):
            print(f"  Error: Cannot find executable LAMMPS ({args.lammps}), check PATH or provide absolute path")
            return 1
        
        subdirs = [d for d in output_dir.iterdir() if d.is_dir() and (d / "run.in").exists()]
        subdirs = sorted(subdirs)
        
        if args.max_jobs is not None and args.max_jobs > 0:
            subdirs = subdirs[:args.max_jobs]
            print(f"  Limited to first {args.max_jobs} jobs")
        
        n_cores = args.n_cores if args.n_cores else os.cpu_count()
        print(f"  Starting {len(subdirs)} jobs (using {n_cores} processes)...")
        
        success_count = 0
        fail_count = 0
        completed = 0
        
        tasks = [(lammps_path, subdir, i+1, len(subdirs)) for i, subdir in enumerate(subdirs)]
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            future_to_task = {executor.submit(run_lammps_wrapper, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                completed += 1
                try:
                    index, dir_name, success, message = future.result()
                    
                    if success:
                        print(f"  [{completed}/{len(subdirs)}] {dir_name}: ✓ {message}")
                        success_count += 1
                    else:
                        print(f"  [{completed}/{len(subdirs)}] {dir_name}: ✗ {message}")
                        fail_count += 1
                        
                except Exception as e:
                    print(f"  [{completed}/{len(subdirs)}] Task exception: {str(e)}")
                    fail_count += 1
        
        print(f"  ✓ Run complete: {success_count} successful, {fail_count} failed")
    else:
        print("\n[Steps 2-5] Skipped LAMMPS run (using existing results)")
    
    # Step 6: Generate energy-volume comparison plot
    print("\n[Step 6/6] Generating energy-volume comparison plot...")
    
    # Read test.xyz structures with energy alignment for tabGAP
    align_for_tabgap = (potential_type == "tabgap")
    if align_for_tabgap:
        print("  Applying energy baseline alignment (NEP → TabGAP)")
    
    structures = read_test_xyz(str(test_xyz_path), align_for_tabgap=align_for_tabgap)
    
    # Collect predicted energies from LAMMPS output
    predicted_energies = collect_predicted_energies(output_dir)
    
    # Combine data
    combined_data = combine_data(structures, predicted_energies)
    
    if len(combined_data) == 0:
        print("  Error: No valid data to plot")
        return 1
    
    # Generate plot
    model_name = ff_path.stem if ff_path.is_file() else potential_type.upper()
    plot_energy_volume_comparison(combined_data, str(plot_output), model_name)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print(f"Workflow complete! Total time: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print("=" * 80)
    
    print(f"\nResults location:")
    print(f"  Raw data: {output_dir}")
    print(f"  Plot: {plot_output}")
    
    return 0


def main_multi_model(args):
    """Multi-model comparison workflow"""
    workspace_root = Path(__file__).parent.parent
    test_xyz_path = Path(args.test_xyz)
    
    if not test_xyz_path.exists():
        print(f"Error: Test xyz file {args.test_xyz} does not exist")
        return 1
    
    print("=" * 100)
    print("Multi-Model Energy-Volume Comparison Workflow")
    print("=" * 100)
    print(f"Number of models: {len(args.forcefield)}")
    print(f"Test xyz: {test_xyz_path}")
    print(f"Output name: {args.output_name}")
    print(f"LAMMPS: {args.lammps} (version: {args.lammps_version})")
    print("=" * 100)
    
    start_time = datetime.now()
    
    # First pass: detect if tabGAP is present
    has_tabgap = False
    potential_types = []
    
    for forcefield in args.forcefield:
        ff_path = Path(forcefield)
        if ff_path.is_file() and ff_path.suffix == '.txt':
            potential_type = "nep"
        elif ff_path.is_dir():
            potential_type = "tabgap"
            has_tabgap = True
        else:
            potential_type = "tabgap"
            has_tabgap = True
        potential_types.append(potential_type)
    
    # Determine baseline alignment strategy
    if has_tabgap:
        print("\n⚠️  TabGAP detected: All energies will be aligned to TabGAP baseline")
        print("   - DFT energies: NEP → TabGAP baseline conversion")
        print("   - NEP predictions: NEP → TabGAP baseline conversion")
        print("   - TabGAP predictions: No conversion (already on TabGAP baseline)")
        align_to_tabgap = True
    else:
        print("\n📌 No TabGAP detected: Using NEP baseline (no energy alignment)")
        align_to_tabgap = False
    
    # Initialize energy aligner if needed
    aligner = EnergyAligner() if align_to_tabgap else None
    
    # Process each model
    all_model_data = []
    model_names = []
    test_names = []
    
    for model_idx, forcefield in enumerate(args.forcefield):
        print(f"\n{'='*100}")
        print(f"Processing Model {model_idx + 1}/{len(args.forcefield)}: {forcefield}")
        print(f"{'='*100}")
        
        potential_type = potential_types[model_idx]
        
        # Generate test name for this model
        test_name = generate_test_name(forcefield)
        test_names.append(test_name)
        
        # Get model display name
        ff_path = Path(forcefield)
        if args.name and model_idx < len(args.name):
            model_name = args.name[model_idx]
        else:
            model_name = ff_path.stem if ff_path.is_file() else potential_type.upper()
        model_names.append(model_name)
        
        raw_data_dir = workspace_root / "run" / "raw_data" / test_name
        
        print(f"  Model name: {model_name}")
        print(f"  Potential type: {potential_type.upper()}")
        print(f"  Test name: {test_name}")
        print(f"  Raw data dir: {raw_data_dir}")
        print(f"  LAMMPS: {args.lammps} (version: {args.lammps_version})")
        if align_to_tabgap and potential_type == "nep":
            print(f"  Energy alignment: NEP → TabGAP baseline")
        
        # Set default run script based on potential type
        run_script = args.run_script
        if run_script is None:
            run_script = "scripts/run_nep.in" if potential_type == "nep" else "scripts/run_gap.in"
        
        # Run LAMMPS workflow if not skipped
        if not args.skip_run:
            print(f"\n  [Model {model_idx + 1}] Running LAMMPS workflow...")
            
            # Step 1: Split xyz file
            print(f"    [1/5] Splitting xyz file...")
            frames = read_xyz_frames(str(test_xyz_path))
            save_frames_to_folders(frames, str(raw_data_dir))
            print(f"    ✓ Saved {len(frames)} frames")
            
            # Step 2: Convert to LAMMPS format
            print(f"    [2/5] Converting to LAMMPS format...")
            success, fail = convert_structures_to_lammps(str(raw_data_dir))
            print(f"    ✓ Conversion: {success} successful, {fail} failed")
            
            # Step 3: Link forcefield files
            print(f"    [3/5] Linking forcefield files...")
            ff_count = create_symlinks_forcefield(forcefield, str(raw_data_dir), potential_type)
            print(f"    ✓ Created {ff_count} forcefield symlinks")
            
            # Step 4: Link run.in script
            print(f"    [4/5] Linking run.in script...")
            run_count = create_symlinks_run_script(run_script, str(raw_data_dir))
            print(f"    ✓ Created {run_count} run.in symlinks")
            
            # Step 5: Run LAMMPS
            print(f"    [5/5] Running LAMMPS calculations...")
            
            lammps_candidate = args.lammps
            found = shutil.which(lammps_candidate)
            lammps_path = Path(found) if found else Path(lammps_candidate).resolve()
            if not lammps_path.exists() or not os.access(str(lammps_path), os.X_OK):
                print(f"    Error: Cannot find executable LAMMPS ({args.lammps}), check PATH or provide absolute path")
                return 1
            
            subdirs = [d for d in raw_data_dir.iterdir() if d.is_dir() and (d / "run.in").exists()]
            subdirs = sorted(subdirs)
            
            if args.max_jobs is not None and args.max_jobs > 0:
                subdirs = subdirs[:args.max_jobs]
            
            n_cores = args.n_cores if args.n_cores else os.cpu_count()
            print(f"    Starting {len(subdirs)} jobs (using {n_cores} processes)...")
            
            success_count = 0
            fail_count = 0
            completed = 0
            
            tasks = [(lammps_path, subdir, i+1, len(subdirs)) for i, subdir in enumerate(subdirs)]
            
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                future_to_task = {executor.submit(run_lammps_wrapper, task): task for task in tasks}
                
                for future in as_completed(future_to_task):
                    completed += 1
                    try:
                        index, dir_name, success, message = future.result()
                        
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                            
                    except Exception as e:
                        fail_count += 1
            
            print(f"    ✓ Run complete: {success_count} successful, {fail_count} failed")
        else:
            print(f"\n  [Model {model_idx + 1}] Skipped LAMMPS run (using existing results)")
        
        # Collect data for this model
        print(f"\n  [Model {model_idx + 1}] Collecting prediction data...")
        
        # Read test.xyz structures with energy alignment
        # All DFT energies use TabGAP baseline if align_to_tabgap is True
        structures = read_test_xyz(str(test_xyz_path), align_for_tabgap=align_to_tabgap)
        
        # Collect predicted energies
        predicted_energies = collect_predicted_energies(raw_data_dir)
        
        # Combine structure data with predictions
        combined_data = combine_data(structures, predicted_energies)
        
        # Apply energy alignment to NEP predictions if needed
        if align_to_tabgap and potential_type == "nep" and len(combined_data) > 0:
            print(f"    Applying TabGAP baseline alignment to NEP predictions...")
            
            # Build a mapping from combined_data index to structure
            # combined_data only contains valid predictions, so we need to match carefully
            valid_idx = 0
            for struct_idx, struct in enumerate(structures):
                if predicted_energies[struct_idx] is not None and valid_idx < len(combined_data):
                    # This structure has a valid prediction
                    if 'alignment_offset' in struct:
                        # Apply the same offset to predicted energy
                        combined_data[valid_idx]['predicted_energy_per_atom'] += struct['alignment_offset']
                    valid_idx += 1
            
            print(f"    ✓ Energy alignment applied to {len(combined_data)} NEP predictions")
        
        if len(combined_data) == 0:
            print(f"    Warning: No valid data for model {model_name}")
            all_model_data.append([])
        else:
            all_model_data.append(combined_data)
            print(f"    ✓ Collected {len(combined_data)} valid data points")
    
    # Generate multi-model comparison plot
    print(f"\n{'='*100}")
    print("Generating Multi-Model Comparison Plot")
    print(f"{'='*100}")
    
    # Filter out empty models
    valid_data = []
    valid_names = []
    for i, data in enumerate(all_model_data):
        if len(data) > 0:
            valid_data.append(data)
            valid_names.append(model_names[i])
    
    if len(valid_data) == 0:
        print("Error: No valid data to plot")
        return 1
    
    plot_output = workspace_root / "run" / "analysis" / args.output_name / "energy_volume_comparison.png"
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    
    plot_multi_model_comparison(valid_data, str(plot_output), valid_names)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 100)
    print(f"Multi-model workflow complete! Total time: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print("=" * 100)
    
    print(f"\nResults:")
    print(f"  Plot: {plot_output}")
    for i, test_name in enumerate(test_names):
        print(f"  Model {i+1} ({model_names[i]}) raw data: run/raw_data/{test_name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
