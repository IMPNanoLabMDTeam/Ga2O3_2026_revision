#!/usr/bin/env python3
"""
Energy alignment script for Ga2O3 training data.

This script aligns the training set zero-point energies from the original values 
(Ga=-0.0244486 eV, O=-0.0350174 eV) to NEP89 zero-point energies 
(Ga=-1.68768 eV, O=-3.19589 eV).

The alignment is performed by:
1. Reading each structure from the XYZ file
2. Counting the number of Ga and O atoms in each structure
3. Calculating the energy offset based on the difference in zero-point energies
4. Adding the offset to the original energy
"""

import re
import sys
from typing import List, Tuple, Dict


class EnergyAligner:
    def __init__(self):
        # Original zero-point energies (eV)
        self.original_zpe = {
            'Ga': -0.0244486,
            'O': -0.0350174
        }
        
        # NEP89 zero-point energies (eV)
        self.nep89_zpe = {
            'Ga': -1.68768,
            'O': -3.19589
        }
        
        # Calculate the energy differences
        self.energy_diff = {
            'Ga': self.nep89_zpe['Ga'] - self.original_zpe['Ga'],
            'O': self.nep89_zpe['O'] - self.original_zpe['O']
        }
        
        print(f"Energy differences to be applied:")
        print(f"Ga: {self.energy_diff['Ga']:.6f} eV per atom")
        print(f"O:  {self.energy_diff['O']:.6f} eV per atom")
    
    def parse_xyz_structure(self, lines: List[str], start_idx: int) -> Tuple[Dict, int]:
        """
        Parse a single structure from XYZ format.
        
        Args:
            lines: List of all lines in the file
            start_idx: Starting line index for this structure
            
        Returns:
            Tuple of (structure_dict, next_start_idx)
        """
        if start_idx >= len(lines):
            return None, start_idx
            
        # Read number of atoms
        try:
            n_atoms = int(lines[start_idx].strip())
        except (ValueError, IndexError):
            return None, start_idx + 1
            
        if start_idx + n_atoms + 1 >= len(lines):
            return None, len(lines)
        
        # Read comment line (contains energy and other properties)
        comment_line = lines[start_idx + 1].strip()
        
        # Extract energy from comment line
        energy_match = re.search(r'Energy=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', comment_line)
        if not energy_match:
            print(f"Warning: Could not find energy in structure starting at line {start_idx + 1}")
            return None, start_idx + n_atoms + 2
            
        original_energy = float(energy_match.group(1))
        
        # Count atoms
        atom_counts = {'Ga': 0, 'O': 0}
        atom_lines = []
        
        for i in range(n_atoms):
            line_idx = start_idx + 2 + i
            if line_idx >= len(lines):
                break
                
            line = lines[line_idx].strip()
            if line:
                parts = line.split()
                if len(parts) >= 1:
                    element = parts[0]
                    if element in atom_counts:
                        atom_counts[element] += 1
                    atom_lines.append(line)
        
        # Calculate energy offset
        energy_offset = (atom_counts['Ga'] * self.energy_diff['Ga'] + 
                        atom_counts['O'] * self.energy_diff['O'])
        
        aligned_energy = original_energy + energy_offset
        
        # Create new comment line with aligned energy
        new_comment_line = re.sub(
            r'Energy=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            f'Energy={aligned_energy:.8f}',
            comment_line
        )
        
        structure = {
            'n_atoms': n_atoms,
            'comment_line': new_comment_line,
            'atom_lines': atom_lines,
            'original_energy': original_energy,
            'aligned_energy': aligned_energy,
            'energy_offset': energy_offset,
            'atom_counts': atom_counts
        }
        
        return structure, start_idx + n_atoms + 2
    
    def align_energies(self, input_file: str, output_file: str):
        """
        Read XYZ file, align energies, and write output file.
        
        Args:
            input_file: Path to input XYZ file
            output_file: Path to output XYZ file
        """
        print(f"Reading input file: {input_file}")
        
        # Read all lines
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        print(f"Total lines in input file: {len(lines)}")
        
        structures = []
        idx = 0
        structure_count = 0
        
        while idx < len(lines):
            structure, next_idx = self.parse_xyz_structure(lines, idx)
            
            if structure is not None:
                structures.append(structure)
                structure_count += 1
                
                if structure_count % 1000 == 0:
                    print(f"Processed {structure_count} structures...")
                
                # Print details for first few structures
                if structure_count <= 5:
                    print(f"\nStructure {structure_count}:")
                    print(f"  Atoms: {structure['atom_counts']}")
                    print(f"  Original energy: {structure['original_energy']:.8f} eV")
                    print(f"  Energy offset: {structure['energy_offset']:.8f} eV")
                    print(f"  Aligned energy: {structure['aligned_energy']:.8f} eV")
            
            idx = next_idx
        
        print(f"\nTotal structures processed: {len(structures)}")
        
        # Write aligned data
        print(f"Writing aligned data to: {output_file}")
        
        with open(output_file, 'w') as f:
            for structure in structures:
                # Write number of atoms
                f.write(f"{structure['n_atoms']}\n")
                
                # Write comment line with aligned energy
                f.write(f"{structure['comment_line']}\n")
                
                # Write atom lines
                for atom_line in structure['atom_lines']:
                    f.write(f"{atom_line}\n")
        
        print(f"Successfully wrote {len(structures)} aligned structures to {output_file}")
        
        # Print summary statistics
        total_original_energy = sum(s['original_energy'] for s in structures)
        total_aligned_energy = sum(s['aligned_energy'] for s in structures)
        total_offset = sum(s['energy_offset'] for s in structures)
        
        print(f"\nSummary:")
        print(f"  Total original energy: {total_original_energy:.6f} eV")
        print(f"  Total aligned energy: {total_aligned_energy:.6f} eV")
        print(f"  Total energy offset: {total_offset:.6f} eV")


def main():
    """Main function to run energy alignment."""
    if len(sys.argv) != 3:
        print("Usage: python energy_alignment.py <input_xyz_file> <output_xyz_file>")
        print("Example: python energy_alignment.py npj2023.xyz npj2023_aligned.xyz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        aligner = EnergyAligner()
        aligner.align_energies(input_file, output_file)
        print("\nEnergy alignment completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during energy alignment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
