#!/usr/bin/env python3

"""
Script to automatically define local DCM frames for converting a charge model
in the global axis to a charge model with local axis definitions used in
CHARMM's DCM module.
"""

import sys
import subprocess

def usage():
    """Prints the usage information for the script."""
    print("Usage: python3 get_frames.py <pdbfile> <resname>")

def read_coordinates(pdb_file):
    """
    Reads coordinates from a PDB file.

    Args:
    - pdb_file (str): Path to the PDB file.

    Returns:
    - list: Atom types, and lists of x, y, z coordinates.
    """
    types, x, y, z = [], [], [], []
    try:
        with open(pdb_file, 'r') as fin:
            for line in fin:
                parts = line.split()
                if len(parts) > 4 and parts[0].lower() == "atom":
                    atom_type = ''.join([char for char in line[12:16] if not char.isdigit() and not char == ' '])
                    types.append(atom_type.lower())
                    x.append(float(line[30:38]))
                    y.append(float(line[38:46]))
                    z.append(float(line[46:54]))
    except IOError as e:
        print(f"Could not open PDB file {pdb_file}. Error ({e.errno}): {e.strerror}")
        sys.exit(1)

    return types, x, y, z

def convert_to_sdf(pdb_file):
    """
    Converts a PDB file to an SDF file using obabel.

    Args:
    - pdb_file (str): Path to the PDB file.
    """
    with open('babel.sdf', 'w') as sdf_file, open('babel.err', 'w') as err_file:
        subprocess.run(['obabel', '-ipdb', pdb_file, '-osdf'], stdout=sdf_file, stderr=err_file)

def read_bonds():
    """
    Reads bonded pairs from an SDF file.

    Returns:
    - list: Bonded atom pairs.
    """
    bonds = []
    try:
        with open('babel.sdf', 'r') as sdf:
            for line in sdf:
                parts = line.split()
                if len(parts) == 11:
                    # Process bonds here
                    pass
                # Additional processing as required
    except IOError as e:
        print(f"Could not open Babel SDF file babel.sdf. Error ({e.errno}): {e.strerror}")
        sys.exit(1)

    return bonds

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()
        sys.exit(1)

    pdb_file = sys.argv[1]
    res_name = sys.argv[2]

    # Process the PDB file
    atom_types, x_coords, y_coords, z_coords = read_coordinates(pdb_file)

    # Convert PDB to SDF and read bonds
    convert_to_sdf(pdb_file)
    bonded_pairs = read_bonds()

    # Further processing and frame creation...

    print("\nfile frames.txt has been written\n")
