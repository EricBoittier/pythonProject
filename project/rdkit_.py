import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from pathlib import Path

dtype = np.float32

def get_water():
    """
    example of rdkit
    returns a water molecule with hydrogens and 3d coordinates
    :return:
    """
    water = Chem.MolFromSmiles('O')
    # add hydrogens
    water = Chem.AddHs(water)
    # add 3d coordinates
    AllChem.EmbedMolecule(water)
    return water

def get_water_data():
    """
    example of rdkit
    returns a water molecule with hydrogens and 3d coordinates
    :return:
    """
    water = get_water()
    return mol2data(water)

def mol2data(mol: rdkit.Chem.Mol, filename=None):
    """
    example of rdkit
    returns a molecule with hydrogens and 3d coordinates
    :return:
    """
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    # Generate a conformation (this will change the geometry)
    # AllChem.EmbedMolecule(mol)
    coordinates = mol.GetConformer(0).GetPositions().astype(dtype)
    if filename is not None:
        pdb = Chem.MolToPDBBlock(mol)
        with open(filename, 'w') as file:
            file.write(pdb)
    return elements, coordinates

def read_pdb(filename):
    """
    example of rdkit
    returns a water molecule with hydrogens and 3d coordinates
    :return:
    """
    with open(filename, 'r') as file:
        pdb = file.read()
    mol = Chem.MolFromPDBBlock(pdb, sanitize=False, removeHs=False)

    return mol

def get_pdb_data(pdb_file="/home/boittier/Documents/phd/pythonProject/pdb/gly-70150091-70a0-453f-b6b8-c5389f387e84-min.pdb"):
    """load test pdb files"""
    pdb_mol = read_pdb(pdb_file)
    return mol2data(pdb_mol)



# get_pdb_data()