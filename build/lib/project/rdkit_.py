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
        print(pdb)
    mol = Chem.MolFromPDBBlock(pdb, sanitize=False, removeHs=False)

    return mol

def get_pdb_data():
    """load test pdb files"""
    pdb_path = Path("/Volumes/Extreme SSD/data/aa/pdb")
    pdb_files = list(pdb_path.glob("*.pdb"))
    pdb_files = [str(p) for p in pdb_files]
    print(pdb_files[1:2])
    pdb_mols = [read_pdb(p) for p in pdb_files[1:2]]
    return pdb_mols[0]


# get_pdb_data()