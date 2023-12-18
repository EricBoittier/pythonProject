import os

import psi4

from project.constants_ import mdcm_path, cubes_path, psi4_path
from project.rdkit_ import get_water_data, get_pdb_data
import jax.numpy as np
from scipy.spatial.distance import cdist

psi4.set_options({'basis': 'def2-TZVP', })
B_to_A = 0.529177249


def make_grid_data(surface_points):
    """
    create a grid.dat file for psi4
    :param surface_points:
    :return:
    """
    with open('grid.dat', 'w') as file:
        for xyz in surface_points:
            for c in xyz:
                file.write(str(c) + ' ')
            file.write('\n')

def get_surface_points(coordinates):
    """
    return surface points for a given molecule
    :param coordinates:
    :return:
    """
    N_points, CUTOFF = 400, 1.0
    monomer_coords = coordinates.copy()
    surface_points = np.random.normal(size=[N_points, 3])
    surface_points = (surface_points / np.linalg.norm(
        surface_points, axis=-1, keepdims=True)) * CUTOFF
    surface_points = np.reshape(
        surface_points[None] + monomer_coords[:, None], [-1, 3])
    surface_points = surface_points[
        np.where(np.all(cdist(surface_points, monomer_coords
                              ) >= (CUTOFF - 1e-1), axis=-1))[0]]
    return surface_points


def get_points_from_cube(filename):
    pass


def get_grid_points(coordinates):
    """
    create a uniform grid of points around the molecule,
    starting from minimum and maximum coordinates of the molecule (plus minus some padding)
    :param coordinates:
    :return:
    """
    bounds = np.array([np.min(coordinates, axis=0),
                       np.max(coordinates, axis=0)])
    padding = 2.0
    bounds = bounds + np.array([-1, 1])[:, None] * padding
    grid_points = np.meshgrid(*[np.linspace(a, b, 10)
                                for a,b in zip(bounds[0], bounds[1])])

    grid_points = np.stack(grid_points, axis=0)
    grid_points = np.reshape(grid_points.T, [-1, 3])
    #  exclude points that are too close to the molecule
    grid_points = grid_points[
        np.where(np.all(cdist(grid_points, coordinates) >= (1.0 - 1e-1), axis=-1))[0]]

    return grid_points


def test_mbis(test="water"):
    """
    example of psi4
    :return:
    """
    cube1 = None
    if test == "water":
        elements, monomer_coords = get_water_data()
    elif test == "cube":
        from cubes_ import cube
        import numpy as np
        cube_file = cubes_path / "gaussian/testjax.chk.p.cube"
        cube1 = cube(cube_file)
        elements, monomer_coords = cube1.get_atom_data()
    elif test == "pdb":
        elements, monomer_coords = get_pdb_data()
    else:
        raise NotImplementedError

    for e,c in zip(elements, monomer_coords):
        print(e, " ".join([str(_) for _ in c]))

    if test == "cube":
        surface_points = cube1.get_grid()
        # surface_points = get_grid_points(monomer_coords)
        # surface_points = get_surface_points(monomer_coords)
    else:
        # surface_points = get_surface_points(monomer_coords)
        surface_points = get_grid_points(monomer_coords)

    esp_calc(surface_points, monomer_coords, elements)
    reference_esp = [float(x) for x in open('grid_esp.dat')]
    data = None
    if cube1 is not None:
        data = cube1.data.flatten()
        MSE = np.mean((data - reference_esp) ** 2)
        print("mse-psi4", MSE)

        for i in range(len(reference_esp)):
            print(i, data[i], reference_esp[i], data[i] / reference_esp[i])

    return surface_points, data, reference_esp, monomer_coords

def esp_calc(surface_points, monomer_coords, elements):
    make_grid_data(surface_points)
    psi4_mol = psi4.core.Molecule.from_arrays(monomer_coords, elem=elements,
                                              fix_orientation=True,
                                              fix_com=True,)
    psi4.core.set_output_file('output.dat', False)
    e, wfn = psi4.energy('PBE0', molecule=psi4_mol, return_wfn=True)
    psi4.oeprop(wfn, 'GRID_ESP', 'MBIS_CHARGES', title='MBIS Multipoles')


def make_psi4_dir(filename):
    """
    create a psi4 directory and change to it
    :param filename:
    :return:
    """
    print(filename)
    psi4_dir = psi4_path / filename.stem
    if not psi4_dir.exists():
        psi4_dir.mkdir()
    os.chdir(psi4_dir)
    return psi4_dir


# test_mbis(test="pdb")
#import os
# os.chdir("/Users/ericboittier/Documents/github/pythonProject/psi4")
# surface_points, data, reference_esp, monomer_coords = test_mbis(test="pdb")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pdb", type=str, default=None)
    args = parser.parse_args()
    if args.pdb is not None:
        pdb = Path(args.pdb)
        print(type(pdb))
        make_psi4_dir(pdb)
    else:
        raise NotImplementedError
    elements, coords = get_pdb_data(pdb)
    surface_points = get_grid_points(coords)
    esp_calc(surface_points, coords, elements)

