import psi4

from constants import mdcm_path, cubes_path
from rdkit_ import get_water_data, get_pdb_data
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
    bounds = np.array([np.min(coordinates, axis=0), np.max(coordinates, axis=0)])
    print(bounds)
    padding = 2.0
    bounds = bounds + np.array([-1, 1])[:, None] * padding
    print(bounds.shape)
    grid_points = np.meshgrid(*[np.linspace(a, b, 10) for a,b in zip(bounds[0], bounds[1])])

    # grid points are now a list of 3d arrays
    print(len(grid_points))
    for i in range(len(grid_points)):
        print(grid_points[i].shape)

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
    make_grid_data(surface_points)
    psi4_mol = psi4.core.Molecule.from_arrays(monomer_coords, elem=elements,
                                              fix_orientation=True,
                                              fix_com=True,)
    psi4.core.set_output_file('output.dat', False)
    e, wfn = psi4.energy('PBE0', molecule=psi4_mol, return_wfn=True)
    print(e, -76.3755896)
    psi4.oeprop(wfn, 'GRID_ESP', 'MBIS_CHARGES', title='MBIS Multipoles')


    reference_esp = [float(x) for x in open('grid_esp.dat')]
    if cube1 is not None:
        data = cube1.data.flatten()
        MSE = np.mean((data - reference_esp) ** 2)
        print("mse-psi4", MSE)


    for i in range(len(reference_esp)):
        print(i, data[i], reference_esp[i], data[i] / reference_esp[i])
    return surface_points, data, reference_esp, monomer_coords

# test_mbis(test="pdb")
surface_points, data, reference_esp, monomer_coords = test_mbis(test="cube")

