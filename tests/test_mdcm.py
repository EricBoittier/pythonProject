from mdcm import conv_clcl_cxyz
from mdcm import compute_local_axes
from mdcm import compute_esp
from mdcm_io import read_charmm_mdcm, write_dcm_xyz
from constants import mdcm_path, cubes_path

import jax.numpy as jnp

doEval = False

atoms, stackData, charges = read_charmm_mdcm(mdcm_path /"charmm/pbe0_dz.mdcm")
# flatten charges
charges = charges.reshape(-1)

test_coords = jnp.array( [
    [0.000, 0.000, 0.000],
    [0.000, 0.000, 0.970],
    [0.000, 0.939, -0.243],
    [0.000, 0.000, 2.470],
    [0.840, 0.000, 2.955],
    [-0.840, 0.000, 2.955],
])

frames = jnp.array([
    [1,0,2],
    [4, 3, 5]])

positions = []
for frame in frames:
    # index by frame order
    cla = compute_local_axes(test_coords[frame,:])
    positions.append(conv_clcl_cxyz(
        test_coords[frame,:],
        cla,
        stackData,
    ))

print(positions)
positions = jnp.stack(positions, axis=0).reshape(-1, 3)
print(positions)
print(charges)
charges = jnp.tile(charges, 2)
print(charges)
write_dcm_xyz("test.xyz", positions, charges)

if doEval:
    from psi4_ import surface_points, data, reference_esp, monomer_coords

    # evaluate grid points
    esp = compute_esp(positions, charges, surface_points)
    error = (esp - jnp.array(reference_esp))**2
    MSE = jnp.sqrt(error.sum()) / len(esp)
    RMSE = jnp.sqrt(error.sum()) / len(esp)
    MAE = jnp.abs(error).sum() / len(esp)
    MAXERROR = jnp.abs(error).max()

    print("MSE", MSE*627.509469)
    print("RMSE", RMSE*627.509469)
    print("MAE", MAE*627.509469)

    for i in range(len(esp)):
        if i % 1000 == 0:
            print(i, esp[i], reference_esp[i], esp[i] - reference_esp[i])
            print(i, esp[i], data[i], esp[i] - data[i])
