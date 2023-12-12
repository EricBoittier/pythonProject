import jax.numpy as jnp
from mdcm import conv_clcl_cxyz
from mdcm import compute_local_axes
import jax
from psi4_ import surface_points, data, reference_esp, monomer_coords

def read_charmm_mdcm(filename):
    """

    :param filename:
    :return:
    """
    lines = open(filename).readlines()
    Nframes = int(lines[0].split()[0])
    #  blank line
    resName = lines[2].split()[0]
    nAxis = int(lines[3].split()[0])
    Nframe1 = int(lines[4].split()[0])
    frameInfo = lines[4]
    a1 = int(lines[4].split()[0])
    a2 = int(lines[4].split()[1])
    a3 = int(lines[4].split()[2])
    frametype = str(lines[4].split()[3])
    Nchg1 = int(lines[5].split()[0])
    atom1 = lines[6:6 + Nchg1]
    Nchg2 = int(lines[6 + Nchg1].split()[0])
    atom2 = lines[7 + Nchg1:7 + Nchg1 + Nchg2]
    Nchg3 = int(lines[7 + Nchg1 + Nchg2].split()[0])
    atom3 = lines[8 + Nchg1 + Nchg2:8 + Nchg1 + Nchg2 + Nchg3]
    aatom1 = [[float(x) for x in line.split()[:-1]] for line in atom1]
    aatom2 = [[float(x) for x in line.split()[:-1]] for line in atom2]
    aatom3 = [[float(x) for x in line.split()[:-1]] for line in atom3]
    cAt1 = [[float(x) for x in line.split()[-1:]] for line in atom1]
    cAt2 = [[float(x) for x in line.split()[-1:]] for line in atom2]
    cAt3 = [[float(x) for x in line.split()[-1:]] for line in atom3]
    atom1 = jnp.array(aatom1)
    atom2 = jnp.array(aatom2)
    atom3 = jnp.array(aatom3)
    ATOMS = []
    for _ in atom1:
        ATOMS.append(_)
    for _ in atom2:
        ATOMS.append(_)
    for _ in atom3:
        ATOMS.append(_)
    stackedData = jnp.array([atom1, atom2, atom3])
    return jnp.array([a1, a2, a3]), stackedData, jnp.array([cAt1, cAt2, cAt3])


atoms, stackData, charges = read_charmm_mdcm("mdcm/charmm/pbe0_dz.mdcm")
charges = charges.reshape(-1)
print("chgs",charges)

test_coords = jnp.array( [
    [0.000, 0.000, 0.000],
    [0.000, 0.000, 0.970],
    [0.000, 0.939, -0.243]
])
test_coords = monomer_coords

cla = compute_local_axes(test_coords.T[0,:],
                         test_coords.T[1,:],
                         test_coords.T[2,:],
                         2, 1, 3, 0)

positions = conv_clcl_cxyz(
    test_coords[[1,0,2],:],
    cla,
    stackData,
)
coloumns_constant = 3.32063711e2 / 627.509469
# evaluate grid points
def coulomb(q1, q2, r12):
    return coloumns_constant * (q1 * q2 / r12)

def compute_esp(positions, charges, grid_points):
    esp = jnp.zeros_like(grid_points.shape[0])
    jax.debug.print("{x}", x=grid_points.shape)
    jax.debug.print("chg{x}", x=charges.shape)

    for i in range(len(charges)):
        distances = jnp.linalg.norm(grid_points - positions[i], axis=1)
        if i == 0:
            jax.debug.print("dist: {x}", x=distances)
        esp += coulomb(charges[i,None], 1.0, distances)
    # esp = esp.reshape(-1)
    return esp

esp = compute_esp(positions, charges, surface_points)
error = (esp - jnp.array(reference_esp))**2
MSE = jnp.sqrt(error.sum()) / len(esp)
RMSE = jnp.sqrt(error.sum()) / len(esp)
MAE = jnp.abs(error).sum() / len(esp)
MAXERROR = jnp.abs(error).max()
print("MSE", MSE*627.509469)
print("RMSE", RMSE*627.509469)
print("MAE", MAE*627.509469)
print("MAXERROR", MAXERROR*627.509469)

# print("error", abs((esp - jnp.array(reference_esp)).sum())/len(esp))
# print("MSE", jnp.sqrt(error.sum()) / len(esp) )
for i in range(len(esp)):
    if i % 1000 == 0:
        print(esp[i], reference_esp[i], esp[i] - reference_esp[i])
