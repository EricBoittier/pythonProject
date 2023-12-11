import jax.numpy as jnp
# import numpy as jnp


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

    atom1 = [[float(x) for x in line.split()] for line in atom1]
    atom2 = [[float(x) for x in line.split()] for line in atom2]
    atom3 = [[float(x) for x in line.split()] for line in atom3]
    atom1 = jnp.array(atom1)
    atom2 = jnp.array(atom2)
    atom3 = jnp.array(atom3)

    print(a1, a2, a3, frametype)
    print(atom1)
    print(atom2)
    print(atom3)

    ATOMS = []
    for _ in atom1:
        ATOMS.append(_)
    for _ in atom2:
        ATOMS.append(_)
    for _ in atom3:
        ATOMS.append(_)
    stackedData = jnp.array(ATOMS)
    print(stackedData)
    print(stackedData.shape)

    return jnp.array([a1, a2, a3]), stackedData


atoms, stackData = read_charmm_mdcm("mdcm/charmm/pbe0_dz.mdcm")

test_coords = jnp.array( [
    [0.000, 0.000, 0.000],
    [0.000, 0.000, 0.970],
    [0.000, 0.939, -0.243]
])

from mdcm import conv_clcl_cxyz
# from mdcm import calc_axis_locl_pos
from mdcm import compute_local_axes

# fatom_elcl = calc_axis_locl_pos(
#     test_coords,
#     1,
#     jnp.array([[2,1,3],[2,1,3],[2,1,3]]),
#     jnp.array([0]),
# )
# print(fatom_elcl)
# print(fatom_elcl.shape)

cla = compute_local_axes(test_coords.T[0,:],
                         test_coords.T[1,:],
                         test_coords.T[2,:],
                         2, 1, 3, 0)

print(cla)

conv_clcl_cxyz(
    test_coords,
    cla,
    6,
    jnp.array([[2,1,3],[2,1,3],[2,1,3]]),
    jnp.array([[2,],[2,],[2,]]),
    stackData.flatten(),
)
