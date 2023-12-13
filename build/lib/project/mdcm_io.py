import jax.numpy as jnp


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


def write_dcm_xyz(filename, positions, charges):
    """

    :param filename:
    :param positions:
    :param charges:
    :return:
    """
    with open(filename, "w") as f:
        f.write("%d\n" % len(positions))
        f.write("\n")
        for i, pos in enumerate(positions):
            f.write(
                "%s %f %f %f %f\n"
                % (
                    "X" if charges[i] <= 0 else "Y",
                    pos[0],
                    pos[1],
                    pos[2],
                    charges[i],
                )
            )