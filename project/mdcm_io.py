import jax.numpy as jnp
from jax import jit

def some_hash_function(x):
  return int(jnp.sum(x))

class HashableArrayWrapper:
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    return some_hash_function(self.val)
  def __eq__(self, other):
    return (isinstance(other, HashableArrayWrapper) and
            jnp.all(jnp.equal(self.val, other.val)))
  def __getitem__(self, key):
    return self.val[key]

def read_charmm_mdcm(filename: str)->tuple:
    """

    :param filename:
    :return:
    """
    lines = open(filename).readlines()
    #  blank line
    resName = lines[2].split()[0]
    nAxis = int(lines[3].split()[0])
    Nframes = int(lines[3].split()[0])
    axisFrames = []

    frameInfos = []
    frameAtoms = []
    chargeArrays = []
    lineJump = 0
    while len(frameInfos) < Nframes:
        frameInfo = lines[4+lineJump]
        print("frameInfo", frameInfo)
        print("resName", resName)
        print("Nframes", Nframes)
        a1 = int(lines[4+lineJump].split()[0])
        a2 = int(lines[4+lineJump].split()[1])
        a3 = int(lines[4+lineJump].split()[2])
        frametype = str(lines[4+lineJump].split()[3])
        Nchg1 = int(lines[5+lineJump].split()[0])
        atom1 = lines[6+lineJump:6 + Nchg1+lineJump]
        Nchg2 = int(lines[6 + Nchg1+lineJump].split()[0])
        atom2 = lines[7+ Nchg1+lineJump:7 + Nchg1 + Nchg2+lineJump]
        Nchg3 = int(lines[7 + Nchg1 + Nchg2+lineJump].split()[0])
        atom3 = lines[8 + Nchg1 + Nchg2+lineJump:8 + Nchg1 + Nchg2 + Nchg3+lineJump]
        aatom1 = [[float(x) for x in line.split()[:-1]] for line in atom1]
        aatom2 = [[float(x) for x in line.split()[:-1]] for line in atom2]
        aatom3 = [[float(x) for x in line.split()[:-1]] for line in atom3]
        cAt1 = [[float(x) for x in line.split()[-1:]] for line in atom1]
        cAt2 = [[float(x) for x in line.split()[-1:]] for line in atom2]
        cAt3 = [[float(x) for x in line.split()[-1:]] for line in atom3]
        print("cAt1", cAt1)
        print("cAt2", cAt2)
        print("cAt3", cAt3)
        # pad arrays with zeros to make them all the same length
        max_charges = 2 #max(Nchg1, Nchg2, Nchg3)
        print("max charges", max_charges)
        if Nchg1 < max_charges:
            for _ in range(max_charges - Nchg1):
                aatom1.append([0, 0, 0])
                cAt1.append([0])
        if Nchg2 < max_charges:
            for _ in range(max_charges - Nchg2):
                aatom2.append([0, 0, 0])
                cAt2.append([0])
        if Nchg3 < max_charges:
            for _ in range(max_charges - Nchg3):
                aatom3.append([0, 0, 0])
                cAt3.append([0])

        cAt1 = jnp.array(cAt1)
        cAt2 = jnp.array(cAt2)
        cAt3 = jnp.array(cAt3)
        atom1 = jnp.array(aatom1)
        atom2 = jnp.array(aatom2)
        atom3 = jnp.array(aatom3)
        lineJump  += 4 + Nchg1 + Nchg2 + Nchg3
        print("lineJump", lineJump)
        chargeArrays.append(jnp.array([cAt1, cAt2, cAt3]))
        frameAtoms.append(jnp.array([a1, a2, a3]))
        frameInfos.append(frameInfo)
        axisFrames.append(jnp.array([atom1, atom2, atom3]))

    chargeArrays = jnp.array(chargeArrays)
    frameAtoms = jnp.array(frameAtoms)

    return frameInfos, frameAtoms, axisFrames, chargeArrays

def write_dcm_xyz(filename, positions, charges):
    """

    :param filename:
    :param positions:
    :param charges:
    :return:
    """
    with open(filename, "w") as f:
        nChgs = len([charges[i] for i in range(len(charges))
                     if not jnp.isclose(charges[i], 0)])
        f.write("%d\n" % nChgs)
        f.write("\n")
        for i, pos in enumerate(positions):
            if not jnp.isclose(charges[i], 0):
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

glu = "/home/boittier/Documents/phd/pythonProject/mdcm/gen/GLY.mdcm"
out = read_charmm_mdcm(glu)

from project.mdcm import compute_esp, calc_global_pos
from project.mdcm_opt import MDCMopt, print_loss
from project.rdkit_ import get_water_data, get_pdb_data

elements, coords = get_pdb_data()
frames, atoms, stackData, charges = out

print("frames", frames)


positions = calc_global_pos(atoms, coords, stackData[0])
charges = charges.flatten()
print("len(charges)", len(charges))

reference_esp = [float(x) for x in
                 open('/home/boittier/Documents/phd/pythonProject/psi4/grid_esp.dat')]

from project.psi4_ import get_grid_points

surface_points = get_grid_points(coords)

MDCMopt = MDCMopt(charges, positions, surface_points, reference_esp, frames)

# evaluate grid points
esp = compute_esp(positions, charges, surface_points)

write_dcm_xyz("test.xyz", positions, charges)

print("len(charges)", len(charges))

from jax.scipy import optimize

loss = MDCMopt.get_charges_loss()

nparms = MDCMopt.Nchgparm
from jax import random
key = random.PRNGKey(1)
randVals = random.uniform(key, shape=(nparms,), minval=-1, maxval=1)

res = optimize.minimize(loss, randVals, method='BFGS', tol=1e-3)
print(res)
print(charges)
x = res.x
x = jnp.take(x, MDCMopt.chg_typ_idx)

new_charges = MDCMopt.get_constraint()(charges * x)

print("new charges", new_charges)
write_dcm_xyz("testout.xyz", positions, new_charges)

print(len([res.x[i] for i in range(len(new_charges)) if new_charges[i] != 0]))
print(new_charges)
print("sum:", new_charges.sum())
esp = compute_esp(positions, new_charges, surface_points)
print_loss(esp,
         reference_esp)