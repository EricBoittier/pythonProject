import jax
import jax.numpy as jnp
from jax import jit

# Constants
eps = 1e-12  # A small epsilon value


@jit
def compute_local_axes(XYZ):
    # assign arrays
    B1 = XYZ[0] - XYZ[1]
    RB1 = jnp.linalg.norm(B1)
    B2 = XYZ[2] - XYZ[1]
    RB2 = jnp.linalg.norm(B2)
    # local z-axis
    eZ = jnp.repeat(B1 / RB1, 3).reshape(3, -1)
    eZ = eZ.at[:, 2].set(B2 / RB2)
    # local y-axis
    FAC = jnp.cross(B1, B2)
    REY = jnp.linalg.norm(FAC)
    eY = jnp.repeat(FAC / REY, 3).reshape(3, -1)
    eX = jnp.cross(eZ.T, eY.T).T
    # local z-axis
    output = jnp.nan_to_num(jnp.array([eX, eY, eZ]))
    return output


@jit
def conv_clcl_cxyz(
        fatom_pos: jnp.ndarray,
        fatom_elcl: jnp.ndarray,
        mdcm_clcl: jnp.ndarray,
) -> jnp.ndarray:
    """Convert from local to global coordinates
    :param fatom_pos:
    :param fatom_elcl:
    :param mdcm_clcl:
    :param charge:
    :return:
    """
    # transpose mdcm_clcl so that it is (3,3,nchg)
    mdcm_clcl = jnp.transpose(mdcm_clcl, (1, 0, 2))
    # multiply the local charge centers by the local axes
    result = mdcm_clcl[:, :, jnp.newaxis, :] * fatom_elcl.T[None, :, :, :]
    # sum over the local axes
    result = result.sum(axis=3) + fatom_pos[None, :, :]
    # transpose to (natm,3)
    result = result.transpose(1, 0, 2).reshape(-1, 3)
    return result


@jit
def coulomb(q1, r12):
    # convert to atomic units
    # r12 = r12 * 1.889725989
    return q1 / (r12 * 1.889725989)


@jit
def compute_esp(positions, charges, grid_points):
    esp = jnp.zeros_like(grid_points.shape[0])
    for i in range(len(charges)):
        distances = jnp.linalg.norm(grid_points - positions[i], axis=1)
        esp += coulomb(charges[i, None], distances)
    return esp

@jit
def compute_esp_multi(positions, charges, grid_points, chg_idx, grid_idx):
    esp = jnp.zeros_like(grid_points[:,0])
    for i in range(len(charges)):
        distances = jnp.linalg.norm(grid_points - positions[i], axis=1)
        jax.debug.print("distances {x}", x=distances.shape)
        cond = jnp.where(chg_idx[i] == grid_idx, 1.0, 0.0).reshape(1, -1)
        jax.debug.print("cond {x}", x=cond.shape)
        ch = coulomb(charges[i], distances) * cond
        jax.debug.print("ch {x}", x=ch[::1000])
        esp = esp.at[:].add(ch.flatten())
    return esp

@jit
def calc_global_pos(atoms, coords, stackData):
    """

    :param atoms:
    :param coords:
    :param stackData:
    :return:
    """
    positions = []
    for i, frame in enumerate(atoms):
        # index by frame order
        _coords = coords[frame - 1, :]
        cla = compute_local_axes(_coords)
        pos = conv_clcl_cxyz(
            _coords,
            cla,
            stackData[i],
        )
        positions.append(pos)
    positions = jnp.array(positions)
    positions = positions.reshape(-1, 3)
    return positions


def process_frames(frames, atomCentered=True):
    keys = []
    for frame in frames:
        atoms, _, key = frame.split("!")
        atoms = [int(i) for i in atoms.split()[:-1]]
        key = key[2:-2].split(",")
        if key != ["hydrogen"]:
            key = [int(i) for i in key]
        else:
            key = [1]
        key = int(jnp.array(key).prod())
        keys.append(key)

    key_set = list(set(keys))
    key_set.sort()
    charges = []
    locals = []
    for ki, key in enumerate(keys):
        if key != 1:
            for i in range(1, 7):
                chg_key = key * i
                charges.append(chg_key)
            for i in range(1, 19):
                if atomCentered:
                    locals.append(1)
                else:
                    locals.append(key * i)
        else:
            for i in range(1, 7):
                charges.append(1)
            for i in range(1, 19):
                locals.append(1)


    all_charge_type_set = list(set(charges))
    all_charge_type_set.sort()
    chg_typ_idx = jnp.array([all_charge_type_set.index(i) for i in charges])

    all_local_type_set = list(set(locals))
    all_local_type_set.sort()
    local_typ_idx = jnp.array([all_local_type_set.index(i) for i in locals])

    return len(all_charge_type_set), \
        chg_typ_idx, \
        len(all_local_type_set), \
        local_typ_idx
