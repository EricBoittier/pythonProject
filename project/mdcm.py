import jax
import jax.numpy as jnp
from jax import jit

coloumns_constant = 3.32063711e2 / 627.509469
# Constants
eps = 1e-12  # A small epsilon value


# @jit
def compute_local_axes(XYZ):
    # assign arrays
    B1 = XYZ[0] - XYZ[1]
    RB1 = jnp.linalg.norm(B1)
    B2 = XYZ[2] - XYZ[1]
    RB2 = jnp.linalg.norm(B2)
    # local z-axis
    eZ = jnp.repeat(B1 / RB1, 3).reshape(3, -1)
    eZ = eZ.at[:,2].set(B2 / RB2)
    # local y-axis
    FAC = jnp.cross(B1, B2)
    REY = jnp.linalg.norm(FAC)
    eY = jnp.repeat(FAC / REY, 3).reshape(3, -1)
    eX = jnp.cross(eZ.T, eY.T).T
    # local z-axis
    output = jnp.array([eX, eY, eZ])
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
    # transpose mdcm_clcl so that it is (3,3,natm)
    mdcm_clcl = jnp.transpose(mdcm_clcl, (1, 0, 2))
    # multiply the local charge centers by the local axes
    result = mdcm_clcl[:, :, jnp.newaxis, :] * fatom_elcl.T[None, :, :, :]
    # sum over the local axes
    result = result.sum(axis=3) + fatom_pos[None, :, :]
    # transpose to (natm,3)
    result = result.transpose(1, 0, 2).reshape(-1, 3)
    return result

@jit
def coulomb(q1, q2, r12):
    return coloumns_constant * (q1 * q2 / r12)

@jit
def compute_esp(positions, charges, grid_points):
    esp = jnp.zeros_like(grid_points.shape[0])
    for i in range(len(charges)):
        distances = jnp.linalg.norm(grid_points - positions[i], axis=1)
        esp += coulomb(charges[i,None], -1.0, distances)
    return esp