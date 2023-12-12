"""
Frames 
local to global
global to local
"""
import jax
import jax.numpy as jnp
from jax import jit


# Constants
eps = 1e-12  # A small epsilon value

@jit
def compute_local_axes(X, Y, Z,
                       ATM1, ATM2, ATM3,
                       FR):
    # assign arrays
    ATM1 = ATM1 - 1
    ATM2 = ATM2 - 1
    ATM3 = ATM3 - 1

    B1X = X[ATM1] - X[ATM2]
    B1Y = Y[ATM1] - Y[ATM2]
    B1Z = Z[ATM1] - Z[ATM2]
    RB12 = B1X ** 2 + B1Y ** 2 + B1Z ** 2
    RB1 = jnp.sqrt(RB12)

    # Normalize the local z-axis
    ZEI = B1X / RB1
    ZEJ = B1Y / RB1
    ZEK = B1Z / RB1

    ZEI = ZEI.repeat(3, axis=0)
    ZEJ = ZEJ.repeat(3, axis=0)
    ZEK = ZEK.repeat(3, axis=0)

    # Compute local y-axis

    B2X = X[ATM3] - X[ATM2]
    B2Y = Y[ATM3] - Y[ATM2]
    B2Z = Z[ATM3] - Z[ATM2]
    RB22 = B2X ** 2 + B2Y ** 2 + B2Z ** 2
    RB2 = jnp.sqrt(RB22)

    ZEI = ZEI.at[2].set(B2X / RB2)
    ZEJ = ZEJ.at[2].set(B2Y / RB2)
    ZEK = ZEK.at[2].set(B2Z / RB2)

    # Compute local y-axis
    FAC1 = -B1Z * B2Y + B1Y * B2Z
    FAC2 = B1Z * B2X - B2Z * B1X
    FAC3 = -B1Y * B2X + B2Y * B1X
    REY2 = FAC1 ** 2 + FAC2 ** 2 + FAC3 ** 2
    REY = jnp.sqrt(REY2)

    YEI = FAC1 / REY
    YEJ = FAC2 / REY
    YEK = FAC3 / REY

    YEI = YEI.repeat(3, axis=0)
    YEJ = YEJ.repeat(3, axis=0)
    YEK = YEK.repeat(3, axis=0)

    # Compute local x-axis
    XEI = YEK * ZEJ - YEJ * ZEK
    XEJ = YEI * ZEK - YEK * ZEI
    XEK = YEJ * ZEI - YEI * ZEJ

    return jnp.stack([
        jnp.stack([XEI, XEJ, XEK]),
        jnp.stack([YEI, YEJ, YEK]),
        jnp.stack([ZEI, ZEJ, ZEK]),
        ])



@jit
def conv_clcl_cxyz(
        fatom_pos: jnp.ndarray,
        fatom_elcl: jnp.ndarray,
        mdcm_clcl: jnp.ndarray,
        charge: jnp.ndarray,
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
    result = mdcm_clcl[:,:,jnp.newaxis,:] * fatom_elcl.T[None,:,:,:]
    # sum over the local axes
    result = result.sum(axis=3) + fatom_pos[None,:,:]
    # transpose to (natm,3)
    result = result.transpose(1,0,2).reshape(-1,3)
    # flatten the charges
    charge = charge.flatten()
    # stack the results
    final = jnp.zeros((result.shape[0], 4))
    final = final.at[:,:-1].set(result)
    final = final.at[:, -1].set(charge)
    jax.debug.print("{x}", x=final)
    print("""0.0000   -0.2910    1.1197    0.1979
0.0000    0.2910    1.2251    0.1055
-0.2910   -0.0065   -0.0004   -0.3034
0.2910   -0.0065   -0.0004   -0.3034
0.0000    1.0110   -0.5623    0.1979
0.0000    1.2589   -0.0251    0.1055""")
    return final
