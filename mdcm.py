"""
Frames 
local to global
global to local
"""

import jax.numpy as jnp
from jax import jit
# import numpy as jnp
from typing import Tuple

# Constants
eps = 1e-12  # A small epsilon value


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

    print(ZEI)
    print(ZEJ)
    print(ZEK)
    print(YEI)
    print(YEJ)
    print(YEK)
    print(XEI)
    print(XEJ)
    print(XEK)

    return jnp.stack([

        jnp.stack([XEI, XEJ, XEK]),
        jnp.stack([YEI, YEJ, YEK]),
        jnp.stack([ZEI, ZEJ, ZEK]),

    ]
    )


# Constants
eps = 1e-12  # A small epsilon value


def conv_clcl_cxyz(
        fatom_pos: jnp.ndarray,
        fatom_elcl: jnp.ndarray,
        Nqdim: int,
        mdcm_afrm: jnp.ndarray,
        mdcm_nchg: jnp.ndarray,
        mdcm_clcl: jnp.ndarray
) -> jnp.ndarray:
    """

    :param fatom_pos:
    :param fatom_elcl:
    :param Nqdim:
    :param mdcm_afrm:
    :param mdcm_nchg:
    :param mdcm_clcl:
    :return:
    """
    nframes = 1
    fatom_cxyz = jnp.zeros(Nqdim * 4)
    l_count = 0
    # loop over frames
    for i in range(nframes):
        # loop over atoms in frame
        for j, frAt in enumerate(mdcm_afrm[1, :]):
            for chg in range(mdcm_nchg[frAt][0]):
                e_frame = fatom_elcl[:, :, j]
                a = mdcm_clcl[l_count]
                b = mdcm_clcl[l_count + 1]
                c = mdcm_clcl[l_count + 2]
                TX = a * e_frame[0, 0] + b * e_frame[1, 0] + c * e_frame[2, 0]
                TY = a * e_frame[0, 1] + b * e_frame[1, 1] + c * e_frame[2, 1]
                TZ = a * e_frame[0, 2] + b * e_frame[1, 2] + c * e_frame[2, 2]
                print(fatom_pos[frAt - 1] + jnp.array([TX, TY, TZ]))
                l_count += 4

    print(fatom_cxyz)

    return fatom_cxyz
