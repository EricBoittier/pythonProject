import jax
from jax import jit
import jax.numpy as jnp
import equinox as eqx
from functools import partial

from project.mdcm import \
    compute_esp_multi, \
    calc_global_pos


class mdcm_eqx(eqx.Module):
    all_atoms: jnp.array
    all_coords: jnp.array
    all_grids: jnp.array
    all_esp: jnp.array
    chg_idx: jnp.array
    grid_idx: jnp.array
    charges: jnp.array
    local_typ_idx: jnp.array
    chg_typ_idx: jnp.array
    n_all_frames: int
    nchgparm: int
    nlocalparm: int
    n_charges: int

    def __init__(self,
                 all_atoms,
                 all_coords,
                 all_grids,
                 all_esp,
                 chg_idx,
                 grid_idx,
                 charges,
                 chg_typ_idx,
                 local_typ_idx,
                 n_all_frames,
                 nchgparm,
                 nlocalparm,
                 n_charges
                 ):
        self.all_esp = all_esp
        self.all_atoms = all_atoms
        self.all_coords = all_coords
        self.all_grids = all_grids
        self.chg_idx = chg_idx
        self.grid_idx = grid_idx
        self.charges = charges
        self.chg_typ_idx = chg_typ_idx
        self.local_typ_idx = local_typ_idx
        self.n_all_frames = n_all_frames
        self.nchgparm = nchgparm
        self.nlocalparm = nlocalparm
        self.n_charges = n_charges

    @jit
    def constrain_charges_multi(self, x0):
        """Constrain the charges to sum to zero."""
        segment_sum = jax.ops.segment_sum(x0 * self.charges, self.chg_idx,
                                          num_segments=2) / self.n_charges
        x0 = x0.at[:].add(-1 * segment_sum[self.chg_idx]) * self.charges
        jax.debug.print("sum(x0)={sum}", sum=x0.sum())
        return x0

    @jit
    def take_chg_parms(self, x0):
        return jnp.take(x0, self.chg_typ_idx)

    @partial(jit, static_argnums=(0, 2, 3, 4))
    def take_local_parms(self, x0, local_typ_idx: int, nchgparm: int, n_all_frames: int):
        x0_local = jnp.take(x0, local_typ_idx + nchgparm)
        x0_local = jnp.clip(x0_local, -0.173, 0.173)
        x0_local = x0_local.reshape(n_all_frames, 3, 2, 3)
        return x0_local

    @jit
    def take_local_parms_only(self, x0):
        return jnp.take(x0, self.local_typ_idx)

    def get_loss_charge_local(self):
        @partial(jit, static_argnums=(1,2,3,4,5,6,7,8,9))
        def loss_charge_local(x0,
                              local_typ_idx: int,
                              nchgparm: int,
                              n_all_frames: int,
                              all_atoms: jnp.array,
                                all_coords: jnp.array,
                                all_grids: jnp.array,
                                all_esp: jnp.array,
                                chg_idx: jnp.array,
                                grid_idx: jnp.array,
                              ) -> float:

            x0_chg = self.take_chg_parms(x0)
            x0_chg = self.constrain_charges_multi(x0_chg)
            x0_local = self.take_local_parms(x0,
                                             local_typ_idx,
                                             nchgparm,
                                             n_all_frames
                                             )

            positions = calc_global_pos(all_atoms,
                                        all_coords,
                                        x0_local)
            esp = compute_esp_multi(positions, x0_chg,
                                    all_grids,
                                    chg_idx,
                                    grid_idx)
            loss = jnp.sum((esp - all_esp) ** 2)
            jax.debug.print(".loss={loss}", loss=loss)
            return loss / 10 ** 6
        return loss_charge_local
