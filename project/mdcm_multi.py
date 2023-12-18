from project.mdcm import compute_esp, compute_esp_multi, calc_global_pos, process_frames
from project.mdcm_opt import MDCMopt, print_loss
from project.rdkit_ import get_water_data, get_pdb_data
from project.psi4_ import get_grid_points, read_grid, read_ref_esp
from project.mdcm_io import read_charmm_mdcm, write_dcm_xyz
from jax import random
import jax.numpy as jnp
from pathlib import Path
from jax.scipy import optimize
import jax
from jax import grad, jit, vmap
from functools import partial

pdb_path = Path("/home/boittier/Documents/phd/pythonProject/pdb")


class MDCMoptMulti:
    def __init__(self,
                 pdbs=None,
                 mdcms=None,
                 atomCentered=False,
                 ):
        self.random_key = random.PRNGKey(0)
        self.atomCentered = atomCentered
        self.pdbs = pdbs
        self.pdb_paths = [pdb_path / pdb for pdb in pdbs]
        self.n_pdbs = len(self.pdbs)
        self.pdb_data = [get_pdb_data(pdb) for pdb in self.pdb_paths]
        self.elements = [e for e, c in self.pdb_data]
        self.coords = [c for e, c in self.pdb_data]
        self.mdcms = [read_charmm_mdcm(m) for m in mdcms]
        self.frames = [f for f, a, s, c in self.mdcms]
        self.atoms = [a for f, a, s, c in self.mdcms]
        self.stackData = [s for f, a, s, c in self.mdcms]
        self.charges = [c for f, a, s, c in self.mdcms]
        self.n_charges = jnp.array([c.flatten().sum() for c
                                    in self.charges],
                                   dtype=jnp.int32)
        self.n_atoms = [len(e) for e in self.elements]
        self.n_frames = [len(f) for f in self.frames]
        self.n_all_frames = sum(self.n_frames)
        self.n_atoms_total = sum(self.n_atoms)
        self.n_frames_total = sum(self.n_frames)
        self.grids = [read_grid(p) for p in self.pdb_paths]
        self.ref_esps = [read_ref_esp(p) for p in self.pdb_paths]

        # append atoms together and increment indices if not 0
        self.all_atoms = []
        atoms_copy = self.atoms[0].copy()
        for i, at in enumerate(self.atoms):
            atoms_copy = jnp.where(atoms_copy > 0,
                                   atoms_copy + sum(self.n_atoms[0:i]),
                                   atoms_copy)
            self.all_atoms.append(atoms_copy)
        self.all_atoms = jnp.concatenate(self.all_atoms)
        # which atoms belong to which molecule
        self.atom_tuples = list(jnp.ones_like(a[:, 0]) * i
                                for i, a in enumerate(self.coords))
        self.atom_idx = jnp.concatenate(self.atom_tuples)
        # append the frame data together
        self.stackedData = jnp.concatenate(self.stackData)
        # which frames belong to which molecule
        self.frame_tuples = list(jnp.ones_like(s[:, :, :, 0]).flatten() * i
                                 for i, s in enumerate(self.stackData))
        self.chg_idx = jnp.concatenate(self.frame_tuples, dtype=jnp.int32)
        # append coords together
        self.all_coords = jnp.concatenate(self.coords)
        # append grids together
        self.all_grids = jnp.concatenate(self.grids)
        # append ref_esp together
        self.all_ref_esp = jnp.concatenate(self.ref_esps)
        # which points belong to which grid
        self.grid_tuples = list(jnp.ones_like(g) * i
                                for i, g in enumerate(self.ref_esps))
        self.grid_idx = jnp.concatenate(self.grid_tuples)
        # compute the global positions
        self.global_pos = calc_global_pos(self.all_atoms,
                                          self.all_coords,
                                          self.stackedData)
        # concat charges together
        self.charges = jnp.concatenate(self.charges)
        self.charges = self.charges.flatten()
        # concat list of lists to list
        self.frames = [item for sublist in self.frames for item in sublist]

        self.Nchgparm, self.chg_typ_idx, \
            self.Nlocalparm, self.local_typ_idx = process_frames(
            self.frames, atomCentered=self.atomCentered
        )

        self.Nparm = self.Nchgparm + self.Nlocalparm



    def init_x0_charges(self):
        return jax.random.uniform(self.random_key, (self.Nchgparm,),
                                  minval=-1.0, maxval=1.0)

    def init_x0_local(self):
        return jax.random.uniform(self.random_key, (self.Nlocalparm,),
                                  minval=-0.1, maxval=0.1)

    def init_x0(self):
        return jnp.concatenate([self.init_x0_charges(),
                                self.init_x0_local()])

    def constrain_charges_multi(self, x0, ):
        """Constrain the charges to sum to zero."""
        segment_sum = jax.ops.segment_sum(x0*self.charges, self.chg_idx,
                                          num_segments=2) / self.n_charges
        x0 = x0.at[:].add(-1 * segment_sum[self.chg_idx]) * self.charges
        return x0

    def take_chg_parms(self, x0):
        return jnp.take(x0, self.chg_typ_idx)

    def take_local_parms(self, x0):
        return jnp.take(x0, self.local_typ_idx + self.Nchgparm)

    def take_local_parms_only(self, x0):
        return jnp.take(x0, self.local_typ_idx)

    def opt_charge_local(self):
        res = optimize.minimize(self.loss_charge_local,
                                self.init_x0(),
                                method='BFGS', )
        print(res.x)
        print(res)
        return res.x

    def loss_charge_local(self, x0):
        x0_chg = self.take_chg_parms(x0)

        x0_chg = self.constrain_charges_multi(x0_chg)

        x0_local = self.take_local_parms(x0)
        x0_local = jnp.clip(x0_local, -0.173, 0.173)
        x0_local = x0_local.reshape(self.n_all_frames, 3, 2, 3)

        positions = calc_global_pos(self.all_atoms,
                                    self.all_coords,
                                    x0_local)

        esp = compute_esp_multi(positions, x0_chg,
                                self.all_grids,
                                self.chg_idx, self.grid_idx)

        loss = jnp.sum((esp - self.all_ref_esp) ** 2)

        # jax.debug.print(".loss={loss}", loss=loss)
        return loss


pdb1 = pdb_path / "gly-70150091-70a0-453f-b6b8-c5389f387e84-end.pdb"
pdb2 = pdb_path / "gly-70150091-70a0-453f-b6b8-c5389f387e84-min.pdb"
glu = "/home/boittier/Documents/phd/pythonProject/mdcm/gen/GLY.mdcm"

m = MDCMoptMulti(pdbs=[pdb1, pdb2], mdcms=[glu, glu])

from mdcm_eqx import mdcm_eqx

m_e = mdcm_eqx(
    m.all_atoms,
    m.all_coords,
    m.all_grids,
    m.all_ref_esp,
    m.chg_idx,
    m.grid_idx,
    m.charges,
    m.chg_typ_idx,
    m.local_typ_idx,
    m.n_all_frames,
    m.Nchgparm,
    m.Nlocalparm,
    m.n_charges,
)

# m_e.loss_charge_local(m.init_x0())

res = optimize.minimize(m_e.loss_charge_local,
                        m.init_x0(),
                        method='BFGS', )

print(res.x)
print(res)
