from project.mdcm import compute_esp, calc_global_pos, process_frames
import jax
import jax.numpy as jnp
from jax import jit


class MDCMopt():
    def __init__(self,
                 charges=None,
                 frameAtoms=None,
                 coords=None,
                 surface_points=None,
                 reference_esp=None,
                 frames=None,
                 locals=None,
                 atomCentered=True,
                 ):

        self.locals = jnp.array(locals)
        print("locals", self.locals.shape)
        self.charges = charges
        self.coords = coords
        self.frameAtoms = frameAtoms
        self.positions = calc_global_pos(self.frameAtoms,
                                         self.coords,
                                         locals)
        self.surface_points = surface_points
        self.reference_esp = jnp.array(reference_esp)
        self.frames = frames
        self.atomCentered = atomCentered
        self.Nchgparm, self.chg_typ_idx, \
            self.Nlocalparm, self.local_typ_idx = process_frames(
            frames, atomCentered=atomCentered
        )
        print("self.Nchgparm", self.Nchgparm)
        print("self.Nlocalparm", self.Nlocalparm)
        print("self.frames", self.frames)

    @jit
    def constrain_charges(self, x0):
        return x0 - ((self.charges * x0.sum()) / self.charges.sum())

    def get_N_params(self) -> int:
        if self.atomCentered:
            return self.Nchgparm
        else:
            return self.Nchgparm + self.Nlocalparm

    def get_charges_loss(self):
        @jit
        def charges_loss(x0):
            @jit
            def constrain_charges(x0):
                return x0 - ((self.charges * x0.sum()) / self.charges.sum())

            x0 = jnp.take(x0, self.chg_typ_idx)
            _charges = self.charges * x0
            _charges = constrain_charges(_charges)
            esp = compute_esp(
                self.positions, _charges, self.surface_points
            )
            error = (esp - jnp.array(self.reference_esp)) ** 2
            MSE = (error.sum()) / len(esp)
            jax.debug.print("{x}", x=MSE)
            return MSE

        return charges_loss

    def get_charges_local_loss(self):
        @jit
        def charges_loss(x0):
            @jit
            def constrain_charges(x0):
                return x0 - ((self.charges * x0.sum()) / self.charges.sum())

            x0 = x0.at[self.Nchgparm].set(0.0)
            charges = jnp.take(x0, self.chg_typ_idx)
            _charges = self.charges * charges
            _charges = constrain_charges(_charges)

            _locals = jnp.take(x0, self.local_typ_idx + self.Nchgparm)
            # constrain locals to be +/- 0.3
            _locals = jnp.clip(_locals, -0.173, 0.173)
            _locals = _locals.reshape(len(self.frames), 3, 2, 3)
            positions = calc_global_pos(self.frameAtoms, self.coords, _locals)

            #  calculate the ESP
            esp = compute_esp(
                positions, _charges, self.surface_points
            )

            error = (esp - jnp.array(self.reference_esp)) ** 2
            MSE = (error.sum()) / len(esp)
            jax.debug.print("{x}", x=MSE * 627.509469**2)
            return MSE / jnp.linalg.norm(
                jnp.array([1000, 1000, 1000]))

        return charges_loss

    def get_constraint(self):
        @jit
        def constraint(x0):
            return x0 - ((self.charges * x0.sum()) / self.charges.sum())

        return constraint


def print_loss(esp,
               reference_esp):
    # esp *= 627.509469
    reference_esp = jnp.array(reference_esp) #* 627.509469
    error = (esp - jnp.array(reference_esp)) ** 2
    MSE = (error.sum()) / len(esp)
    RMSE = jnp.sqrt(error.sum()) / len(esp)
    MAE = jnp.abs((esp - jnp.array(reference_esp))).sum() / len(esp)
    MAXERROR = jnp.abs(error).max()
    print("Npoints", len(esp))
    print("MSE", MSE * 627.509469**2)
    print("RMSE", RMSE * 627.509469)
    print("MAE", MAE * 627.509469)
    print("MAXERROR", MAXERROR * 627.509469)
    for i in range(len(esp)):
        if i % 10 == 0:
            print(i, esp[i], reference_esp[i],
                  (esp[i] - reference_esp[i])*627.509469)
    return MSE
