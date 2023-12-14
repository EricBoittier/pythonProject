from project.mdcm import compute_esp, calc_global_pos, process_frames
import jax
import jax.numpy as jnp
from jax import jit


class MDCMopt():
    def __init__(self,
                 charges=None,
                 positions=None,
                 surface_points=None,
                 reference_esp=None,
                 frames=None,
                 ):
        self.charges = charges
        self.positions = positions
        self.surface_points = surface_points
        self.reference_esp = jnp.array(reference_esp)
        self.frames = frames
        self.Nchgparm, self.chg_typ_idx = process_frames(frames)

        print("self.Nchgparm", self.Nchgparm)
        print("self.frames", self.frames)

    @jit
    def constrain_charges(self, x0):
        return x0 - ((self.charges * x0.sum()) / self.charges.sum())

    @jit
    def charges_loss(self, x0):
        x0 = jnp.take(x0, self.chg_typ_idx)
        _charges = self.charges * x0
        _charges = self.constrain_charges(_charges)
        esp = compute_esp(self.positions, _charges, self.surface_points) * 627.509469
        error = (esp - jnp.array(self.reference_esp* 627.509469)) ** 2
        MSE = (error.sum()) / len(esp)
        jax.debug.print("{x}", x=MSE)
        return MSE

    def get_charges_loss(self):
        @jit
        def charges_loss(x0):
            @jit
            def constrain_charges(x0):
                return x0 - ((self.charges * x0.sum()) / self.charges.sum())

            x0 = jnp.take(x0, self.chg_typ_idx)
            _charges = self.charges * x0
            _charges = constrain_charges(_charges)
            esp = compute_esp(self.positions, _charges, self.surface_points) * 627.509469
            error = (esp - jnp.array(self.reference_esp* 627.509469)) ** 2
            MSE = (error.sum()) / len(esp)
            # MAE = jnp.abs((esp - jnp.array(self.reference_esp* 627.509469))).sum() / len(esp)
            jax.debug.print("{x}", x=MSE)
            # jax.debug.print("{x}", x=MAE)
            return MSE

        return charges_loss

    def get_constraint(self):
        @jit
        def constraint(x0):
            return x0 - ((self.charges * x0.sum()) / self.charges.sum())

        return constraint


def print_loss(esp,
               reference_esp):
    esp *= 627.509469
    reference_esp = jnp.array(reference_esp) * 627.509469
    error = (esp - jnp.array(reference_esp)) ** 2
    MSE = (error.sum()) / len(esp)
    RMSE = jnp.sqrt(error.sum()) / len(esp)
    MAE = jnp.abs((esp - jnp.array(reference_esp))).sum() / len(esp)
    MAXERROR = jnp.abs(error).max()
    print("MSE", MSE)
    print("RMSE", RMSE)
    print("MAE", MAE)
    print("MAXERROR", MAXERROR)
    for i in range(len(esp)):
        if i % 100 == 0:
            print(i, esp[i], reference_esp[i], esp[i] - reference_esp[i])
    return MSE
