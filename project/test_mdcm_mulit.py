from project.mdcm import compute_esp_multi, calc_global_pos, process_frames
from project.rdkit_ import get_pdb_data
from project.psi4_ import read_grid, read_ref_esp
from project.mdcm_io import read_charmm_mdcm, write_dcm_xyz
from jax import random
import jax.numpy as jnp
from pathlib import Path
from jax.scipy import optimize
import jax
from project.mdcm_multi import MDCMoptMulti

mac = False
testGLY = False
testTIP3 = True
if testGLY:
    pdb_path = Path("/home/boittier/Documents/phd/pythonProject/pdb")
    if mac:
        pdb_path = Path("/Users/ericboittier/Documents/github/pythonProject/pdb")

    pdb1 = pdb_path / "gly-70150091-70a0-453f-b6b8-c5389f387e84-end.pdb"
    pdb2 = pdb_path / "gly-70150091-70a0-453f-b6b8-c5389f387e84-min.pdb"

    glu = "/home/boittier/Documents/phd/pythonProject/mdcm/gen/GLY.mdcm"
    if mac:
        glu = '/Users/ericboittier/Documents/github/pythonProject/mdcm/gen/GLY.mdcm'

    m = MDCMoptMulti(pdbs=[pdb1, pdb2], mdcms=[glu, glu])
    loss = m.get_chg_local_loss()
    res = optimize.minimize(loss,
                            m.init_x0(),
                            method='BFGS',
                            tol=1e-1,)

    print(res.x)
    print(res)
    pos, chgs, esp = m.get_pos_chgs_esp(res.x)
    print(pos)
    print(chgs)

    write_dcm_xyz("dcm_test.xyz", pos, chgs)

    for i in range(len(m.all_ref_esp)):
        if i % 1000:
            print(i, m.all_ref_esp[i], esp[i], m.all_grids[i])

if testTIP3:
    pdb_path = Path("/home/boittier/Documents/phd/pythonProject/pdb")
    if mac:
        pdb_path = Path("/Users/ericboittier/Documents/github/pythonProject/pdb")


    pdb1 = pdb_path / "test.pdb"
    glu = "/home/boittier/Documents/phd/pythonProject/mdcm/gen/TIP3.mdcm"
    if mac:
        glu = '/Users/ericboittier/Documents/github/pythonProject/mdcm/gen/GLY.mdcm'

    m = MDCMoptMulti(pdbs=[pdb1], mdcms=[glu])

    loss = m.get_chg_local_loss()
    x0 = m.init_x0()
    print("parms:", x0)
    # print("loss:", loss(x0))
    parms = jnp.array([0.197911, 0.105500, -0.303411,
                       -0.303411, 0.197911, 0.105500,
                       0.291047,  0.000000,  0.149694,  -0.291047,  0.000000,
                       0.255090, 0.006537,  0.291047,  -0.000410, 0.006537,
                       -0.291047,  -0.000410, -0.291047,  0.000000,  0.149694,
                       0.291047,  0.000000,  0.255090])
    print(parms)
    print("\n\nparms loss:", loss(parms))

    res = optimize.minimize(loss,
                            x0,
                            method='BFGS',
                            tol=1e-1,)

    print(res.x)
    print(res)
    print(sum(res.x))
    pos, chgs, esp = m.get_pos_chgs_esp(res.x)
    print(sum(chgs))
    print("positions", pos)
    print("charges", chgs)
    #
    write_dcm_xyz("dcm_test.xyz", pos, chgs)

    for i in range(len(m.all_ref_esp)):
        if i % 100 == 0:
            print(i, m.all_ref_esp[i], esp[i], m.all_ref_esp[i]/esp[i], m.all_grids[i])