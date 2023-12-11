from cubes_ import cube
import numpy as np
cube_file = "../cubes/gaussian/testjax.chk.d.cube"

c = cube(cube_file)
print(c.NX, c.NY, c.NZ)
print(c.origin)
print(c.X, c.Y, c.Z)
print(c.atoms)
print(c.atomsXYZ)
grid = c.get_grid()
print(grid.shape)


