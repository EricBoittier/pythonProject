import blender_plots as bplt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import ast
import bpy
import blf
import mathutils
import math

from pathlib import Path

data_path = Path("/Users/ericboittier/Documents/github/pythonProject")

#  read data
grid = np.genfromtxt(data_path / "grid.dat")
grid = grid.astype(np.float32)
# grid = grid * 0.529177249

#  read esp
esp = np.genfromtxt(data_path / "grid_esp.dat")

print(esp.shape)
print(grid.shape)
print(esp.min(), esp.max())
print(grid.min(), grid.max())
print(esp)
print(grid)

# convert esp to colors using red to blue colormap
cmap = cm.get_cmap('bwr')
norm = plt.Normalize(vmin=esp.min(), vmax=esp.max())
norm = plt.Normalize(vmin=-0.1, vmax=0.1)
colors = cmap(norm(esp))
colors = colors[:, :3]

#  remove old objects
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

#  add points
bplt.Scatter(
        grid,
        color=colors,
        marker_type="uv_spheres",
        radius=0.1235,
        name="points",
    )

#  molecule from Molecular Nodes
import MolecularNodes as mn
print(mn)

style = "ball_and_stick"
pdb_path = Path("/Volumes/Extreme SSD/data/aa/pdb")
pdb_files = list(pdb_path.glob("*.pdb"))
pdb_files = ["/Volumes/Extreme SSD/data/aa/pdb/initial-test-79786930-01c3-4eb1-8484-aa3d374e5c0d.pdb"]
pdb_files = [str(p) for p in pdb_files]
mn.load.molecule_local(pdb_files[0], default_style=2)


#  scale molecule
bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["Name"].select_set(True)
bpy.ops.transform.resize(value=(100., 100., 100.))


#  add camera
bpy.ops.object.camera_add(
        enter_editmode=False,
        align="VIEW",
        location=(0, 0, 0),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
# set camera to orthographic
bpy.data.cameras["Camera"].type = "ORTHO"
#  set camera to orthographic scale
bpy.data.cameras["Camera"].ortho_scale = 1.5
#  make active camera
bpy.context.scene.camera = bpy.data.objects["Camera"]

#  set white background
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (
        1,
        1,
        1,
        1,
    )


bpy.context.scene.render.resolution_x = 1256  # perhaps set resolution in code
bpy.context.scene.render.resolution_y = 1256
bpy.context.scene.render.engine = "CYCLES"

#  set camera to fit all points
bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["points"].select_set(True)
bpy.ops.view3d.camera_to_view_selected()


bpy.ops.render.render()
bpy.data.images["Render Result"].save_render(f"/Users/ericboittier/Downloads/test.png")
