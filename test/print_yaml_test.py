import os
from maia.utils.yaml.pretty_print import pretty_print

dir_path = os.path.dirname(os.path.realpath(__file__))
mesh_dir = dir_path+"/../share/sample_meshes/"

with open(mesh_dir+"4_cubes_lite.yaml") as yt:
  pretty_print(yt)

with open(mesh_dir+"4_cubes_lite_dist_0.yaml") as yt:
  pretty_print(yt)

with open(mesh_dir+"4_cubes_lite_dist_1.yaml") as yt:
  pretty_print(yt)
