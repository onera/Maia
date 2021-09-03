import os
from maia.utils.yaml.pretty_print import pretty_print

from maia.utils import mesh_dir

with open(mesh_dir+"4_cubes_lite.yaml") as yt:
  pretty_print(yt)

with open(mesh_dir+"4_cubes_lite_dist_0.yaml") as yt:
  pretty_print(yt)

with open(mesh_dir+"4_cubes_lite_dist_1.yaml") as yt:
  pretty_print(yt)
