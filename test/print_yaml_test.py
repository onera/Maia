import os
from maia.utils.yaml.pretty_print import pretty_print
from maia.utils import test_utils as TU


# TODO move to doc? (used by generated images in the doc)
with open(os.path.join(TU.mesh_dir,"../sample_meshes/4_cubes_lite.yaml")) as yt:
  pretty_print(yt)

with open(os.path.join(TU.mesh_dir,"../sample_meshes/4_cubes_lite_dist_0.yaml")) as yt:
  pretty_print(yt)

with open(os.path.join(TU.mesh_dir,"../sample_meshes/4_cubes_lite_dist_1.yaml")) as yt:
  pretty_print(yt)
