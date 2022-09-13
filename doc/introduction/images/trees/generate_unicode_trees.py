from maia.pytree.yaml.pretty_print import pretty_print
from pathlib import Path
import os

this_script_dir = Path(__file__).resolve().parent

with open(os.path.join(this_script_dir,"4_cubes_lite.yaml")) as yt:
  pretty_print(yt)

with open(os.path.join(this_script_dir,"4_cubes_lite_dist_0.yaml")) as yt:
  pretty_print(yt)

with open(os.path.join(this_script_dir,"4_cubes_lite_dist_1.yaml")) as yt:
  pretty_print(yt)
