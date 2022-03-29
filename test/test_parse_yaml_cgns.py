from maia.utils import parse_yaml_cgns
import Converter.Internal as I
import numpy as np
import os
from maia.utils import test_utils as TU

with open(os.path.join(TU.mesh_dir,"cube_4.yaml")) as yt:
  t = parse_yaml_cgns.to_nodes(yt)

  I.printTree(t)
