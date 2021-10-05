import os
from maia.utils import parse_yaml_cgns
from maia.utils.yaml.pretty_print import pretty_print
import Converter.PyTree as C
import numpy as np
from maia.utils.test_utils import mesh_dir

sample_mesh_dir = mesh_dir + '/../sample_meshes'
with open(os.path.join(sample_mesh_dir,"octree_3_8_part_2.yaml")) as yt:
  #pretty_print(yt)
  t = parse_yaml_cgns.to_nodes(yt)
  C.convertPyTree2File(t,"octree_3_8_part_2.cgns")
