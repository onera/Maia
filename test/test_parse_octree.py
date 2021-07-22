from maia.utils import parse_yaml_cgns
from maia.utils.yaml.pretty_print import pretty_print
import Converter.PyTree as C
import numpy as np
from maia.utils import mesh_dir

with open(mesh_dir+"octree_3_8_part_2.yaml") as yt:
  #pretty_print(yt)
  t = parse_yaml_cgns.to_nodes(yt)
  C.convertPyTree2File(t,"octree_3_8_part_2.cgns")
