from maia.utils import parse_yaml_cgns
from maia.utils.yaml.pretty_print import pretty_print
import Converter.PyTree as C
import numpy as np

with open("/scratchm/bberthou/travail/git_all_projects/external/maia/share/meshes/octree.yaml") as yt:
  #pretty_print(yt)
  t = parse_yaml_cgns.to_pytree(yt)
  C.convertPyTree2File(t,"octree.cgns")



