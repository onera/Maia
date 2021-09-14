from mpi4py import MPI
comm = MPI.COMM_WORLD

from maia.utils import mesh_dir
from maia.utils import parse_yaml_cgns
import Converter.PyTree as C

for mesh_name in ["cube_1","cube_2","cube_4","cube_4_part_2","hex_prism_pyra_tet"]:
  with open(mesh_dir+mesh_name+".yaml") as yt:
    t = parse_yaml_cgns.to_nodes(yt)
    C.convertPyTree2File(t,mesh_name+".cgns")
