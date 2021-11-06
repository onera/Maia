from mpi4py import MPI
comm = MPI.COMM_WORLD

import maia.cgns_io.cgns_io_tree as L
import maia.utils.parse_cgns_yaml as P

t = L.file_to_dist_tree("hex_2_prism_2.cgns",comm)
s = P.to_yaml(t)

with open("hex_2_prism_2_dist_"+str(comm.Get_rank())+".yaml",'w') as f:
  for l in s:
    f.write(l+'\n')
