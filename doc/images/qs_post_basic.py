import numpy as np
import maia
import maia.pytree as PT

tree = maia.io.read_tree('U_twoblocks.cgns')

i_bc = 0
for zone in PT.get_all_Zone_t(tree):
  face_flag = -1*np.ones(PT.Zone.n_face(zone), dtype=int)
  for bc in PT.get_nodes_from_label(zone, 'BC_t'):
    pl = PT.get_node_from_name(bc, 'PointList')[1][0]
    face_flag[pl-1] = i_bc
    i_bc += 1
  PT.new_FlowSolution('FS', loc='FaceCenter', fields={'FaceFlag': face_flag}, parent=zone)

maia.io.write_tree(tree, 'U_twoblocks2.cgns')

