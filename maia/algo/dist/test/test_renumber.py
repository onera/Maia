import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import par_utils

from maia.algo.dist import renumber as RENUM

@pytest_parallel.mark.parallel(2)
def test_renumber_vertices(comm):
  tree1 = maia.factory.generate_dist_block([3,2], 'S', comm, origin=[0,0])
  tree2 = maia.factory.generate_dist_block([3,2], 'S', comm, origin=[1,0])
  zone1 = PT.get_all_Zone_t(tree1)[0]
  zone2 = PT.get_all_Zone_t(tree2)[0]
  ztype = PT.get_np_value(zone1).dtype
  PT.set_name(zone1, 'Left')
  PT.set_name(zone2, 'Right')
  PT.rm_nodes_from_name(zone1, 'Xmax')
  PT.rm_nodes_from_name(zone2, 'Xmin')
  # Add jns
  pr  = np.array([[3,3], [1,2]], ztype)
  prd = np.array([[1,1], [1,2]], ztype)
  distri = par_utils.uniform_distribution(1*2, comm)
  jn = PT.new_GridConnectivity1to1(donor_name='Right', point_range=pr, point_range_donor=prd,
                                   transform=[1,2],
                                   parent=PT.new_ZoneGridConnectivity(parent=zone1))
  MT.new_Distribution({'Index' : distri}, jn)
  jn = PT.new_GridConnectivity1to1(donor_name='Base/Left', point_range=prd, point_range_donor=pr,
                                   transform=[1,2],
                                   parent=PT.new_ZoneGridConnectivity(parent=zone2))
  MT.new_Distribution({'Index' : distri}, jn)
  tree = PT.union(tree1, tree2)

  maia.algo.dist.convert_s_to_u(tree, 'Standard', comm)
  
  vtx_distri = MT.distribution_value(PT.find_node_from_name(tree, 'Left'), 'Vertex')
  new_vtx_id = np.array([5,4,3,2,1,0])[vtx_distri[0]:vtx_distri[1]]

  # For Right zone, only GC should be modified
  expt_zone2 = PT.deep_copy(PT.find_node_from_name(tree, 'Right'))
  gc = PT.find_node_from_name(expt_zone2, 'GC')
  distri = MT.distribution_value(gc, 'Index')
  pld = PT.get_np_value(PT.find_child_from_name(gc, 'PointListDonor'))
  pld[0,:] = np.array([4,1])[distri[0]:distri[1]]

  # For Left zone, vertices should be reordered
  zt = 'I8' if PT.get_np_value(expt_zone2).dtype == np.int64 else 'I4'
  expt_zone1_f = PT.yaml.to_node(f"""
  Left Zone_t {zt} [[6, 2, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [1.0, 0.5, 0.0, 1.0, 0.5, 0.0]:
      CoordinateY DataArray_t R8 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]:
    ZoneBC ZoneBC_t:
      Xmin BC_t 'Null':
        GridLocation GridLocation_t 'Vertex':
        PointList IndexArray_t {zt} [[6, 3]]:
      Ymin BC_t 'Null':
        GridLocation GridLocation_t 'Vertex':
        PointList IndexArray_t {zt} [[6, 5, 4]]:
      Ymax BC_t 'Null':
        GridLocation GridLocation_t 'Vertex':
        PointList IndexArray_t {zt} [[3, 2, 1]]:
    ZoneGridConnectivity ZoneGridConnectivity_t:
      GC GridConnectivity_t 'Right':
        GridConnectivityType GridConnectivityType_t 'Abutting1to1':
        GridLocation GridLocation_t 'Vertex':
        GridConnectivityDonorName Descriptor_t 'GC':
        PointListDonor IndexArray_t {zt} [[1, 4]]:
        PointList IndexArray_t {zt} [[4, 1]]:
    QUAD_4 Elements_t I4 [7, 0]:
      ElementRange IndexRange_t {zt} [7, 8]:
      ElementConnectivity DataArray_t {zt} [6, 5, 2, 3, 5, 4, 1, 2]:
    BAR_2 Elements_t I4 [3, 0]:
      ElementRange IndexRange_t {zt} [1, 6]:
      ElementConnectivity DataArray_t {zt} [3, 6, 4, 1, 6, 5, 5, 4, 2, 3, 1, 2]:
  """)
  expt_zone1 = maia.factory.full_to_dist_tree(expt_zone1_f, comm)
  # RM Distri/ElementConnectivity for comparaison
  for elt in PT.get_nodes_from_label(expt_zone1, 'Elements_t'):
    PT.rm_node_from_path(elt, ':CGNS#Distribution/ElementConnectivity')


  RENUM.renumber_vertices(tree, 'Base/Left', new_vtx_id, comm)

  assert PT.is_same_tree(PT.find_node_from_name(tree, 'Left'), expt_zone1)
  assert PT.is_same_tree(PT.find_node_from_name(tree, 'Right'), expt_zone2)