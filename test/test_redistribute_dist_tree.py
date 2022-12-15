from   pytest_mpi_check._decorator import mark_mpi_test
import pytest
import os
import numpy as np

import maia
import maia.io            as Mio
import maia.pytree        as PT
import maia.pytree.maia   as MT

from   maia.utils         import par_utils
from   maia.algo.dist     import redistribute as RDT
from   maia.pytree.yaml   import parse_yaml_cgns

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'


@mark_mpi_test([1, 2, 3])
def test_redistribute_zone_U(sub_comm, write_output):
  
  # Reference directory and file
  ref_dir  = os.path.join(os.path.dirname(__file__), 'references')
  ref_file = os.path.join(ref_dir, f'cube_bcdataset_and_periodic.yaml')

  dist_tree   = Mio.file_to_dist_tree(ref_file, sub_comm)
  gather_tree = dist_tree
  
  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.write_trees(dist_tree, os.path.join(out_dir, 'dist_tree.cgns'), sub_comm)

  for zone in PT.get_all_Zone_t(gather_tree):
    gather_zone = RDT.redistribute_zone(zone, par_utils.gathering_distribution, sub_comm)

  if write_output:
    Mio.write_trees(gather_tree, os.path.join(out_dir, 'gather_tree.cgns'), sub_comm)


  if sub_comm.Get_rank()==0:
    distri_to_check = {'Base/zone/NGonElements/:CGNS#Distribution/Element'                  : np.array([ 0,  36,  36]),
                       'Base/zone/NGonElements/:CGNS#Distribution/ElementConnectivity'      : np.array([ 0, 144, 144]), 
                       'Base/zone/NFaceElements/:CGNS#Distribution/Element'                 : np.array([ 0,   8,   8]),
                       'Base/zone/NFaceElements/:CGNS#Distribution/ElementConnectivity'     : np.array([ 0,  48,  48]), 
                       'Base/zone/:CGNS#Distribution/Vertex'                                : np.array([ 0,  27,  27]), 
                       'Base/zone/:CGNS#Distribution/Cell'                                  : np.array([ 0,   8,   8]), 
                       'Base/zone/ZoneBC/Xmin/:CGNS#Distribution/Index'                     : np.array([ 0,   4,   4]), 
                       'Base/zone/ZoneBC/Xmax/:CGNS#Distribution/Index'                     : np.array([ 0,   4,   4]), 
                       'Base/zone/ZoneBC/Ymin/:CGNS#Distribution/Index'                     : np.array([ 0,   4,   4]), 
                       'Base/zone/ZoneBC/Ymax/:CGNS#Distribution/Index'                     : np.array([ 0,   4,   4]), 
                       'Base/zone/ZoneGridConnectivity/Zmin_match/:CGNS#Distribution/Index' : np.array([ 0,   4,   4]), 
                       'Base/zone/ZoneGridConnectivity/Zmin_match/:CGNS#Distribution/Index' : np.array([ 0,   4,   4])
    }
    # Check distribution on full  ranks
    for key in distri_to_check:
      assert np.array_equal(PT.get_node_from_path(gather_tree, key)[1], distri_to_check[key])

    ref_tree = Mio.read_tree(ref_file)
    PT.rm_nodes_from_name(gather_tree, ':CGNS#Distribution')
    assert PT.is_same_tree(ref_tree[2][1], gather_tree[2][1])


  else :
    distri_to_check = {'Base/zone/NGonElements/:CGNS#Distribution/Element'                  : np.array([ 36,  36,  36]),
                       'Base/zone/NGonElements/:CGNS#Distribution/ElementConnectivity'      : np.array([144, 144, 144]), 
                       'Base/zone/NFaceElements/:CGNS#Distribution/Element'                 : np.array([  8,   8,   8]),
                       'Base/zone/NFaceElements/:CGNS#Distribution/ElementConnectivity'     : np.array([ 48,  48,  48]), 
                       'Base/zone/:CGNS#Distribution/Vertex'                                : np.array([ 27,  27,  27]), 
                       'Base/zone/:CGNS#Distribution/Cell'                                  : np.array([  8,   8,   8]), 
                       'Base/zone/ZoneBC/Xmin/:CGNS#Distribution/Index'                     : np.array([  4,   4,   4]), 
                       'Base/zone/ZoneBC/Xmax/:CGNS#Distribution/Index'                     : np.array([  4,   4,   4]), 
                       'Base/zone/ZoneBC/Ymin/:CGNS#Distribution/Index'                     : np.array([  4,   4,   4]), 
                       'Base/zone/ZoneBC/Ymax/:CGNS#Distribution/Index'                     : np.array([  4,   4,   4]), 
                       'Base/zone/ZoneGridConnectivity/Zmin_match/:CGNS#Distribution/Index' : np.array([  4,   4,   4]), 
                       'Base/zone/ZoneGridConnectivity/Zmin_match/:CGNS#Distribution/Index' : np.array([  4,   4,   4])
    }
    # Check distribution on empty ranks
    for key in distri_to_check:
      assert np.array_equal(PT.get_node_from_path(gather_tree, key)[1], distri_to_check[key])

    # Cleaning distribution and GridConnectivityProperty nodes to check arrays
    PT.rm_nodes_from_name(gather_tree, ':CGNS#Distribution')
    PT.rm_nodes_from_name(gather_tree, 'GridConnectivityProperty')
          
    # Check arrays
    for node in PT.get_nodes_from_label(gather_zone, 'IndexArray_t'):
      assert node[1].size==0

    for node in PT.get_nodes_from_label(gather_zone, 'DataArray_t'):
      assert node[1].size==0
    




@mark_mpi_test([1, 2, 3])
def test_redistribute_tree_U(sub_comm, write_output):
  
  # Reference directory and file
  ref_dir  = os.path.join(os.path.dirname(__file__), 'references')
  ref_file = os.path.join(ref_dir, f'cube_bcdataset_and_periodic.yaml')

  # Loading file
  dist_tree     = Mio.file_to_dist_tree(ref_file, sub_comm)
  dist_tree_ref = PT.deep_copy(dist_tree)

  # Gather and uniform
  gather_tree = RDT.redistribute_tree(dist_tree  , sub_comm, policy='gather')
  dist_tree   = RDT.redistribute_tree(gather_tree, sub_comm, policy='uniform')

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_ref, os.path.join(out_dir, 'ref_tree.cgns'), sub_comm)
    Mio.dist_tree_to_file(dist_tree    , os.path.join(out_dir, 'out_tree.cgns'), sub_comm)

  # assert PT.is_same_tree(dist_tree, dist_tree_ref)
