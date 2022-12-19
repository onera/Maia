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

# =======================================================================================
# ---------------------------------------------------------------------------------------

distribution = lambda n_elt, comm : par_utils.gathering_distribution(0, n_elt, comm)

# ---------------------------------------------------------------------------------------
@mark_mpi_test([3])
def test_redistribute_pl_node_U(sub_comm):
  if sub_comm.Get_rank() == 0:
    yt_bc = f"""
    BC0 BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[1, 5, 7, 12]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {dtype} [0, 4, 11]:
    """
  if sub_comm.Get_rank() == 1:
    yt_bc = f"""
    BC0 BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[8, 11, 10, 21]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {dtype} [4, 8, 11]:
    """
  if sub_comm.Get_rank() == 2:
    yt_bc = f"""
    BC0 BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[22, 23, 30]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {dtype} [8, 11, 11]:
    """
  
  dist_bc = parse_yaml_cgns.to_cgns_tree(yt_bc)[2][0]
  
  gather_bc = RDT.redistribute_pl_node(dist_bc, distribution, sub_comm)
  
  if sub_comm.Get_rank()==0:
    assert np.array_equal(PT.get_node_from_name(gather_bc, 'Index'    )[1], np.array([0,11,11]))
    assert np.array_equal(PT.get_node_from_name(gather_bc, 'PointList')[1], np.array([[1, 5, 7, 12, 8, 11, 10, 21, 22, 23, 30]]))
  
  else:
    assert np.array_equal(PT.get_node_from_name(gather_bc, 'Index'    )[1], np.array([11,11,11]))
    assert PT.get_node_from_name(gather_bc, 'PointList')[1].size==0
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
@mark_mpi_test([3])
def test_redistribute_data_node_U(sub_comm):
  if sub_comm.Get_rank() == 0:
    yt_bc = f"""
    FlowSolution FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      Density   DataArray_t R8 [0.9, 1.1]:
      MomentumX DataArray_t R8 [0.0, 1.0]:
    """
    old_distrib = np.array([0,2,6])
    new_distrib = np.array([0,6,6])
  if sub_comm.Get_rank() == 1:
    yt_bc = f"""
    FlowSolution FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      Density   DataArray_t R8 [0.8, 1.2]:
      MomentumX DataArray_t R8 [0.1, 0.9]:
    """
    old_distrib = np.array([2,4,6])
    new_distrib = np.array([6,6,6])
  if sub_comm.Get_rank() == 2:
    yt_bc = f"""
    FlowSolution FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      Density   DataArray_t R8 [0.7, 1.3]:
      MomentumX DataArray_t R8 [0.2, 0.8]:
    """
    old_distrib = np.array([4,6,6])
    new_distrib = np.array([6,6,6])
  
  dist_fs = parse_yaml_cgns.to_cgns_tree(yt_bc)[2][0]
  
  gather_fs = RDT.redistribute_data_node(dist_fs, old_distrib, new_distrib, sub_comm)
  
  if sub_comm.Get_rank()==0:
    assert np.array_equal(PT.get_node_from_name(gather_fs, 'Density')[1],
                          np.array([0.9, 1.1, 0.8, 1.2, 0.7, 1.3]))
    assert np.array_equal(PT.get_node_from_name(gather_fs, 'MomentumX')[1],
                          np.array([0.0, 1.0, 0.1, 0.9, 0.2, 0.8]))
  
  else:
    assert PT.get_node_from_name(gather_fs, 'Density'  )[1].size==0
    assert PT.get_node_from_name(gather_fs, 'MomentumX')[1].size==0
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("elt", ["NGON_n", 'TRI_3'])
@mark_mpi_test([3])
def test_redistribute_elements_node_U(elt, sub_comm):
  cgns_elt = {'NGON_n': 22, 'TRI_3': 5}
  if sub_comm.Get_rank() == 0:
    str_ESO = {'TRI_3' :'',
               'NGON_n':'ElementStartOffset  DataArray_t  I4 [0, 3, 6, 9]:' }
    str_DEC = {'TRI_3' : '',
               'NGON_n':f'ElementConnectivity DataArray_t {dtype} [0, 9, 24]:' }
    yt_bc = f"""
    Elements Elements_t I4 [{cgns_elt[elt]}, 0]:
      ElementRange        IndexRange_t I4 [1, 8]:
      {str_ESO[elt]}
      ElementConnectivity DataArray_t  I4 [1, 3, 7, 2, 9, 2, 4, 2, 7]:
      ParentElements      DataArray_t  I4 [[1, 4], [9, 2], [9, 4]]:
      :CGNS#Distribution UserDefinedData_t:
        Element             DataArray_t {dtype} [0, 3,  8]:
        {str_DEC[elt]}
    """

  if sub_comm.Get_rank() == 1:
    str_ESO = {'TRI_3' :'',
               'NGON_n':'ElementStartOffset  DataArray_t  I4 [9, 12, 15, 18]:' }
    str_DEC = {'TRI_3' : '',
               'NGON_n':f'ElementConnectivity DataArray_t {dtype} [ 9, 18, 24]:' }
    yt_bc = f"""
    Elements Elements_t I4 [{cgns_elt[elt]}, 0]:
      ElementRange        IndexRange_t I4 [1, 8]:
      {str_ESO[elt]}
      ElementConnectivity DataArray_t  I4 [7, 8, 3, 6, 1, 4, 2, 6, 5]:
      ParentElements      DataArray_t  I4 [[2, 5], [1, 2], [7, 6]]:
      :CGNS#Distribution UserDefinedData_t:
        Element             DataArray_t {dtype} [ 3,  6,  8]:
        {str_DEC[elt]}
    """
  if sub_comm.Get_rank() == 2:
    str_ESO = {'TRI_3' :'',
               'NGON_n':'ElementStartOffset  DataArray_t  I4 [18, 21, 24]:' }
    str_DEC = {'TRI_3' : '',
               'NGON_n':f'ElementConnectivity DataArray_t {dtype} [18, 24, 24]:' }
    yt_bc = f"""
    Elements Elements_t I4 [{cgns_elt[elt]}, 0]:
      ElementRange        IndexRange_t I4 [1, 8]:
      {str_ESO[elt]}
      ElementConnectivity DataArray_t  I4 [9, 2, 3, 1, 5, 4]:
      ParentElements      DataArray_t  I4 [[8, 1], [9, 2]]:
      :CGNS#Distribution UserDefinedData_t:
        Element             DataArray_t {dtype} [ 6,  8,  8]:
        {str_DEC[elt]}
    """
  
  dist_elt = parse_yaml_cgns.to_cgns_tree(yt_bc)[2][0]

  gather_elt = RDT.redistribute_elements_node(dist_elt, distribution, sub_comm)

  assert np.array_equal(PT.get_node_from_name(gather_elt, 'ElementRange')[1], np.array([1, 8]))

  if sub_comm.Get_rank()==0:
    node_value = {'ElementConnectivity'                    : np.array([1, 3, 7, 2, 9, 2, 4, 2, 7, 7, 8, 3, 6, 1, 4, 2, 6, 5, 9, 2, 3, 1, 5, 4]),
                  'ElementStartOffset'                     : np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]),
                  'ParentElements'                         : np.array([ [1, 4], [9, 2], [9, 4], [2, 5], [1, 2], [7, 6], [8, 1], [9, 2]]),
                  ':CGNS#Distribution/Element'             : np.array([0, 8, 8 ]),
                  ':CGNS#Distribution/ElementConnectivity' : np.array([0, 24, 24 ]),
                  }

    for node in node_value :
      if node in ['ElementStartOffset', ':CGNS#Distribution/ElementConnectivity']:
        if elt=='NGON_n': assert np.array_equal(PT.get_node_from_path(gather_elt, node)[1], node_value[node])
        else            : assert                PT.get_node_from_path(gather_elt, node)   is None
      else:
        assert                   np.array_equal(PT.get_node_from_path(gather_elt, node)[1], node_value[node])
  
  else:
    assert PT.get_child_from_name(gather_elt, 'ElementConnectivity')[1].size==0
    assert PT.get_child_from_name(gather_elt, 'ParentElements'     )[1].size==0
    assert np.array_equal(PT.get_node_from_path(gather_elt, ':CGNS#Distribution/Element')[1],
                          np.array([8, 8, 8 ]))
    if elt=='NGON_n':
      assert PT.get_child_from_name(gather_elt, 'ElementStartOffset' )[1].size==0
      assert np.array_equal(PT.get_node_from_path(gather_elt, ':CGNS#Distribution/ElementConnectivity')[1],
                            np.array([24, 24, 24 ]))
    else :
      assert PT.get_node_from_name(gather_elt, 'ElementStartOffset'                    ) is None
      assert PT.get_node_from_name(gather_elt, ':CGNS#Distribution/ElementConnectivity') is None
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
@mark_mpi_test([2])
def test_redistribute_mixed_elements_node_U(sub_comm):

  if sub_comm.Get_rank() == 0:
    yt_bc = f"""
    Mixed Elements_t I4 [20, 0]:
      ElementRange        IndexRange_t I4 [1, 8]:
      ElementStartOffset  DataArray_t  I4 [0, 3, 8, 12]:
      ElementConnectivity DataArray_t  I4 [1, 3, 7, 2, 9, 2, 4, 2, 7, 1, 3, 4]:
      ParentElements      DataArray_t  I4 [[1, 4], [9, 2], [9, 4]]:
      :CGNS#Distribution  UserDefinedData_t:
        Element             DataArray_t {dtype} [0, 3,  6]:
        ElementConnectivity DataArray_t {dtype} [0, 12, 24]:
    """

  if sub_comm.Get_rank() == 1:
    yt_bc = f"""
    Mixed Elements_t I4 [20, 0]:
      ElementRange        IndexRange_t I4 [1, 8]:
      ElementStartOffset  DataArray_t  I4 [12, 15, 19, 24]:
      ElementConnectivity DataArray_t  I4 [7, 8, 3, 6, 1, 4, 2, 6, 5, 6, 3, 1]:
      ParentElements      DataArray_t  I4 [[2, 5], [1, 2], [7, 6]]:
      :CGNS#Distribution  UserDefinedData_t:
        Element             DataArray_t {dtype} [ 3,  6,  6]:
        ElementConnectivity DataArray_t {dtype} [12, 24, 24]:
    """

  dist_elt = parse_yaml_cgns.to_cgns_tree(yt_bc)[2][0]

  gather_elt = RDT.redistribute_elements_node(dist_elt, distribution, sub_comm)

  assert np.array_equal(PT.get_node_from_name(gather_elt, 'ElementRange')[1], np.array([1, 8]))

  if sub_comm.Get_rank()==0:
    node_value = {'ElementConnectivity'                    : np.array([1, 3, 7, 2, 9, 2, 4, 2, 7, 1, 3, 4, 7, 8, 3, 6, 1, 4, 2, 6, 5, 6, 3, 1]),
                  'ElementStartOffset'                     : np.array([0, 3, 8, 12, 15, 19, 24]),
                  'ParentElements'                         : np.array([ [1, 4], [9, 2], [9, 4], [2, 5], [1, 2], [7, 6]]),
                  ':CGNS#Distribution/Element'             : np.array([0, 6, 6]),
                  ':CGNS#Distribution/ElementConnectivity' : np.array([0, 24, 24]),
                  }

    for node in node_value :
      if node in ['ElementStartOffset', ':CGNS#Distribution/ElementConnectivity']:
        assert np.array_equal(PT.get_node_from_path(gather_elt, node)[1], node_value[node])
      else:
        assert                   np.array_equal(PT.get_node_from_path(gather_elt, node)[1], node_value[node])

  else:
    assert PT.get_child_from_name(gather_elt, 'ElementConnectivity')[1].size==0
    assert PT.get_child_from_name(gather_elt, 'ParentElements'     )[1].size==0
    assert np.array_equal(PT.get_node_from_path(gather_elt, ':CGNS#Distribution/Element')[1],
                          np.array([6, 6, 6]))
    assert PT.get_child_from_name(gather_elt, 'ElementStartOffset' )[1].size==0
    assert np.array_equal(PT.get_node_from_path(gather_elt, ':CGNS#Distribution/ElementConnectivity')[1],
                          np.array([24, 24, 24]))
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
@mark_mpi_test([2])
def test_redistribute_gc_node_U(sub_comm):
  if sub_comm.Get_rank() == 0:
    yt_gc = f"""
    ZGC ZoneGridConnectivity_t:
        Zmin_match GridConnectivity_t 'zone':
            GridConnectivityType     GridConnectivityType_t 'Abutting1to1':
            GridLocation             GridLocation_t         'FaceCenter':
            PointList                IndexArray_t       I4 [[]]:
            PointListDonor           IndexArray_t       I4 [[]]:
            GridConnectivityProperty GridConnectivityProperty_t:
            :CGNS#Distribution UserDefinedData_t:
                Index DataArray_t {dtype} [0, 0, 4]:
        Zmax_match GridConnectivity_t 'zone':
            GridConnectivityType     GridConnectivityType_t 'Abutting1to1':
            GridLocation             GridLocation_t         'FaceCenter':
            PointList                IndexArray_t       I4 [[9, 10, 11, 12]]:
            PointListDonor           IndexArray_t       I4 [[1, 2, 3, 4]]:
            GridConnectivityProperty GridConnectivityProperty_t:
            :CGNS#Distribution UserDefinedData_t:
                Index DataArray_t {dtype} [0, 4, 4]:
                    """
  if sub_comm.Get_rank() == 1:
    yt_gc = f"""
    ZGC ZoneGridConnectivity_t:
        Zmin_match GridConnectivity_t 'zone':
            GridConnectivityType     GridConnectivityType_t 'Abutting1to1':
            GridLocation             GridLocation_t         'FaceCenter':
            PointList                IndexArray_t       I4 [[1, 2, 3, 4]]:
            PointListDonor           IndexArray_t       I4 [[9, 10, 11, 12]]:
            GridConnectivityProperty GridConnectivityProperty_t:
            :CGNS#Distribution UserDefinedData_t:
                Index DataArray_t {dtype} [0, 4, 4]:
        Zmax_match GridConnectivity_t 'zone':
            GridConnectivityType     GridConnectivityType_t 'Abutting1to1':
            GridLocation             GridLocation_t         'FaceCenter':
            PointList                IndexArray_t       I4 [[]]:
            PointListDonor           IndexArray_t       I4 [[]]:
            GridConnectivityProperty GridConnectivityProperty_t:
            :CGNS#Distribution UserDefinedData_t:
                Index DataArray_t {dtype} [4, 4, 4]:
                    """

  zgc_n = parse_yaml_cgns.to_cgns_tree(yt_gc)[2][0]

  for gc_n in PT.get_children_from_label(zgc_n, 'GridConnectivity_t') :
    gather_gc = RDT.redistribute_pl_node(gc_n, distribution, sub_comm)

    if sub_comm.Get_rank()==0:
      if gc_n[0]=='Zmin_match':
        assert np.array_equal(PT.get_node_from_path(gather_gc, 'PointList'     )[1], np.array([[1,  2,  3,  4]]))
        assert np.array_equal(PT.get_node_from_path(gather_gc, 'PointListDonor')[1], np.array([[9, 10, 11, 12]]))
      elif gc_n[0]=='Zmax_match':
        assert np.array_equal(PT.get_node_from_path(gather_gc, 'PointList'     )[1], np.array([[9, 10, 11, 12]]))
        assert np.array_equal(PT.get_node_from_path(gather_gc, 'PointListDonor')[1], np.array([[1,  2,  3,  4]]))
      assert np.array_equal(PT.get_node_from_path(gather_gc, ':CGNS#Distribution/Index')[1], np.array([0,4,4]))
    
    else:
      assert                PT.get_node_from_name(gather_gc, 'PointList')[1].size==0
      assert np.array_equal(PT.get_node_from_path(gather_gc, ':CGNS#Distribution/Index')[1], np.array([4,4,4]))  
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
@mark_mpi_test([1, 2, 3])
def test_redistribute_zone_U(sub_comm, write_output):
  
  # Reference directory and file
  ref_dir  = os.path.join(os.path.dirname(__file__), 'references')
  ref_file = os.path.join(ref_dir, f'cube_bcdataset_and_periodic.yaml')

  dist_tree   = Mio.file_to_dist_tree(ref_file, sub_comm)
  gather_tree = dist_tree
  
  for zone in PT.get_all_Zone_t(gather_tree):
    distribution = lambda n_elt, comm : par_utils.gathering_distribution(0, n_elt, comm)
    gather_zone  = RDT.redistribute_zone(zone, distribution, sub_comm)

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
    if write_output:
      print(os.path.join(out_dir, ref_file.split('/')[-1].split('.')[0]+'.cgns'))
      Mio.write_tree(ref_tree, os.path.join(out_dir, ref_file.split('/')[-1].split('.')[0]+'.cgns'))
      Mio.write_tree(gather_tree, os.path.join(out_dir, 'gather_tree.cgns'))

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
# ---------------------------------------------------------------------------------------
    


# =======================================================================================
