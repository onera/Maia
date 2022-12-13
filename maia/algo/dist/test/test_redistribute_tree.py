from   pytest_mpi_check._decorator import mark_mpi_test
import pytest
import os
import numpy as np

import maia
import maia.io            as Mio
import maia.pytree        as PT
import maia.pytree.maia   as MT

from   maia.utils         import par_utils
from   maia.algo.dist     import redistribute_tree as RDT
from   maia.pytree.yaml   import parse_yaml_cgns

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'

# =======================================================================================
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
  
  gather_bc = RDT.redistribute_pl_node(dist_bc, par_utils.gathering_distribution, sub_comm)
  
  if sub_comm.Get_rank()==0:
    assert np.array_equal(PT.get_node_from_name(gather_bc, 'Index'    )[1], np.array([0,11,11]))
    assert np.array_equal(PT.get_node_from_name(gather_bc, 'PointList')[1], np.array([[1, 5, 7, 12, 8, 11, 10, 21, 22, 23, 30]]))
  
  else:
    assert np.array_equal(PT.get_node_from_name(gather_bc, 'Index'    )[1], np.array([11,11,11]))
    assert PT.get_node_from_name(gather_bc, 'PointList')[1].size==0



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



@mark_mpi_test([3])
def test_redistribute_elements_node_U(sub_comm):
  if sub_comm.Get_rank() == 0:
    yt_bc = f"""
    NGonElements Elements_t I4 [22, 0]:
      ElementRange        IndexRange_t I4 [1, 8]:
      ElementStartOffset  DataArray_t  I4 [0, 4, 7, 11]:
      ElementConnectivity DataArray_t  I4 [1, 3, 7, 2, 3, 9, 2, 4, 7, 2, 7]:
      ParentElements      DataArray_t  I4 [[1, 4], [9, 2], [9, 4]]:
      :CGNS#Distribution UserDefinedData_t:
        Element             DataArray_t {dtype} [0, 3, 8]:
        ElementConnectivity DataArray_t {dtype} [0, 11, 28]:
    """
  if sub_comm.Get_rank() == 1:
    yt_bc = f"""
    NGonElements Elements_t I4 [22, 0]:
      ElementRange        IndexRange_t I4 [1, 8]:
      ElementStartOffset  DataArray_t  I4 [11, 15, 18, 21]:
      ElementConnectivity DataArray_t  I4 [7, 8, 3, 6, 1, 4, 9, 2, 6, 5]:
      ParentElements      DataArray_t  I4 [[2, 5], [1, 2], [7, 6]]:
      :CGNS#Distribution UserDefinedData_t:
        Element             DataArray_t {dtype} [ 3,  6, 8]:
        ElementConnectivity DataArray_t {dtype} [11, 21, 28]:
    """
  if sub_comm.Get_rank() == 2:
    yt_bc = f"""
    NGonElements Elements_t I4 [22, 0]:
      ElementRange        IndexRange_t I4 [1, 8]:
      ElementStartOffset  DataArray_t  I4 [21, 24, 28]:
      ElementConnectivity DataArray_t  I4 [9, 2, 3, 1, 5, 4, 3]:
      ParentElements      DataArray_t  I4 [[8, 1], [9, 2]]:
      :CGNS#Distribution UserDefinedData_t:
        Element             DataArray_t {dtype} [6, 8, 8]:
        ElementConnectivity DataArray_t {dtype} [21, 28, 28]:
    """
  
  dist_elt = parse_yaml_cgns.to_cgns_tree(yt_bc)[2][0]
  
  gather_elt = RDT.redistribute_elements_node(dist_elt, par_utils.gathering_distribution, sub_comm)

  assert np.array_equal(PT.get_node_from_name(gather_elt, 'ElementRange')[1],
                          np.array([1, 8]))

  if sub_comm.Get_rank()==0:
    assert np.array_equal(PT.get_child_from_name(gather_elt, 'ElementStartOffset')[1],
                           np.array([0, 4, 7, 11, 15, 18, 21, 24, 28 ]))
    assert np.array_equal(PT.get_child_from_name(gather_elt, 'ElementConnectivity')[1],
                           np.array([1, 3, 7, 2, 3, 9, 2, 4, 7, 2, 7, \
                                     7, 8, 3, 6, 1, 4, 9, 2, 6, 5, \
                                     9, 2, 3, 1, 5, 4, 3]))
    assert np.array_equal(PT.get_child_from_name(gather_elt, 'ParentElements')[1],
                           np.array([ [1, 4], [9, 2], [9, 4],\
                                      [2, 5], [1, 2], [7, 6],\
                                      [8, 1], [9, 2]]))
    assert np.array_equal(PT.get_node_from_path(gather_elt, ':CGNS#Distribution/Element')[1],
                           np.array([0, 8, 8 ]))
    assert np.array_equal(PT.get_node_from_path(gather_elt, ':CGNS#Distribution/ElementConnectivity')[1],
                           np.array([0, 28, 28 ]))
  
  else:
    assert PT.get_child_from_name(gather_elt, 'ElementStartOffset' )[1].size==0
    assert PT.get_child_from_name(gather_elt, 'ElementConnectivity')[1].size==0
    assert PT.get_child_from_name(gather_elt, 'ParentElements'     )[1].size==0
    assert np.array_equal(PT.get_node_from_path(gather_elt, ':CGNS#Distribution/Element')[1],
                           np.array([8, 8, 8 ]))
    assert np.array_equal(PT.get_node_from_path(gather_elt, ':CGNS#Distribution/ElementConnectivity')[1],
                           np.array([28, 28, 28 ]))




# ---------------------------------------------------------------------------------------
# =======================================================================================