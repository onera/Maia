import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import maia.io
import maia.pytree as PT

from maia.pytree.yaml import parse_yaml_cgns
import maia.utils.test_utils as TU

import os

@mark_mpi_test(1)
def test_dist_tree_to_file_1proc(sub_comm):
  yt = """
Base CGNSBase_t I4 [3, 3]:
  Zone Zone_t I4 [[4, 0, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [0., 1., 2., 3.]:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [0, 4, 4]:
      Cell DataArray_t I4 [0, 0, 0]:
"""

  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)

  tmp_dir = TU.create_collective_tmp_dir(sub_comm)
  out_file = os.path.join(tmp_dir, 'yt.cgns')
  maia.io.dist_tree_to_file(dist_tree, out_file, sub_comm)

  t = maia.io.cgns_io_tree.read_tree(out_file)
  assert (PT.get_value(PT.get_node_from_name(t,"CoordinateX")) == [0.,1.,2.,3.]).all()
  TU.rm_collective_dir(tmp_dir, sub_comm)


@mark_mpi_test(2)
def test_dist_tree_to_file_2procs(sub_comm):
  if sub_comm.Get_rank()==0:
    yt = """
Base CGNSBase_t I4 [3, 3]:
  Zone Zone_t I4 [[4, 0, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [0., 1.]:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [0, 2, 4]:
      Cell DataArray_t I4 [0, 0, 0]:
"""
  else:
    yt = """
Base CGNSBase_t I4 [3, 3]:
  Zone Zone_t I4 [[4, 0, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [2., 3.]:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [2, 4, 4]:
      Cell DataArray_t I4 [0, 0, 0]:
"""

  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)

  tmp_dir = TU.create_collective_tmp_dir(sub_comm)
  out_file = os.path.join(tmp_dir, 'yt.cgns')
  maia.io.dist_tree_to_file(dist_tree, out_file, sub_comm)

  if sub_comm.Get_rank()==0:
    t = maia.io.cgns_io_tree.read_tree(out_file)
    assert (PT.get_value(PT.get_node_from_name(t,"CoordinateX")) == [0.,1.,2.,3.]).all()
  TU.rm_collective_dir(tmp_dir, sub_comm)
