import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia              import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.pytree.yaml  import parse_yaml_cgns
from maia.factory      import dcube_generator as DCG

from maia.algo.part import interpolate as ITP

dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'

src_part_0 = f"""
ZoneU Zone_t [[18,4,0]]:
  ZoneType ZoneType_t "Unstructured":
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]:
    CoordinateY DataArray_t R8 [0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1., 0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1.]:
    CoordinateZ DataArray_t R8 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
  NGon Elements_t [22,0]:
    ElementRange IndexRange_t [1,20]:
    ElementConnectivity DataArray_t:
       I4 : [ 2,  5,  4,  1,  3,  6,  5,  2,  5,  8,  7,  4,  6,  9,  8,  5, 10,
             13, 14, 11, 11, 14, 15, 12, 13, 16, 17, 14, 14, 17, 18, 15,  1,  4,
             13, 10,  4,  7, 16, 13, 11, 14,  5,  2, 14, 17,  8,  5, 12, 15,  6,
              3, 15, 18,  9,  6, 10, 11,  2,  1, 11, 12,  3,  2,  4,  5, 14, 13,
              5,  6, 15, 14,  7,  8, 17, 16,  8,  9, 18, 17]
    ElementStartOffset DataArray_t I4 [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80]:
    ParentElements DataArray_t:
      I4 : [[1, 0], [2, 0], [4, 0], [3, 0], [1, 0], [2, 0], [4, 0], [3, 0], [1, 0], [4, 0],
            [1, 2], [4, 3], [2, 0], [3, 0], [1, 0], [2, 0], [1, 4], [2, 3], [4, 0], [3, 0]]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {dtype} [1,2,3,4,5,6,7,8,13,14,17,18,21,22,25,27,29,31,33,35]:
  NFaceElements Elements_t [23,0]:
    ElementRange IndexRange_t [21,24]:
    ElementConnectivity DataArray_t I4 [1,5,9,11,15,17,2,6,-11,13,16,18,4,8,-12,14,-18,20,3,7,10,12,-17,19]:
    ElementStartOffset DataArray_t I4 [0,6,12,18,24]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {dtype} [1,2,3,4]:
  MySolution FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    val DataArray_t R8 [1.,2.,3.,4.]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {dtype} [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
    Cell DataArray_t {dtype} [1,2,3,4]:
"""
src_part_1 = f"""
ZoneU Zone_t [[18,4,0]]:
  ZoneType ZoneType_t "Unstructured":
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]:
    CoordinateY DataArray_t R8 [0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1., 0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1.]:
    CoordinateZ DataArray_t R8 [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
  NGon Elements_t [22,0]:
    ElementRange IndexRange_t [1,20]:
    ElementConnectivity DataArray_t:
       I4 : [10, 11, 14, 13, 11, 12, 15, 14, 13, 14, 17, 16, 14, 15, 18, 17,  1,
              4,  5,  2,  2,  5,  6,  3,  4,  7,  8,  5,  5,  8,  9,  6, 10, 13,
              4,  1, 13, 16,  7,  4,  2,  5, 14, 11,  5,  8, 17, 14,  3,  6, 15,
             12,  6,  9, 18, 15,  1,  2, 11, 10,  2,  3, 12, 11, 13, 14,  5,  4,
             14, 15,  6,  5, 16, 17,  8,  7, 17, 18,  9,  8]
    ElementStartOffset DataArray_t I4 [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80]:
    ParentElements DataArray_t:
      I4 : [[1, 0], [2, 0], [3, 0], [4, 0], [1, 0], [2, 0], [3, 0], [4, 0], [1, 0], [3, 0],
            [1, 2], [3, 4], [2, 0], [4, 0], [1, 0], [2, 0], [1, 3], [2, 4], [3, 0], [4, 0]]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {dtype} [5,6,7,8,9,10,11,12,15,16,19,20,23,24,26,28,30,32,34,36]:
  NFaceElements Elements_t [23,0]:
    ElementRange IndexRange_t [21,24]:
    ElementConnectivity DataArray_t I4 [1,5,9,11,15,17,2,6,-11,13,16,18,3,7,10,12,-17,19,4,8,-12,14,-18,20]:
    ElementStartOffset DataArray_t I4 [0,6,12,18,24]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {dtype} [5,6,7,8]:
  MySolution FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    val DataArray_t R8 [5.,6.,7.,8.]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {dtype} [19,20,21,22,23,24,25,26,27,10,11,12,13,14,15,16,17,18]:
    Cell DataArray_t {dtype} [5,6,7,8]:
"""
tgt_part_0 = f"""
ZoneU Zone_t [[12,2,0]]:
  ZoneType ZoneType_t "Unstructured":
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0.2, 0.2, 0.2, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7]:
    CoordinateY DataArray_t R8 [0.2 , 0.2 , 0.2 , 0.2 , 0.7, 0.7, 0.2 , 0.7, 0.7, 0.2 , 0.7, 0.7]:
    CoordinateZ DataArray_t R8 [-0.2 , 0.3, 0.8 , -0.2 , -0.2 , -0.2 , 0.3, 0.3, 0.3, 0.8, 0.8, 0.8]:
  NGon Elements_t [22,0]:
    ElementRange IndexRange_t [1,11]:
    ElementConnectivity DataArray_t:
       I4 : [4,  6,  5,  1,  2,  8,  9,  7,  3, 11, 12, 10,  1,  5,  8,  2,  2,
             8, 11,  3,  7,  9,  6,  4, 10, 12,  9,  7,  2,  7,  4,  1,  3, 10,
             7,  2,  5,  6,  9,  8,  8,  9, 12, 11]
    ElementStartOffset DataArray_t I4 [0,4,8,12,16,20,24,28,32,36,40,44]:
    ParentElements DataArray_t:
      I4 : [[1, 0], [1, 2], [2, 0], [1, 0], [2, 0], [1, 0], [2, 0], [1, 0], [2, 0], [1, 0], [2,0]]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {dtype} [1,5,9,13,15,17,19,25,26,29,30]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {dtype} [1,10,19,2,4,5,11,13,14,20,22,23]:
    Cell DataArray_t {dtype} [1,5]:
"""
tgt_part_1 = f"""
ZoneU Zone_t [[16,3,0]]:
  ZoneType ZoneType_t "Unstructured":
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0.2, 0.2, 0.2, 0.7, 1.2, 0.2, 0.7, 0.7, 0.2, 0.7, 1.2, 0.7, 1.2, 0.2, 0.7, 1.2]:
    CoordinateY DataArray_t R8 [1.2, 1.2, 1.2, 1.2, 1.2, 0.7, 0.7, 1.2, 0.7, 0.7, 0.7, 1.2, 1.2, 0.7, 0.7, 0.7]:
    CoordinateZ DataArray_t R8 [-.2, 0.3, 0.8, 0.8, 0.8, -.2,-.2 ,-.2 , 0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8]:
  NGon Elements_t [22,0]:
    ElementRange IndexRange_t [1,16]:
    ElementConnectivity DataArray_t:
       I4 : [7,  8,  1,  6,  9,  2, 12, 10, 10, 11, 13, 12, 14,  3,  4, 15, 15,
             4,  5, 16,  6,  1,  2,  9,  9,  2,  3, 14, 10, 12,  8,  7, 15,  4,
            12, 10, 16,  5, 13, 11,  6,  9, 10,  7,  9, 14, 15, 10, 10, 15, 16,
            11,  1,  8, 12,  2,  2, 12,  4,  3, 12, 13,  5,  4]
    ElementStartOffset DataArray_t I4 [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]:
    ParentElements DataArray_t:
      I4 : [[3, 0], [3, 1], [2, 0], [1, 0], [2, 0], [3, 0], [1, 0], [3, 0],
            [1, 2], [2, 0], [3, 0], [1, 0], [2, 0], [3, 0], [1, 0], [2, 0]]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {dtype} [3,7,8,11,12,14,16,18,20,24,29,30,32,33,34,36]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {dtype} [7,16,25,26,27,4,5,8,13,14,15,17,18,22,23,24]:
    Cell DataArray_t {dtype} [7,8,3]:
"""
tgt_part_2 = f"""
ZoneU Zone_t [[16,3,0]]:
  ZoneType ZoneType_t "Unstructured":
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [1.2, 1.2, 1.2, 1.2, 1.2, 0.7, 0.7, 0.7, 0.7, 0.7, 1.2, 0.7, 1.2, 0.7, 0.7, 1.2]:
    CoordinateY DataArray_t R8 [0.2, 0.7, 1.2, 0.2, 0.2, 0.2, 0.7, 1.2, 0.2, 0.7, 0.7, 1.2, 1.2, 0.2, 0.7, 0.7]:
    CoordinateZ DataArray_t R8 [-.2, -.2, -.2, 0.3, 0.8, -.2, -.2, -.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8]:
  NGon Elements_t [22,0]:
    ElementRange IndexRange_t [1,16]:
    ElementConnectivity DataArray_t:
       I4 : [ 1,  2,  7,  6,  2,  3,  8,  7,  9, 10, 11,  4, 10, 12, 13, 11, 14,
             15, 16,  5,  9,  6,  7, 10, 10,  7,  8, 12, 14,  9, 10, 15,  4, 11,
              2,  1, 11, 13,  3,  2,  5, 16, 11,  4,  9,  4,  1,  6, 14,  5,  4,
              9,  7,  2, 11, 10, 10, 11, 16, 15,  8,  3, 13, 12]
    ElementStartOffset DataArray_t I4 [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]:
    ParentElements DataArray_t:
      I4 : [[1, 0], [3, 0], [1, 2], [3, 0], [2, 0], [1, 0], [3, 0], [2, 0],
            [1, 0], [3, 0], [2, 0], [1, 0], [2, 0], [1, 3], [2, 0], [3, 0]]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {dtype} [2,4,6,8,10,17,18,19,21,22,23,27,28,31,32,35]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {dtype} [3,6,9,12,21,2,5,8,11,14,15,17,18,20,23,24]:
    Cell DataArray_t {dtype} [2,6,4]:
"""

@mark_mpi_test(2)
def test_create_src_to_tgt(sub_comm):
  #Here we just check if src_to_tgt is created
  if sub_comm.Get_rank() == 0:
    pt = src_part_0
  else:
    pt = src_part_1
  zones = parse_yaml_cgns.to_nodes(pt)
  for zone in zones:
    pe = PT.get_node_from_name(zone, 'ParentElements')
    #Put it in F order
    newpe = np.empty(pe[1].shape, dtype=np.int32, order='F')
    newpe[:] = np.copy(pe[1][:])
    PT.set_value(pe, newpe)

  src_parts_per_dom = [zones]
  tgt_parts_per_dom = [[PT.deep_copy(zone) for zone in zones]]
  excp_target = np.array([1,2,3,4]) if sub_comm.Get_rank() == 0 else np.array([5,6,7,8])
  src_to_tgt = ITP.create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, sub_comm)
  assert (src_to_tgt[0]['target_idx'] == [0,1,2,3,4]).all()
  assert (src_to_tgt[0]['target'] == excp_target).all()
  src_to_tgt = ITP.create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, sub_comm, strategy='Closest')
  assert (src_to_tgt[0]['target_idx'] == [0,1,2,3,4]).all()
  assert (src_to_tgt[0]['target'] == excp_target).all()

  for tgt_zones in tgt_parts_per_dom:
    for tgt_zone in tgt_zones:
      cx = PT.get_node_from_name(zone, 'CoordinateX')
      cx[1] += .5
  excp_target = np.array([2,1,3,4]) if sub_comm.Get_rank() == 0 else np.array([6,5,8,7])
  src_to_tgt = ITP.create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, sub_comm, strategy='LocationAndClosest')
  assert (src_to_tgt[0]['target'] == excp_target).all()

  excp_target = np.array([2,3]) if sub_comm.Get_rank() == 0 else np.array([6,8])
  src_to_tgt = ITP.create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, sub_comm, strategy='Location')
  assert (src_to_tgt[0]['target'] == excp_target).all()

def test_interpolator_reductions():
  class Empty: #Used to create a interpolator like object
    pass
  fake_interpolator = Empty()

  fake_interpolator.sending_gnums = [{'come_from_idx' : np.array([0,1,2,3])}]
  data = np.array([1,2,3], np.int32)
  out = ITP.Interpolator._reduce_single_val(fake_interpolator, 0, data)
  assert out is data

  fake_interpolator.sending_gnums = [{'come_from_idx' : np.array([0,2,4,6])}]
  fake_interpolator.tgt_dist = [np.array([1,1,0,1,3,1])]
  data = np.array([1,2, 10,11, 20,30], np.float64)
  out = ITP.Interpolator._reduce_mean_dist(fake_interpolator, 0, data)
  assert (out == np.array([1.5, 10., 27.5])).all()

  fake_interpolator.sending_gnums = [{'come_from_idx' : np.array([0,2,5,6])}]
  fake_interpolator.tgt_dist = [np.array([1,1,0,1,3,1])]
  data = np.array([1,2, 10,11, 20,30], np.float64)
  with pytest.raises(AssertionError):
    out = ITP.Interpolator._reduce_mean_dist(fake_interpolator, 0, data)

@mark_mpi_test(2)
def test_interpolate_fields(sub_comm):
  if sub_comm.Get_rank() == 0:
    pt = src_part_0
    expected_sol = np.array([2.,2.,2.,3.,3.,3.,3.,3.,3., 2.,2.,2.,3.,3.,3.,3.,3.,3.])
  else:
    pt = src_part_1
    expected_sol = np.array([6.,6.,6.,8.,8.,8.,8.,8.,8., 2.,2.,2.,3.,3.,3.,3.,3.,3.])
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  src_parts_per_dom = [PT.get_all_Zone_t(part_tree)]
  tgt_parts_per_dom = [[PT.deep_copy(zone) for zone in PT.get_all_Zone_t(part_tree)]]
  for tgt_zones in tgt_parts_per_dom:
    for tgt_zone in tgt_zones:
      cx = PT.get_node_from_name(tgt_zone, 'CoordinateX')
      cy = PT.get_node_from_name(tgt_zone, 'CoordinateY')
      cz = PT.get_node_from_name(tgt_zone, 'CoordinateZ')
      cx[1] += .55
      cy[1] += .05
      cz[1] -= .05

  src_to_tgt = ITP.create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, sub_comm, location='Vertex')
  interpolator = ITP.Interpolator(src_parts_per_dom, tgt_parts_per_dom, src_to_tgt, 'Vertex', sub_comm)
  interpolator.exchange_fields('MySolution')

  for tgt_zones in tgt_parts_per_dom:
    for tgt_zone in tgt_zones:
      fs = PT.get_node_from_name(tgt_zone, 'MySolution')
      assert PT.Subset.GridLocation(fs) == 'Vertex'
      assert (PT.get_child_from_name(fs, 'val')[1] == expected_sol).all()

@mark_mpi_test(2)
class Test_interpolation_api():
  src_zone_0 = parse_yaml_cgns.to_node(src_part_0)
  src_zone_1 = parse_yaml_cgns.to_node(src_part_1)
  tgt_zone_0 = parse_yaml_cgns.to_node(tgt_part_0)
  tgt_zone_1 = parse_yaml_cgns.to_node(tgt_part_1)
  tgt_zone_2 = parse_yaml_cgns.to_node(tgt_part_2)
  # - For  1 to 9 (bottom)  : 1 2 2 4 3 3 4 3 3
  # - For 10 to 18 (middle) : 1 2 2 4 3 3 4 3 3
  # - For 19 to 27 (top)    : 5 6 6 7 8 8  7 8 8
  expected_vtx_sol = [ np.array([1., 1, 5, 2, 4, 3, 2, 4, 3, 6, 7, 8]),
                       np.array([4., 4, 7, 8, 8, 4, 3, 3, 4, 3, 3, 3, 3, 7, 8, 8]),
                       np.array([2., 3, 3, 2, 6, 2, 3, 3, 2, 3, 3, 3, 3, 6, 8, 8])]
  expected_cell_sol = [ np.array([1., 5.]),
                        np.array([7., 8., 4.]),
                        np.array([2., 6., 3.])]
  all_zones = [src_zone_0, src_zone_1, tgt_zone_0, tgt_zone_1, tgt_zone_2]

  def test_interpolate_from_parts_per_dom(self,sub_comm):
    if sub_comm.Get_rank() == 0:
      src_parts_per_dom = [[self.src_zone_0]]
      tgt_parts_per_dom = [[self.tgt_zone_0, self.tgt_zone_1]]
      expected_vtx_sol = [self.expected_vtx_sol[k] for k in [0,1]]
    elif sub_comm.Get_rank() == 1:
      src_parts_per_dom = [[self.src_zone_1]]
      tgt_parts_per_dom = [[self.tgt_zone_2]]
      expected_vtx_sol = [self.expected_vtx_sol[k] for k in [2]]

    ITP.interpolate_from_parts_per_dom(src_parts_per_dom, tgt_parts_per_dom, sub_comm, \
        ['MySolution'], 'Vertex', strategy='Closest')

    for tgt_zones in tgt_parts_per_dom:
      for i_tgt, tgt_zone in enumerate(tgt_zones):
        fs = PT.get_child_from_name(tgt_zone, 'MySolution')
        assert PT.Subset.GridLocation(fs) == 'Vertex'
        assert (PT.get_child_from_name(fs, 'val')[1] == expected_vtx_sol[i_tgt]).all()

  def test_interpolate_from_dom_part_trees(self,sub_comm):
    src_tree = PT.new_CGNSTree()
    src_base = PT.new_CGNSBase(parent=src_tree)
    tgt_tree = PT.new_CGNSTree()
    tgt_base = PT.new_CGNSBase(parent=tgt_tree)

    if sub_comm.Get_rank() == 0:
      self.src_zone_0[0] = 'Source.P0.N0'
      self.tgt_zone_0[0] = 'Target.P0.N0'
      self.tgt_zone_1[0] = 'Target.P0.N1'
      self.tgt_zone_2[0] = 'Target.P0.N2'
      PT.add_child(src_base, self.src_zone_0)
      PT.add_child(tgt_base, self.tgt_zone_0)
      PT.add_child(tgt_base, self.tgt_zone_1)
      PT.add_child(tgt_base, self.tgt_zone_2)
      expected_vtx_sol = [self.expected_vtx_sol[k] for k in [0,1,2]]
    elif sub_comm.Get_rank() == 1:
      self.src_zone_1[0] = 'Source.P1.N0'
      PT.add_child(src_base, self.src_zone_1)
      expected_vtx_sol = [self.expected_vtx_sol[k] for k in []]

    ITP.interpolate_from_part_trees(src_tree, tgt_tree, sub_comm, \
        ['MySolution'], 'Vertex', strategy='Closest')

    for i_tgt, tgt_zone in enumerate(PT.get_all_Zone_t(tgt_tree)):
      fs = PT.get_child_from_name(tgt_zone, 'MySolution')
      assert PT.Subset.GridLocation(fs) == 'Vertex'
      assert (PT.get_child_from_name(fs, 'val')[1] == expected_vtx_sol[i_tgt]).all()

