import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia
import maia.pytree        as PT
from   maia.pytree.yaml   import parse_yaml_cgns

from maia.algo.part import extraction_utils as EU

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'

import Pypdm.Pypdm as PDM

def test_get_relative_pl():
  part_zone = parse_yaml_cgns.to_node(
  """
  VolZone.P0.N0 Zone_t:
    ZSR1 ZoneSubRegion_t:
      BCRegionName Descriptor_t "BC":
    ZSR2 ZoneSubRegion_t:
      GridConnectivityRegionName Descriptor_t "GC":
    ZSR3 ZoneSubRegion_t:
      GridLocation GridLocation_t "CellCenter" :
      PointList    IndexArray_t   [[4,8,1]]:
    ZoneBC ZoneBC_t:
      BC BC_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList    IndexArray_t   [[9,3,2]]:
    ZoneGC ZoneGridConnectivity_t:
      GC GridConnectivity_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList    IndexArray_t   [[4,3,7]]:
  """)
  zsr1  = PT.get_child_from_name(part_zone,'ZSR1')
  zsr2  = PT.get_child_from_name(part_zone,'ZSR2')
  zsr3  = PT.get_child_from_name(part_zone,'ZSR3')
  pl1_n = PT.get_value(EU.get_relative_pl(zsr1, part_zone))[0]
  pl2_n = PT.get_value(EU.get_relative_pl(zsr2, part_zone))[0]
  pl3_n = PT.get_value(EU.get_relative_pl(zsr3, part_zone))[0]
  assert np.array_equal(pl1_n, np.array([9,3,2],dtype=np.int32))
  assert np.array_equal(pl2_n, np.array([4,3,7],dtype=np.int32))
  assert np.array_equal(pl3_n, np.array([4,8,1],dtype=np.int32))


def test_local_pl_offset():
  zone = PT.new_Zone('Zone', type='Unstructured')
  PT.new_NGonElements('NGon', erange=[1, 10], parent=zone)
  PT.new_NFaceElements('NFace', erange=[11, 20], parent=zone)
  assert EU.local_pl_offset(zone, 0) == 0
  assert EU.local_pl_offset(zone, 2) == 0
  assert EU.local_pl_offset(zone, 3) == 10
  zone = PT.new_Zone('Zone', type='Unstructured')
  PT.new_NFaceElements('NFace', erange=[1, 5], parent=zone)
  PT.new_NGonElements('NGon', erange=[6, 10], parent=zone)
  assert EU.local_pl_offset(zone, 2) == 5
  assert EU.local_pl_offset(zone, 3) == 0
  zone = PT.new_Zone('Zone', type='Unstructured')
  PT.new_Elements('TRI_3', type='TRI_3',erange=[1, 5], parent=zone)
  PT.new_Elements('BAR_2', type='BAR_2',erange=[6,15], parent=zone)
  assert EU.local_pl_offset(zone, 0) == 0
  assert EU.local_pl_offset(zone, 1) == 5
  assert EU.local_pl_offset(zone, 2) == 0


@mark_mpi_test(2)
def test_get_partial_container_stride_and_order(sub_comm):
  if sub_comm.Get_rank()==0:
    pt = """
    Zone.P0.N0 Zone_t:
      NGonElements Elements_t I4 [22,0]:
        ElementRange IndexRange_t I4 [1,5]:
      FSol_A FlowSolution_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t I4 [[2,4,1]]:

    Zone.P0.N1 Zone_t:
      NGonElements Elements_t I4 [22,0]:
        ElementRange IndexRange_t I4 [1,3]:
      FSol_A FlowSolution_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t I4 [[3]]:
    """
    p1_lngn      = [np.array([1,3,5,7,9],   dtype=pdm_gnum_dtype)]
    p2_lngn      = [np.array([1,2,3,4,5],   dtype=pdm_gnum_dtype),
                    np.array([6,7,8],       dtype=pdm_gnum_dtype)]
    p1_to_p2     = [np.array([9,10,9,10,13],dtype=pdm_gnum_dtype)]
    p1_to_p2_idx = [np.array([0,1,2,3,4,5], dtype=np.int32)]
  else:
    pt = """
    Zone.P1.N0 Zone_t:
      NGonElements Elements_t I4 [22,0]:
        ElementRange IndexRange_t I4 [1,5]:
      FSol_A FlowSolution_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t I4 [[4,2]]:
    """
    p1_lngn      = [np.array([2,4,6,8],       dtype=pdm_gnum_dtype)]
    p2_lngn      = [np.array([9,10,11,12,13], dtype=pdm_gnum_dtype)]
    p1_to_p2     = [np.array([1,2,6,7],       dtype=pdm_gnum_dtype)]
    p1_to_p2_idx = [np.array([0,1,2,3,4],     dtype=np.int32)]

  '''
  part1 = isosurface or extract_part (only defined here by its lngn `p1_lngn` 
          and parent connectivity `p1_to_p2*`)
  part2 = volumic parent (defined by the yaml tree `pt`)

  To get part_gnum1: start from ref_lnum2, extract gnum2 using p2_lngn; then search
  the positions of these gnum in p1_to_p2. Finaly, read p1_lngn at these positions

  PROC 0:
    part0:
      part_gnum1_idx = [0 1 2]
      part_gnum1     = [2 4]   
      ref_lnum2      = [1 2] 
        --> first two elts of part2 on proc0
            are linked with [2 4] of part1 
            (proc1: p1_to_p2 = [1,2,x,x] and p1_lngn = [2,4,x,x])
      PL_part2 = [2 4 1], elts 2 and 1 are a parent of an elt of part1
      (because 2 and 1 are present in ref_lnum2)
        --> pl_gnum1 = [2 0] to put data on same order than part1_gnum
            (pl_part2[pl_gnum1]=[1 2])
            stride = [1 1] because elts [1 2] of part2 must send data
    part1:
      part_gnum1_idx = [0 1 2]
      part_gnum1     = [6 8]
      ref_lnum2      = [1 2]
        --> first two elts of part2 on proc0
            are linked with [6 8] of part1 
            (proc1: p1_to_p2 = [x,x,6,7] and p1_lngn = [x,x,6,8])
      PL_part2 = [] 
      (because no element of pl occurs in ref_lnum2)
        --> pl_gnum1=[] 
            stride  =[0 0] because no elts of part2 must send data

  PROC 1:
    part0:
      part_gnum1_idx = [0 2 4 5]
      part_gnum1 = [1 5 3 7 9]
      ref_lnum2  = [1 2 5]
      --> elts [1 2 5] of part2 on proc1
          are linked with [1,5,3,7,9] of part1 
          (proc0: p1_to_p2 = [9,10,9,10,13]
           order is not the same because of
           part_to_part)
      PL_part2 = [4 2], elts 10 only is a parent of two elt of part1 (12 not appears)
      (only 2 is present in ref_lnum2, at position 1)
        --> pl_gnum1 = [1 1] to put data on same order than part1_gnum
            and duplicate it (pl_part2[pl_gnum1]=[2 2] which reference to elt 10)
            stride = [0 0 1 1 0] because 10 will send data to 2 elts of part1
            which are 3 and 7
  '''


  part_tree = parse_yaml_cgns.to_cgns_tree(pt)
  part_zones = PT.get_all_Zone_t(part_tree)

  # > P2P Object
  ptp = PDM.PartToPart(sub_comm, p1_lngn, p2_lngn, p1_to_p2_idx, p1_to_p2)

  # Container 
  pl_gnum1, stride = EU.get_partial_container_stride_and_order(part_zones, 'FSol_A', 'Vertex', ptp, sub_comm)
  if sub_comm.Get_rank()==0:
    assert np.array_equal(pl_gnum1[0], np.array([2,0], dtype=pdm_gnum_dtype))
    assert np.array_equal(pl_gnum1[1], np.array([   ], dtype=pdm_gnum_dtype))
    assert np.array_equal(  stride[0], np.array([1,1], dtype=pdm_gnum_dtype))
    assert np.array_equal(  stride[1], np.array([0,0], dtype=pdm_gnum_dtype))
  if sub_comm.Get_rank()==1:
    assert np.array_equal(pl_gnum1[0], np.array([1,1],       dtype=pdm_gnum_dtype))
    assert np.array_equal(  stride[0], np.array([0,0,1,1,0], dtype=pdm_gnum_dtype))
  
