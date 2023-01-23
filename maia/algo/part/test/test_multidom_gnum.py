import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

from maia.pytree.yaml import parse_yaml_cgns

import maia.algo.part.multidom_gnum as MGM

@mark_mpi_test(3)
def test_get_shifted_arrays(sub_comm):
  if sub_comm.Get_rank() == 0:
    arrays = [[np.array([1,8,4])],  [np.array([3,6,1,9])]]
    expected = [[np.array([1,8,4])],  [np.array([13,16,11,19])]]
  elif sub_comm.Get_rank() == 1:
    arrays = [[],  [np.array([9,3,1,4]), np.array([8,6])]]
    expected = [[],  [np.array([19,13,11,14]), np.array([18,16])]]
  elif sub_comm.Get_rank() == 2:
    arrays = [[np.array([10,2])],  [np.empty(0)]]
    expected = [[np.array([10,2])],  [np.empty(0)]]

  offset, shifted_arrays = MGM._get_shifted_arrays(arrays, sub_comm)
  assert (offset == [0,10,19]).all()
  for i in range(2):
    for t1, t2 in zip(shifted_arrays[i], expected[i]):
      assert (t1 == t2).all()


@mark_mpi_test(2)
@pytest.mark.parametrize("unify_jns", [False, True])
def test_get_mdom_gnum_vtx(unify_jns, sub_comm):
  if sub_comm.Get_rank() == 0:
    yt = """
    ZoneA.P0.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t [4,2,1,3,5]:
      ZGC ZoneGridConnectivity_t:
        matchAB.0 GridConnectivity_t "ZoneB.P1.N0":
          GridConnectivityDonorName Descriptor_t "matchBA.0":
          GridLocation GridLocation_t "Vertex":
          PointList IndexArray_t [[4,5]]: 
          PointListDonor IndexArray_t [[2,1]]: 
          :CGNS#GlobalNumbering UserDefinedData_t:
            Index DataArray_t [1,2]:
    ZoneB.P0.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t [4,1]:
    """
    if unify_jns:
      expected_gn = [[np.array([3,2,1,6,7])], [np.array([5,4])]]
    else:
      expected_gn = [[np.array([4,2,1,3,5])], [np.array([9,6])]]
  elif sub_comm.Get_rank() == 1:
    yt = """
    ZoneB.P1.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t [2,3]:
      ZGC ZoneGridConnectivity_t:
        matchBA.0 GridConnectivity_t "ZoneA.P0.N0":
          GridConnectivityDonorName Descriptor_t "matchAB.0":
          GridLocation GridLocation_t "Vertex":
          PointList IndexArray_t [[2,1]]: 
          PointListDonor IndexArray_t [[4,5]]: 
          :CGNS#GlobalNumbering UserDefinedData_t:
            Index DataArray_t [1,2]:
    """
    if unify_jns:
      expected_gn = [[], [np.array([7,6])]]
    else:
      expected_gn = [[], [np.array([7,8])]]

  all_part_zones = parse_yaml_cgns.to_nodes(yt)
  part_per_doms = {'Base/ZoneA' : [z for z in all_part_zones if 'ZoneA' in z[0]],
                   'Base/ZoneB' : [z for z in all_part_zones if 'ZoneB' in z[0]]}

  global_gnum = MGM.get_mdom_gnum_vtx(part_per_doms, sub_comm, unify_jns)

  for i_dom, parts in enumerate(part_per_doms.values()):
    for i_part in range(len(parts)):
      assert np.array_equal(global_gnum[i_dom][i_part], expected_gn[i_dom][i_part])

