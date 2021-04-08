from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I

from maia.utils         import parse_yaml_cgns
from maia.generate import disttree_from_parttree as DFP

def test_match_jn_from_ordinals():
  dt = """
Base CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      perio1 GridConnectivity_t:
        Ordinal UserDefinedData_t [1]:
        OrdinalOpp UserDefinedData_t [2]:
        PointList IndexArray_t [[1,3]]:
      perio2 GridConnectivity_t:
        Ordinal UserDefinedData_t [2]:
        OrdinalOpp UserDefinedData_t [1]:
        PointList IndexArray_t [[2,4]]:
      match1 GridConnectivity_t:
        Ordinal UserDefinedData_t [3]:
        OrdinalOpp UserDefinedData_t [4]:
        PointList IndexArray_t [[10,100]]:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      match2 GridConnectivity_t:
        Ordinal UserDefinedData_t [4]:
        OrdinalOpp UserDefinedData_t [3]:
        PointList IndexArray_t [[-100,-10]]:
  """
  dist_tree = parse_yaml_cgns.to_complete_pytree(dt)
  DFP.match_jn_from_ordinals(dist_tree)
  expected_names  = ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA']
  expected_pl_opp = [[2,4], [1,3], [-100,-10], [10,100]]
  for i, jn in enumerate(I.getNodesFromType(dist_tree, 'GridConnectivity_t')):
    assert I.getValue(jn) == expected_names[i]
    assert (I.getNodeFromName1(jn, 'PointListDonor')[1] == expected_pl_opp[i]).all()

