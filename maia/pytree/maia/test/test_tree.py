import pytest
import pytest_parallel
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.pytree.yaml   import parse_yaml_cgns

from maia.pytree.maia import tree

@pytest_parallel.mark.parallel(3)
def test_rename_zones(comm):
  if comm.Get_rank() == 0:
    tree = parse_yaml_cgns.to_cgns_tree("""
    Base CGNSBase_t:
      ZoneA Zone_t:
      ZoneB Zone_t:
        ZGC ZoneGridConnectivity_t:
          match1 GridConnectivity_t "ZoneD":
    """)
    old_to_new = {'Base/ZoneA' : 'Base/ZoneI', 'Base/ZoneB': 'Base/ZoneII'}
  elif comm.Get_rank() == 1:
    tree = parse_yaml_cgns.to_cgns_tree("""
    Base CGNSBase_t:
      ZoneC Zone_t:
        ZGC ZoneGridConnectivity_t:
          PerioA GridConnectivity_t "Base/ZoneC":
          PerioB GridConnectivity_t "ZoneC":
    """)
    old_to_new = {'Base/ZoneC' : 'Base/ZoneIII'}
  elif comm.Get_rank() == 2:
    tree = parse_yaml_cgns.to_cgns_tree("""
    Base CGNSBase_t:
      ZoneD Zone_t:
        ZGC ZoneGridConnectivity_t:
          match2 GridConnectivity_t "ZoneB":
    """)
    old_to_new = {'Base/ZoneD' : 'Base/ZoneIV'}

  MT.tree.rename_zones(tree, old_to_new, comm)

  if comm.Get_rank() == 0:
    assert PT.get_all_Zone_t(tree)[1][0] == 'ZoneII'
    assert PT.get_value(PT.get_node_from_name(tree, 'match1')) == 'Base/ZoneIV'
  elif comm.Get_rank() == 1:
    assert PT.get_value(PT.get_node_from_name(tree, 'PerioA')) == 'Base/ZoneIII'
    assert PT.get_value(PT.get_node_from_name(tree, 'PerioB')) == 'Base/ZoneIII'
  elif comm.Get_rank() == 2:
    assert PT.get_all_Zone_t(tree)[0][0] == 'ZoneIV'
    assert PT.get_value(PT.get_node_from_name(tree, 'match2')) == 'Base/ZoneII'
