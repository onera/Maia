import pytest

import maia.pytree as PT

from maia.pytree.sids import adjust

def test_enforceDonorAsPath():
  yt = """
  Base CGNSBase_t:
    ZoneA Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "ZoneA":
        match2 GridConnectivity_t "Base/ZoneA":
        match3 GridConnectivity1to1_t "ZoneB":
    ZoneB Zone_t:
      ZGC1 ZoneGridConnectivity_t:
        match4 GridConnectivity1to1_t "ZoneA":
  """
  tree = PT.yaml.parse_yaml_cgns.to_cgns_tree(yt)
  adjust.enforceDonorAsPath(tree)
  assert PT.get_value(PT.get_node_from_name(tree, "match1")) == "Base/ZoneA"
  assert PT.get_value(PT.get_node_from_name(tree, "match2")) == "Base/ZoneA"
  assert PT.get_value(PT.get_node_from_name(tree, "match3")) == "Base/ZoneB"
  assert PT.get_value(PT.get_node_from_name(tree, "match4")) == "Base/ZoneA"

@pytest.mark.parametrize('mode', ['move', 'copy'])
def test_zsr_fields_to_bc(mode):
  yt = """
  Base CGNSBase_t:
    Zone Zone_t:
      ZoneBC ZoneBC_t:
        BC1 BC_t:
        BC2 BC_t:
      UnlinkedZSR ZoneSubRegion_t:
        field DataArray_t [1,2,3,4]:
      LinkedZSR ZoneSubRegion_t:
        field DataArray_t [10,20,30,40]:
        BCRegionName Descriptor_t "BC2":
  """
  tree = PT.yaml.parse_yaml_cgns.to_cgns_tree(yt)
  adjust.subregion_fields_to_bcdataset(tree, mode)
  assert (PT.get_node_from_path(tree, 'Base/Zone/ZoneBC/BC2/LinkedZSR/DirichletData/field')[1] == [10, 20, 30, 40]).all()
  if mode == 'move':
    assert PT.get_node_from_path(tree, 'Base/Zone/LinkedZSR/field') is None
  assert PT.get_node_from_path(tree, 'Base/Zone/LinkedZSR') is not None