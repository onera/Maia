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

@pytest.mark.parametrize('mode', ['move', 'copy', 'view'])
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

  bcds_field = PT.get_node_from_path(tree, 'Base/Zone/ZoneBC/BC2/LinkedZSR/DirichletData/field')[1]
  assert (bcds_field == [10, 20, 30, 40]).all()
  if mode == 'move':
    assert PT.get_node_from_path(tree, 'Base/Zone/LinkedZSR/field') is None
  else:
    zsr_field = PT.get_node_from_path(tree, 'Base/Zone/LinkedZSR/field')[1]
    assert (zsr_field is bcds_field) == (mode == 'view')
  assert PT.get_node_from_path(tree, 'Base/Zone/LinkedZSR') is not None
  assert PT.get_node_from_path(tree, 'Base/Zone/UnlinkedZSR/field') is not None

@pytest.mark.parametrize('mode', ['move', 'view', 'copy'])
def test_zsr_fields_from_bc(mode):
  yt = """
  Zone Zone_t:
    ZoneBC ZoneBC_t:
      BC1 BC_t:
      BC2 BC_t:
        PointList IndexArray_t [[2,4,6,8]]:
        GridLocation GridLocation_t "Vertex":
        LinkedZSR BCDataSet_t:
          DirichletData BCData_t:
            LocalData DataArray_t [10,20,30,40]:
            GlobalData DataArray_t [18.]:
        RandomBCDS BCDataSet_t:
          DirichletData BCData_t:
            RandomData DataArray_t [42.]:
    UnlinkedZSR ZoneSubRegion_t:
      field DataArray_t [1,2,3,4]:
    LinkedZSR ZoneSubRegion_t:
      BCRegionName Descriptor_t "BC2":
  """
  zone = PT.yaml.parse_yaml_cgns.to_node(yt)

  if mode == 'view':
    with pytest.warns():
      adjust.subregion_fields_from_bcdataset(zone, mode)
  else:
    adjust.subregion_fields_from_bcdataset(zone, mode)

  assert (PT.get_node_from_path(zone, 'LinkedZSR/GlobalData')[1] == [18., 18, 18, 18]).all()
  assert (PT.get_node_from_path(zone, 'LinkedZSR/LocalData')[1] == [10, 20, 30, 40]).all()
  if mode == 'move':
    assert PT.get_node_from_path(zone, 'ZoneBC/BC2/LinkedZSR/DirichletData') is None
  else:
    bcds_field = PT.get_node_from_path(zone, 'ZoneBC/BC2/LinkedZSR/DirichletData/LocalData')[1]
    zsr_field  = PT.get_node_from_path(zone, 'LinkedZSR/LocalData')[1]
    assert (zsr_field is bcds_field) == (mode == 'view')
  assert PT.get_node_from_path(zone, 'ZoneBC/BC2/RandomBCDS') is not None
  assert len(PT.get_nodes_from_label(PT.get_node_from_path(zone, 'UnlinkedZSR'), 'DataArray_t')) == 1