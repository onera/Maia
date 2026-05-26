import pytest

import maia.pytree as PT

from maia.pytree.yaml import parse_yaml_cgns

from maia.pytree.node import name_utils as NU

class Test_shorten_names:
  yt = """
  MyVeryLooooonnnggggFlowSolutionName FlowSolution_t:
    TurbulentSANuTildeDensityGradientX DataArray_t:
    TurbulentSANuTildeDensityGradientY DataArray_t:
    MyShortName DataArray_t:
  """
  node = parse_yaml_cgns.to_node(yt)

  def test_shorten_field_names(self):
      NU.shorten_field_names(self.node,quiet=True)

      expected_yt = """
      MyVeryLooooonnnggggFlowSolutionName FlowSolution_t:
        TurbSANuTildDensGradX DataArray_t:
        TurbSANuTildDensGradY DataArray_t:
        MyShortName DataArray_t:
      """
      expected_node = parse_yaml_cgns.to_node(expected_yt)
      assert self.node == expected_node

  def test_shorten_names(self):
      NU.shorten_names(self.node,quiet=True)

      expected_yt = """
      MyVeryLoooFlowSoluName FlowSolution_t:
        TurbSANuTildDensGradX DataArray_t:
        TurbSANuTildDensGradY DataArray_t:
        MyShortName DataArray_t:
      """
      expected_node = parse_yaml_cgns.to_node(expected_yt)
      assert self.node == expected_node

def test_unambiguous_short_names():
  siblings = [
    'Name1',
    'Name2',
    'Name1',
  ]
  with pytest.raises(RuntimeError) as e:
    NU._unambiguous_short_names(siblings)
  assert str(e.value) == "There are two siblings of the same name among ['Name1', 'Name2', 'Name1']"

  siblings = [
    'Density',
    'AVeryLongFieldNameWithLotsOfDetailsAboutTurbulentDensityRootMeanSquareResidual1',
    'RSDTurbulentDissipationRateDensityRMS',
    'AnotherVeryLongFieldNameWithLotsOfDetailsAboutTurbulentDensityRootMeanSquareResidual',
    'AVeryLongFieldNameWithLotsOfDetailsAboutTurbulentDensityRootMeanSquareResidual2',
  ]

  assert NU._unambiguous_short_names(siblings) == [
    'Density',
    'AVeryLongFielNameWithLo.4fd4d905',
    'RSDTurbDissRateDensRMS',
    'AnotVeryLongFielNameWithLotsOfDe',
    'AVeryLongFielNameWithLo.628b0bac',
  ]

def test_short_name_with_hash():
   assert NU.short_name_with_hash('Short') == 'Short'
   name = 'AVeryLongFieldNameWithLotsOfDetailsAboutTurbulentDensityRootMeanSquareResidual1'
   assert NU.short_name_with_hash(name) == 'AVeryLongFielNameWithLo.a2497715'

   name = 'ALongNameAgainButThisOneIsVectorialX'
   assert NU.short_name_with_hash(name) == 'ALongNameAgaiButThisOn.81718ddfX'
   name = 'ALongNameAgainButThisOneIsVectorialY'
   assert NU.short_name_with_hash(name) == 'ALongNameAgaiButThisOn.81718ddfY'

   name = 'AndToFinishTheSameWithATensor:)XZ'
   assert NU.short_name_with_hash(name) == 'AndToFinishTheSameWit.2d92541cXZ'
   name = 'AndToFinishTheSameWithATensor:)ZY'
   assert NU.short_name_with_hash(name) == 'AndToFinishTheSameWit.2d92541cZY'

def test_get_full_name():
  assert NU.get_full_name(PT.new_Zone('SomeZoneNode')) == 'SomeZoneNode'
  node = PT.new_Zone('ShortenedName')
  PT.new_Descriptor(NU.FULL_NAME_NODE_NAME, 'TrueLongName', parent=node)
  assert NU.get_full_name(node) == 'TrueLongName'

def test_rename_zone():
  yt = """
  Base CGNSBase_t:
    ZoneA Zone_t:
    ZoneB Zone_t:
    ZoneC Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "ZoneA":
        match2 GridConnectivity_t "ZoneB":
        match3 GridConnectivity_t "ZoneB":
  """
  t = parse_yaml_cgns.to_cgns_tree(yt)
  NU.rename_zone(t, 'ZoneB', 'ZoneBB')
  assert PT.get_node_from_name(t, 'ZoneB') is None
  assert PT.get_node_from_name(t, 'ZoneBB') is not None

  assert PT.get_value(PT.get_node_from_name(t, 'match1')) == 'ZoneA'
  assert PT.get_value(PT.get_node_from_name(t, 'match2')) == 'ZoneBB'
  assert PT.get_value(PT.get_node_from_name(t, 'match3')) == 'ZoneBB'
