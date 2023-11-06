import pytest

import maia.pytree as PT

from maia.pytree.yaml import parse_yaml_cgns
from maia.pytree.node import shorten_names, shorten_field_names
from maia.pytree.node import rename_zone

class Test_shorten_names:
  yt = """
  MyVeryLooooonnnggggFlowSolutionName FlowSolution_t:
    TurbulentSANuTildeDensityGradientX DataArray_t:
    TurbulentSANuTildeDensityGradientY DataArray_t:
    MyShortName DataArray_t:
  """
  node = parse_yaml_cgns.to_node(yt)

  def test_shorten_field_names(self):
      shorten_field_names(self.node,quiet=True)

      expected_yt = """
      MyVeryLooooonnnggggFlowSolutionName FlowSolution_t:
        TurbSANuTildDensGradX DataArray_t:
        TurbSANuTildDensGradY DataArray_t:
        MyShortName DataArray_t:
      """
      expected_node = parse_yaml_cgns.to_node(expected_yt)
      assert self.node == expected_node

  def test_shorten_names(self):
      shorten_names(self.node,quiet=True)

      expected_yt = """
      MyVeryLoooFlowSoluName FlowSolution_t:
        TurbSANuTildDensGradX DataArray_t:
        TurbSANuTildDensGradY DataArray_t:
        MyShortName DataArray_t:
      """
      expected_node = parse_yaml_cgns.to_node(expected_yt)
      print(self.node)
      assert self.node == expected_node

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
  rename_zone(t, 'ZoneB', 'ZoneBB')
  assert PT.get_node_from_name(t, 'ZoneB') is None
  assert PT.get_node_from_name(t, 'ZoneBB') is not None

  assert PT.get_value(PT.get_node_from_name(t, 'match1')) == 'ZoneA'
  assert PT.get_value(PT.get_node_from_name(t, 'match2')) == 'ZoneBB'
  assert PT.get_value(PT.get_node_from_name(t, 'match3')) == 'ZoneBB'
