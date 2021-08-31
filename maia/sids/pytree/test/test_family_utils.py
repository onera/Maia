import pytest
import numpy as np

import Converter.Internal as I

from maia.sids.cgns_keywords import Label as CGL
from maia.sids import pytree as PT
from maia.sids.pytree import family_utils as FU

from maia.utils        import parse_yaml_cgns


def test_getFamily():
  yt = """
Base CGNSBase_t [3,3]:
  SecondFamily Family_t:
  FirstFamily Family_t:
"""
  node = parse_yaml_cgns.to_node(yt)
  assert FU.getFamily(node, 'FirstFamily') == PT.getNodeFromName(node, 'FirstFamily')
  with pytest.raises(ValueError):
    FU.getFamily(node, 'ThirdFamily')

def test_getGridConnectivitiesFromFamily():
  yt = """
ZGC ZoneGridConnectivity_t:
  GC1 GridConnectivity_t:
    FamilyName FamilyName_t "SecondFamily":
  GC2 GridConnectivity_t:
    FamilyName FamilyName_t "SecondFamily":
  GC3 GridConnectivity_t:
    FamilyName FamilyName_t "FirstFamily":
"""
  node = parse_yaml_cgns.to_node(yt)
  assert FU.getGridConnectivitiesFromFamily(node, 'FirstFamily') == PT.getNodeFromName(node, "GC3")
  with pytest.raises(ValueError):
    FU.getGridConnectivitiesFromFamily(node, 'ThirdFamily')

  assert FU.getAllGridConnectivitiesFromFamily(node, 'SecondFamily') == \
      [PT.getNodeFromName(node, "GC1"), PT.getNodeFromName(node, "GC2")]

  assert FU.getAllGridConnectivitiesFromFamily(node, 'ThirdFamily') == []

  assert list(FU.iterAllGridConnectivitiesFromFamily(node, 'SecondFamily')) == \
      [PT.getNodeFromName(node, "GC1"), PT.getNodeFromName(node, "GC2")]

def test_getGridConnectivitiesFromAdditionalFamily():
  yt = """
ZGC ZoneGridConnectivity_t:
  GC1 GridConnectivity_t:
    AdditionalFamilyName AdditionalFamilyName_t "SecondFamily":
  GC2 GridConnectivity_t:
    AdditionalFamilyName AdditionalFamilyName_t "SecondFamily":
  GC3 GridConnectivity_t:
    AdditionalFamilyName AdditionalFamilyName_t "FirstFamily":
"""
  node = parse_yaml_cgns.to_node(yt)
  assert FU.getGridConnectivitiesFromAdditionalFamily(node, 'FirstFamily') == PT.getNodeFromName(node, "GC3")
  with pytest.raises(ValueError):
    FU.getGridConnectivitiesFromAdditionalFamily(node, 'ThirdFamily')

  assert FU.getAllGridConnectivitiesFromAdditionalFamily(node, 'SecondFamily') == \
      [PT.getNodeFromName(node, "GC1"), PT.getNodeFromName(node, "GC2")]

  assert FU.getAllGridConnectivitiesFromAdditionalFamily(node, 'ThirdFamily') == []

  assert list(FU.iterAllGridConnectivitiesFromAdditionalFamily(node, 'SecondFamily')) == \
      [PT.getNodeFromName(node, "GC1"), PT.getNodeFromName(node, "GC2")]

def test_iterFromLabelsAndFamily():
  yt = """
ZGC ZoneGridConnectivity_t:
  GC1 GridConnectivity_t:
    FamilyName FamilyName_t "SecondFamily":
  GC2 GridConnectivity_t:
    FamilyName FamilyName_t "SecondFamily":
  GC3 GridConnectivity_t:
    FamilyName FamilyName_t "FirstFamily":
  GC3 GridConnectivity1to1_t:
    FamilyName FamilyName_t "SecondFamily":
"""
  node = parse_yaml_cgns.to_node(yt)

  assert [I.getName(n) for n in FU.iterFromLabelsAndFamily(node, ['GridConnectivity_t'], 'SecondFamily')] == ["GC1", "GC2"]

  assert [I.getName(n) for n in FU.getFromLabelsAndFamily(node, ['GridConnectivity_t', 'GridConnectivity1to1_t'], 'SecondFamily')] == ["GC1", "GC2", "GC3"]

  with pytest.raises(ValueError):
    FU.getOneFromLabelsAndFamily(node, ['BC_t'], 'SecondFamily')
