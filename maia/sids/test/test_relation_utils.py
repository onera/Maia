import pytest
import Converter.Internal as I

from maia.utils               import parse_yaml_cgns
from maia.sids.relation_utils import get_subregion_extent

def test_get_subregion_extent():
  yt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
    BC2 BC_t:
  ZGC1 ZoneGridConnectivity_t:
    GCA GridConnectivity_t:
    GCB GridConnectivity_t:
  ZGC2 ZoneGridConnectivity_t:
    GC1to1A GridConnectivity1to1_t:
    GC1to1B GridConnectivity1to1_t:
  UnLinkedZSR ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[]]:
  BCLinkedZSR ZoneSubRegion_t:
    BCRegionName Descriptor_t "BC2":
  GCLinkedZSR ZoneSubRegion_t:
    GridConnectivityRegionName Descriptor_t "GC1to1B":
  OrphelanZSR ZoneSubRegion_t:
    BCRegionName Descriptor_t "BC9":
  WrongZSR WrongType_t:
    BCRegionName Descriptor_t "BC":
  """
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  zone = I.getZones(tree)[0]

  assert get_subregion_extent(I.getNodeFromName(zone, 'UnLinkedZSR'), zone) == 'UnLinkedZSR'
  assert get_subregion_extent(I.getNodeFromName(zone, 'BCLinkedZSR'), zone) == 'ZBC/BC2'
  assert get_subregion_extent(I.getNodeFromName(zone, 'GCLinkedZSR'), zone) == 'ZGC2/GC1to1B'

  with pytest.raises(ValueError):
    get_subregion_extent(I.getNodeFromName(zone, 'OrphelanZSR'), zone)
  with pytest.raises(AssertionError):
    get_subregion_extent(I.getNodeFromName(zone, 'WrongZSR'), zone)
