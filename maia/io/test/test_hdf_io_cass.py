import pytest

know_cassiopee = True
try:
  import Converter
except ImportError:
  know_cassiopee = False

import maia.pytree as PT
from maia.pytree.yaml import parse_yaml_cgns

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
def test_add_sizes_to_zone_tree():
  from maia.io import _hdf_io_cass as LC
  yt = """
Zone Zone_t:
  Hexa Elements_t [17, 0]:
    ElementConnectivity DataArray_t None:
  ZBC ZoneBC_t:
    bc BC_t "farfield":
      PointList IndexArray_t None:
    bc_withds BC_t "farfield":
      PointList IndexArray_t None:
      BCDataSet BCDataSet_t:
        PointList IndexArray_t None:
  ZGC ZoneGridConnectivity_t:
    gc GridConnectivity_t:
      PointList IndexArray_t None:
      PointListDonor IndexArray_t None:
  ZSR ZoneSubRegion_t:
    PointList IndexArray_t None:
  FS FlowSolution_t:
  FSPL FlowSolution_t:
    PointList IndexArray_t None:
"""
  zone = parse_yaml_cgns.to_node(yt)
  size_data = {'/Zone/Hexa/ElementConnectivity' : (1, 'I4', 160),
               '/Zone/ZBC/bc/PointList' : (1, 'I4', (1,30)),
               '/Zone/ZBC/bc_withds/PointList' : (1, 'I4', (1,100)),
               '/Zone/ZBC/bc_withds/BCDataSet/PointList' : (1, 'I4', (1,10)),
               '/Zone/ZGC/gc/PointList' : (1, 'I4', (1,20)),
               '/Zone/ZGC/gc/PointListDonor' : (1, 'I4', (1,20)),
               '/Zone/ZSR/PointList' : (1, 'I4', (1,34)),
               '/Zone/FSPL/PointList' : (1, 'I4', (1,10)),
              }

  LC.add_sizes_to_zone_tree(zone, '/Zone', size_data)

  assert PT.get_node_from_path(zone, 'Hexa/ElementConnectivity#Size') is None

  assert (PT.get_node_from_path(zone, 'ZBC/bc/PointList#Size')[1] == [1,30]).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc_withds/PointList#Size')[1] == [1,100]).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc_withds/BCDataSet/PointList#Size')[1] == [1,10]).all()

  assert (PT.get_node_from_path(zone, 'ZGC/gc/PointList#Size')[1] == [1,20]).all()

  assert (PT.get_node_from_path(zone, 'ZSR/PointList#Size')[1] == [1,34]).all()

  assert (PT.get_node_from_path(zone, 'FSPL/PointList#Size')[1] == [1,10]).all()
  assert (PT.get_node_from_path(zone, 'FS/PointList#Size') is None)

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
def test_add_sizes_to_tree():
  from maia.io import _hdf_io_cass as LC
  yt = """
BaseA CGNSBase_t:
  Zone Zone_t:
    Hexa Elements_t [17, 0]:
      ElementConnectivity DataArray_t None:
    ZBC ZoneBC_t:
      bc BC_t "farfield":
        PointList IndexArray_t None:
      bc_withds BC_t "farfield":
        PointList IndexArray_t None:
        BCDataSet BCDataSet_t:
          PointList IndexArray_t None:
    ZGC ZoneGridConnectivity_t:
      gc GridConnectivity_t:
        PointList IndexArray_t None:
        PointListDonor IndexArray_t None:
    ZSR ZoneSubRegion_t:
      PointList IndexArray_t None:
"""
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  size_data_tree = {'/BaseA/Zone/Hexa/ElementConnectivity' : (1, 'I4', 160),
                    '/BaseA/Zone/ZBC/bc/PointList' : (1, 'I4', (1,30)),
                    '/BaseA/Zone/ZBC/bc_withds/PointList' : (1, 'I4', (1,100)),
                    '/BaseA/Zone/ZBC/bc_withds/BCDataSet/PointList' : (1, 'I4', (1,10)),
                    '/BaseA/Zone/ZGC/gc/PointList' : (1, 'I4', (1,20)),
                    '/BaseA/Zone/ZGC/gc/PointListDonor' : (1, 'I4', (1,20)),
                    '/BaseA/Zone/ZSR/PointList' : (1, 'I4', (1,34)),
                   }
  LC.add_sizes_to_tree(tree, size_data_tree)
  assert len(PT.get_nodes_from_name(tree, '*#Size')) == 6
