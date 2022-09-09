import pytest
import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.pytree.yaml import parse_yaml_cgns
from maia.io.hdf import cgns_elements

def test_gen_elemts():
  yt = """
Zone Zone_t [[27],[8],[0]]:
  NGon Elements_t [22, 0]:
  Hexa Elements_t [17, 0]:
  SomeOtherNode OtherType_t:
"""
  zone = parse_yaml_cgns.to_node(yt)
  elmt_gen = cgns_elements.gen_elemts(zone)
  assert hasattr(elmt_gen, '__next__')
  assert [I.getName(elt) for elt in elmt_gen] == ['NGon', 'Hexa']

def test_create_zone_std_elements_filter():
  yt = """
Hexa Elements_t [17, 0]:
  :CGNS#Distribution UserDefinedData_t:
    Element DataArray_t [2,7,10]:
Tri Elements_t [5, 0]:
  ParentElements DataArray_t:
  ParentElementsPosition DataArray_t:
  :CGNS#Distribution UserDefinedData_t:
    Element DataArray_t [40,60,60]:
"""
  elements = parse_yaml_cgns.to_nodes(yt)
  hdf_filter = dict()
  cgns_elements.create_zone_std_elements_filter(elements[0], "path/to/zone", hdf_filter)
  cgns_elements.create_zone_std_elements_filter(elements[1], "path/to/zone", hdf_filter)
  assert len(hdf_filter) == 4
  assert hdf_filter['path/to/zone/Hexa/ElementConnectivity'] == \
      [[0], [1], [(7-2)*8], [1], [2*8], [1], [(7-2)*8], [1], [10*8], [0]]
  assert hdf_filter['path/to/zone/Tri/ElementConnectivity'] == \
      [[0], [1], [(60-40)*3], [1], [40*3], [1], [(60-40)*3], [1], [60*3], [0]]
  assert hdf_filter['path/to/zone/Tri/ParentElements'] == \
      [[0,0], [1,1], [(60-40),2], [1,1], [40,0], [1,1], [(60-40),2], [1,1], [60,2], [1]]

def test_load_element_connectivity_from_eso():
  yt = """
NGon Elements_t [22, 0]:
  ParentElements DataArray_t None:
  ElementConnectivity#Size IndexArray_t [40]:
  ElementStartOffset DataArray_t [8,12,16,20,24,28]:
  :CGNS#Distribution UserDefinedData_t:
    Element DataArray_t [2,7,10]:
"""
  element = parse_yaml_cgns.to_node(yt)
  hdf_filter = dict()
  cgns_elements.load_element_connectivity_from_eso(element, 'pathtozone', hdf_filter)
  assert hdf_filter['pathtozone/NGon/ElementConnectivity'] == \
      [[0], [1], [28-8], [1], [8], [1], [28-8], [1], [40], [0]]
  element_connectivity_distri = I.getVal(MT.getDistribution(element, 'ElementConnectivity'))
  assert (element_connectivity_distri == [8,28,40]).all()
      

def test_create_zone_eso_elements_filter():
  yt = """
NGon Elements_t [22, 0]:
  ParentElements DataArray_t None:
  ElementStartOffset DataArray_t None:
  ElementConnectivity DataArray_t None:
  :CGNS#Distribution UserDefinedData_t:
    Element DataArray_t [2,7,10]:
"""
  element = parse_yaml_cgns.to_node(yt)
  read_filter, write_filter = dict(), dict()
  cgns_elements.create_zone_eso_elements_filter(element, 'pathtozone', read_filter, 'read')
  cgns_elements.create_zone_eso_elements_filter(element, 'pathtozone', write_filter, 'write')
  assert read_filter['pathtozone/NGon/ParentElements'] == \
      [[0, 0], [1, 1], [(7-2), 2], [1, 1], [2, 0], [1, 1], [(7-2), 2], [1, 1], [10, 2], [1]]
  assert read_filter['pathtozone/NGon/ElementStartOffset'] == \
      [[0], [1], [(7-2)+1], [1], [2], [1], [(7-2)+1], [1], [10+1], [0]]
  assert write_filter['pathtozone/NGon/ParentElements'] == read_filter['pathtozone/NGon/ParentElements']
  assert write_filter['pathtozone/NGon/ElementStartOffset'] == \
      [[0], [1], [(7-2)], [1], [2], [1], [(7-2)], [1], [10+1], [0]]
  partial_func = read_filter['pathtozone/NGon/ElementConnectivity']
  assert partial_func.func is cgns_elements.load_element_connectivity_from_eso
  assert partial_func.args == (element, 'pathtozone')

def test_create_zone_mixed_elements_filter():
  hdf_filter = dict()
  element = I.newElements('Mixed', 'MIXED')
  with pytest.raises(NotImplementedError):
    cgns_elements.create_zone_mixed_elements_filter(element, "path/to/zone", hdf_filter)

def test_create_zone_elements_filter():
  yt = """
Zone Zone_t:
  NGon Elements_t [22, 0]:
    ParentElements DataArray_t None:
    ElementStartOffset DataArray_t None:
    ElementConnectivity DataArray_t None:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t [2,7,10]:
  Tri Elements_t [5, 0]:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t [30,60,120]:
"""
  zone = parse_yaml_cgns.to_node(yt)
  hdf_filter = dict()
  cgns_elements.create_zone_elements_filter(zone, 'zone', hdf_filter, 'read')
  ngon = PT.get_node_from_name(zone, 'NGon')
  tri  = PT.get_node_from_name(zone, 'Tri')
  ngon_filter, tri_filter = dict(), dict()
  cgns_elements.create_zone_eso_elements_filter(ngon, 'zone', ngon_filter, 'read')
  cgns_elements.create_zone_std_elements_filter(tri, 'zone', tri_filter)
  for key,value in tri_filter.items():
    assert hdf_filter[key] == value
  for key,value in ngon_filter.items():
    if 'ElementConnectivity' in key:
      assert value.func == hdf_filter[key].func and value.args == hdf_filter[key].args
    else:
      assert hdf_filter[key] == value
  assert len(hdf_filter) == (len(ngon_filter) + len(tri_filter))

