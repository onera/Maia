from pytest_mpi_check._decorator import mark_mpi_test
import mpi4py.MPI as MPI
import numpy      as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from   maia.pytree.yaml    import parse_yaml_cgns
from   maia.utils.parallel import utils as par_utils
from   maia.io             import distribution_tree

@mark_mpi_test(3)
def test_create_distribution_node(sub_comm):
  node = PT.new_node('ParentNode', 'UserDefinedData_t')
  distribution_tree.create_distribution_node(100, sub_comm, 'MyDistribution', node)

  distri_ud   = MT.getDistribution(node)
  assert distri_ud is not None
  assert PT.get_label(distri_ud) == 'UserDefinedData_t'
  distri_node = PT.get_child_from_name(distri_ud, 'MyDistribution')
  assert distri_node is not None
  assert (PT.get_value(distri_node) == par_utils.uniform_distribution(100, sub_comm)).all()
  assert PT.get_label(distri_node) == 'DataArray_t'

@mark_mpi_test(2)
def test_compute_plist_or_prange_distribution(sub_comm):
  node = PT.new_BC(name='BC', point_range=[[1,3],[1,3],[3,3]])
  distribution_tree.compute_plist_or_prange_distribution(node, sub_comm)

  distrib_ud = MT.getDistribution(node)
  assert PT.get_label(distrib_ud) == 'UserDefinedData_t'
  distrib    = PT.get_child_from_name(distrib_ud, 'Index')
  assert PT.get_label(distrib) == 'DataArray_t'
  assert (PT.get_value(distrib) == par_utils.uniform_distribution(3*3*1, sub_comm)).all()

  node = PT.new_BC(name='BC')
  PT.new_PointList('PointList', None, parent=node)
  PT.new_PointList('PointList#Size', [1,9], parent=node)
  distribution_tree.compute_plist_or_prange_distribution(node, sub_comm)

  distrib_ud = MT.getDistribution(node)
  assert PT.get_label(distrib_ud) == 'UserDefinedData_t'
  distrib    = PT.get_child_from_name(distrib_ud, 'Index')
  assert PT.get_label(distrib) == 'DataArray_t'
  assert (PT.get_value(distrib) == par_utils.uniform_distribution(1*9, sub_comm)).all()

@mark_mpi_test(2)
def test_compute_elements_distribution(sub_comm):
  zoneS = PT.new_Zone('ZoneS', type='Structured')
  zoneU = PT.new_Zone('ZoneS', type='Unstructured')
  hexa = PT.new_Elements('Hexa', 'HEXA_8', erange=[1,100],parent=zoneU)
  tri  = PT.new_Elements('Tri', 'TRI_3', erange=[101,1000],parent=zoneU)
  distribution_tree.compute_elements_distribution(zoneS, sub_comm)
  distribution_tree.compute_elements_distribution(zoneU, sub_comm)
  assert (PT.get_node_from_name(hexa, 'Element')[1] == \
      par_utils.uniform_distribution(100, sub_comm)).all()
  assert (PT.get_node_from_name(tri , 'Element')[1] == \
      par_utils.uniform_distribution(900, sub_comm)).all()
  assert PT.get_node_from_name(zoneS, 'Element') == None



@mark_mpi_test(2)
class Test_compute_zone_distribution:
  def test_unstruct(self, sub_comm):
    yt = """
Zone Zone_t [[27,8,0]]:  
  ZoneType ZoneType_t "Unstructured":
  Ngon Elements_t [22,0]:
    ElementRange IndexArray_t [1,36]:
  ZBC ZoneBC_t:
    bc1 BC_t "Farfield":
      PointList IndexArray_t None:
      PointList#Size IndexArray_t [1,4]:
      bcds BCDataSet_t:
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,2]:
  ZGC ZoneGridConnectivity_t:
    match GridConnectivity_t "otherzone":
      PointList IndexArray_t None:
      PointListDonor IndexArray_t None:
      PointList#Size IndexArray_t [1,4]:
  ZSR ZoneSubRegion_t:
    PointList IndexArray_t None:
    PointList#Size IndexArray_t [1,12]:
  FS FlowSolution_t:
    PointList IndexArray_t None:
    PointList#Size IndexArray [1,10]:
  """
    zone = parse_yaml_cgns.to_node(yt)
    distribution_tree.compute_zone_distribution(zone, sub_comm)
    assert len(PT.get_nodes_from_name(zone, 'Index')) == 5
    assert len(PT.get_nodes_from_name(zone, 'Element')) == 1

  def test_struct(self, sub_comm):
    yt = """
Zone Zone_t [[3,3,3],[2,2,2],[0,0,0]]:
  ZoneType ZoneType_t "Structured":
  ZBC ZoneBC_t:
    bc1 BC_t "Farfield":
      PointRange IndexRange_t [[1,3],[1,3],[1,1]]:
      bcds BCDataSet_t:
        PointRange IndexRange_t [[1,3],[1,1],[1,1]]:
  ZSR ZoneSubRegion_t:
    PointRange IndexRange_t [[2,2],[2,2],[1,1]]:
  """
    zone = parse_yaml_cgns.to_node(yt)
    distribution_tree.compute_zone_distribution(zone, sub_comm)
    assert PT.get_node_from_name(zone, 'PointList#Size') is None
    assert len(PT.get_nodes_from_name(zone, 'Index')) == 3


@mark_mpi_test(2)
def test_add_distribution_info(sub_comm):
  dist_tree = parse_yaml_cgns.to_cgns_tree("""
Base CGNSBase_t [3,3]:
  ZoneU Zone_t [[27,8,0]]:
    ZoneType ZoneType_t "Unstructured":
    Ngon Elements_t [22,0]:
      ElementRange IndexArray_t [1,36]:
    ZBC ZoneBC_t:
      bc1 BC_t "Farfield":
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,4]:
        bcds BCDataSet_t:
          PointList IndexArray_t None:
          PointList#Size IndexArray_t [1,2]:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "otherzone":
        PointList IndexArray_t None:
        PointListDonor IndexArray_t None:
        PointList#Size IndexArray_t [1,4]:
    ZSR ZoneSubRegion_t:
      PointList IndexArray_t None:
      PointList#Size IndexArray_t [1,12]:
  ZoneS Zone_t [[3,3,3],[2,2,2],[0,0,0]]:
    ZoneType ZoneType_t "Structured":
    ZBC ZoneBC_t:
      bc1 BC_t "Farfield":
        PointRange IndexRange_t [[1,3],[1,3],[1,1]]:
        bcds BCDataSet_t:
          PointRange IndexRange_t [[1,3],[1,1],[1,1]]:
    ZSR ZoneSubRegion_t:
      PointRange IndexRange_t [[2,2],[2,2],[1,1]]:
""")
  distribution_tree.add_distribution_info(dist_tree, sub_comm)
  assert len(PT.get_nodes_from_name(dist_tree, 'Index')) == 4+3

def test_clean_distribution_info():
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t [[27],[8],[0]]:
    Ngon Elements_t [22,0]:
      ElementConnectivity DataArray_t [1,2,3,4]:
      :CGNS#Distribution UserDefinedData_t:
    ZBC ZoneBC_t:
      bc1 BC_t "Farfield":
        PointList IndexArray_t [[1,2]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [0,2,4]:
        bcds BCDataSet_t:
          :CGNS#Distribution UserDefinedData_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [1,4,7,10]:
        PointListDonor IndexArray_t [13,16,7,10]:
        :CGNS#Distribution UserDefinedData_t:
    FS FlowSolution_t:
      field DataArray_t:
      :CGNS#Distribution UserDefinedData_t:
    :CGNS#Distribution UserDefinedData_t:
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)
  distribution_tree.clean_distribution_info(dist_tree)
  assert PT.get_node_from_name(dist_tree, ':CGNS#Distribution') is None
  assert len(PT.get_nodes_from_name(dist_tree, 'PointList')) == 2

