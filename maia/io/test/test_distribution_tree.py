from pytest_mpi_check._decorator import mark_mpi_test
import mpi4py.MPI as MPI
import numpy      as np

import Converter.Internal as I
import maia.pytree.maia   as MT

from   maia.utils.yaml     import parse_yaml_cgns
from   maia.utils.parallel import utils as par_utils
from   maia.io             import distribution_tree

@mark_mpi_test(3)
def test_create_distribution_node(sub_comm):
  node = I.createNode('ParentNode', 'UserDefinedData_t')
  distribution_tree.create_distribution_node(100, sub_comm, 'MyDistribution', node)

  distri_ud   = MT.getDistribution(node)
  assert distri_ud is not None
  assert I.getType(distri_ud) == 'UserDefinedData_t'
  distri_node = I.getNodeFromName1(distri_ud, 'MyDistribution')
  assert distri_node is not None
  assert (I.getValue(distri_node) == par_utils.uniform_distribution(100, sub_comm)).all()
  assert I.getType(distri_node) == 'DataArray_t'

@mark_mpi_test(2)
def test_compute_plist_or_prange_distribution(sub_comm):
  node = I.newBC(name='BC', pointRange=[[1,3],[1,3],[3,3]])
  distribution_tree.compute_plist_or_prange_distribution(node, sub_comm)

  distrib_ud = MT.getDistribution(node)
  assert I.getType(distrib_ud) == 'UserDefinedData_t'
  distrib    = I.getNodeFromName1(distrib_ud, 'Index')
  assert I.getType(distrib) == 'DataArray_t'
  assert (I.getValue(distrib) == par_utils.uniform_distribution(3*3*1, sub_comm)).all()

  node = I.newBC(name='BC')
  I.newIndexArray('PointList', None, parent=node)
  I.newIndexArray('PointList#Size', [1,9], parent=node)
  distribution_tree.compute_plist_or_prange_distribution(node, sub_comm)

  distrib_ud = MT.getDistribution(node)
  assert I.getType(distrib_ud) == 'UserDefinedData_t'
  distrib    = I.getNodeFromName1(distrib_ud, 'Index')
  assert I.getType(distrib) == 'DataArray_t'
  assert (I.getValue(distrib) == par_utils.uniform_distribution(1*9, sub_comm)).all()

@mark_mpi_test(2)
def test_compute_elements_distribution(sub_comm):
  zoneS = I.newZone('ZoneS', ztype='Structured')
  zoneU = I.newZone('ZoneS', ztype='Unstructured')
  hexa = I.newElements('Hexa', 'HEXA', erange=[1,100],parent=zoneU)
  tri  = I.newElements('Tri', 'TRI', erange=[101,1000],parent=zoneU)
  distribution_tree.compute_elements_distribution(zoneS, sub_comm)
  distribution_tree.compute_elements_distribution(zoneU, sub_comm)
  assert (I.getNodeFromName(hexa, 'Element')[1] == \
      par_utils.uniform_distribution(100, sub_comm)).all()
  assert (I.getNodeFromName(tri , 'Element')[1] == \
      par_utils.uniform_distribution(900, sub_comm)).all()
  assert I.getNodeFromName(zoneS, 'Element') == None



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
    assert len(I.getNodesFromName(zone, 'Index')) == 5
    assert len(I.getNodesFromName(zone, 'Element')) == 1

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
    assert len(I.getNodesFromName(zone, 'Index')) == 3


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
  assert len(I.getNodesFromName(dist_tree, 'Index')) == 4+3

def test_clean_distribution_info():
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t [[27],[8],[0]]:
    Ngon Elements_t [22,0]:
      ElementConnectivity DataArray_t [1,2,3,4]:
      ElementConnectivity#Size DataArray_t [12]:
      :CGNS#Distribution UserDefinedData_t:
    ZBC ZoneBC_t:
      bc1 BC_t "Farfield":
        PointList IndexArray_t [[1,2]]:
        PointList#Size IndexArray_t [1,4]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [0,2,4]:
        bcds BCDataSet_t:
          PointList#Size IndexArray_t [1,2]:
          :CGNS#Distribution UserDefinedData_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [1,4,7,10]:
        PointListDonor IndexArray_t [13,16,7,10]:
        PointList#Size IndexArray_t [1,8]:
    FS FlowSolution_t:
      field DataArray_t:
      PointList#Size IndexArray_t [1,2]:
      :CGNS#Distribution UserDefinedData_t:
    :CGNS#Distribution UserDefinedData_t:
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)
  distribution_tree.clean_distribution_info(dist_tree)
  assert I.getNodeFromName(dist_tree, ':CGNS#Distribution') is None
  assert I.getNodeFromName(dist_tree, 'PointList#Size') is None
  assert len(I.getNodesFromName(dist_tree, 'PointList')) == 2
  assert I.getNodeFromName(dist_tree, 'ElementConnectivity#Size') is None

