import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import mpi4py.MPI as MPI
import numpy      as np
import Converter.Internal as I

from   maia.utils        import parse_yaml_cgns
from   maia.distribution import distribution_tree
import maia.distribution.distribution_function as MID

@mark_mpi_test(2)
def test_compute_plist_or_prange_distribution(sub_comm):
  node = I.newBC(name='BC', pointRange=[[1,3],[1,3],[3,3]])
  distribution_tree.compute_plist_or_prange_distribution(node, sub_comm)

  distrib_ud = I.getNodeFromName1(node, ':CGNS#Distribution')
  assert I.getType(distrib_ud) == 'UserDefinedData_t'
  distrib    = I.getNodeFromName1(distrib_ud, 'Index')
  assert I.getType(distrib) == 'DataArray_t'
  assert (I.getValue(distrib) == MID.uniform_distribution(3*3*1, sub_comm)).all()

  node = I.newBC(name='BC')
  I.newIndexArray('PointList', None, parent=node)
  I.newIndexArray('PointList#Size', [1,9], parent=node)
  distribution_tree.compute_plist_or_prange_distribution(node, sub_comm)

  distrib_ud = I.getNodeFromName1(node, ':CGNS#Distribution')
  assert I.getType(distrib_ud) == 'UserDefinedData_t'
  distrib    = I.getNodeFromName1(distrib_ud, 'Index')
  assert I.getType(distrib) == 'DataArray_t'
  assert (I.getValue(distrib) == MID.uniform_distribution(1*9, sub_comm)).all()

@mark_mpi_test(2)
def test_compute_elements_distribution(sub_comm):
  zoneS = I.newZone('ZoneS', ztype='Structured')
  zoneU = I.newZone('ZoneS', ztype='Unstructured')
  hexa = I.newElements('Hexa', 'HEXA', erange=[1,100],parent=zoneU)
  tri  = I.newElements('Tri', 'TRI', erange=[101,1000],parent=zoneU)
  distribution_tree.compute_elements_distribution(zoneS, sub_comm)
  distribution_tree.compute_elements_distribution(zoneU, sub_comm)
  assert (I.getNodeFromName(hexa, 'Element')[1] == \
      MID.uniform_distribution(100, sub_comm)).all()
  assert (I.getNodeFromName(tri , 'Element')[1] == \
      MID.uniform_distribution(900, sub_comm)).all()
  assert I.getNodeFromName(zoneS, 'Element') == None



#@pytest.mark.mpi(min_size=2)
#@pytest.mark.parametrize("sub_comm", [2], indirect=['sub_comm'])
#@mark_mpi_test(2)
#@pytest.mark.parametrize("sub_comm", [2], indirect=['sub_comm'])

@mark_mpi_test(2)
class Test_compute_zone_distribution:
  def test_unstruct(self, sub_comm):
    if(sub_comm == MPI.COMM_NULL):
      return
    # assert False
    yt = """
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
  """
    tree = parse_yaml_cgns.to_complete_pytree(yt)
    #Create zone by hand, otherwith ZoneType is misformed
    zone = I.newZone('Zone', np.array([[27,8,0]]), 'Unstructured')
    for child in tree[2]:
      I.addChild(zone, child)
    distribution_tree.compute_zone_distribution(zone, sub_comm)
    assert len(I.getNodesFromName(zone, 'Index')) == 4
    assert len(I.getNodesFromName(zone, 'Element')) == 1

  def test_struct(self, sub_comm):
    if(sub_comm == MPI.COMM_NULL):
      return
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
    tree = parse_yaml_cgns.to_complete_pytree(yt)
    #Create zone by hand, otherwith ZoneType is misformed
    zone = I.getZones(tree)[0]
    distribution_tree.compute_zone_distribution(zone, sub_comm)
    assert len(I.getNodesFromName(zone, 'Index')) == 3


@mark_mpi_test(2)
def test_add_distribution_info(sub_comm):
  dist_tree = parse_yaml_cgns.to_complete_pytree("""
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
    :CGNS#Distribution UserDefinedData_t:
"""
  dist_tree = parse_yaml_cgns.to_complete_pytree(yt)
  distribution_tree.clean_distribution_info(dist_tree)
  assert I.getNodeFromName(dist_tree, ':CGNS#Distribution') is None
  assert I.getNodeFromName(dist_tree, 'PointList#Size') is None
  assert len(I.getNodesFromName(dist_tree, 'PointList')) == 2
  assert I.getNodeFromName(dist_tree, 'ElementConnectivity#Size') is None

