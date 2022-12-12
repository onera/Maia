import pytest
import maia.pytree as PT
from maia.pytree.yaml import parse_yaml_cgns
from maia.io.hdf import cgns_subsets

def test_create_pl_filter():
  yt = """
  match1 GridConnectivity_t "ZoneA":
    PointList IndexArray_t None:
    PointList#Size IndexArray_t [1,10]:
    PointListDonor IndexArray_t None:
    PointListDonor#Size IndexArray_t [3,10]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [5,10,10]:
  """
  hdf_filter = dict()
  node = parse_yaml_cgns.to_node(yt)

  distri = PT.maia.getDistribution(node, 'Index')[1]
  cgns_subsets._create_pl_filter(node, "path/to/node", "PointList", distri, hdf_filter)
  cgns_subsets._create_pl_filter(node, "path/to/node", "PointListDonor", distri, hdf_filter)

  assert hdf_filter['path/to/node/PointList'] == [[0, 0], [1, 1], [1, 5], [1, 1], \
      [0, 5], [1, 1], [1, 5], [1, 1],      [1, 10], [0]]
  assert hdf_filter['path/to/node/PointListDonor'] == [[0, 0], [1, 1], [3, 5], [1, 1], \
      [0, 5], [1, 1], [3, 5], [1, 1],      [3, 10], [0]]

  yt = """
  match1 GridConnectivity_t "ZoneA":
    PointRange IndexRange_t [[1,3],[1,3],[1,1]]:
    PointRangeDonor IndexRange_t [[1,3],[1,3],[3,3]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [2,4,6]:
  """
  hdf_filter = dict()
  node = parse_yaml_cgns.to_node(yt)
  distri = PT.maia.getDistribution(node, 'Index')[1]
  cgns_subsets._create_pl_filter(node, "path/to/node", "PointList", distri, hdf_filter)
  assert len(hdf_filter) == 0

def test_create_zone_bc_filter():
  #Don't test the value of value of dataspace, this is done by test_hdf_dataspace
  yt = """
Base CGNSBase_t [3,3]:
  Zone Zone_t [[27],[8],[0]]:
    ZBC ZoneBC_t:
      bc_only BC_t "wall":
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,10]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [5,10,10]:
      bc_with_ds BC_t "wall":
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,50]:
        BCDataSet BCDataSet_t:
          BCData BCData_t:
            array1 DataArray_t None:
            array2 DataArray_t None:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [20,50,50]:
      bc_with_subds BC_t "wall":
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,50]:
        BCDataSet BCDataSet_t:
          PointList IndexArray_t None:
          PointList#Size IndexArray_t [1,30]:
          BCData BCData_t:
            array1 DataArray_t None:
            array2 DataArray_t None:
          :CGNS#Distribution UserDefinedData_t:
            Index DataArray_t [20,30,30]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [20,50,50]:
"""
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  hdf_filter = dict()
  cgns_subsets.create_zone_bc_filter(PT.get_all_Zone_t(size_tree)[0], "Base/Zone", hdf_filter)
  assert len(hdf_filter.keys()) == 8
  prefix = 'Base/Zone/ZBC/'
  assert prefix+'bc_only/PointList' in hdf_filter
  assert prefix+'bc_with_ds/PointList' in hdf_filter
  assert prefix+'bc_with_ds/BCDataSet/BCData/array1' in hdf_filter
  assert prefix+'bc_with_ds/BCDataSet/BCData/array2' in hdf_filter
  assert prefix+'bc_with_subds/PointList' in hdf_filter
  assert prefix+'bc_with_subds/BCDataSet/PointList' in hdf_filter
  assert prefix+'bc_with_subds/BCDataSet/BCData/array2' in hdf_filter
  assert prefix+'bc_with_subds/BCDataSet/BCData/array1' in hdf_filter
  assert hdf_filter[prefix+'bc_with_subds/BCDataSet/PointList'] !=\
         hdf_filter[prefix+'bc_with_subds/PointList']

def test_create_zone_grid_connectivity_filter():
  #Don't test the value of value of dataspace, this is done by test_hdf_dataspace
  yt = """
Base CGNSBase_t [3,3]:
  ZoneU Zone_t [[27],[8],[0]]:
    ZGC ZoneGridConnectivity_t:
      match1 GridConnectivity_t "ZoneA":
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,10]:
        PointListDonor IndexArray_t None:
        PointListDonor#Size IndexArray_t [1,10]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [5,10,10]:
      match2 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,50]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [20,50,50]:
  ZoneS Zone_t None:
    ZGC ZoneGridConnectivity_t:
      match3 GridConnectivity1to1_t "ZoneA":
        PointRange IndexRange_t [[1,3],[1,3],[1,1]]:
        PointRangeDonor IndexRange_t [[1,3],[1,3],[3,3]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [5,9,9]:
"""
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  hdf_filter = dict()
  zoneU = PT.get_node_from_name(size_tree, 'ZoneU')
  zoneS = PT.get_node_from_name(size_tree, 'ZoneS')
  cgns_subsets.create_zone_grid_connectivity_filter(zoneU, "Base/ZoneU", hdf_filter)
  cgns_subsets.create_zone_grid_connectivity_filter(zoneS, "Base/ZoneS", hdf_filter)
  assert len(hdf_filter.keys()) == 3
  assert 'Base/ZoneU/ZGC/match1/PointList' in hdf_filter
  assert 'Base/ZoneU/ZGC/match1/PointListDonor' in hdf_filter
  assert 'Base/ZoneU/ZGC/match2/PointList' in hdf_filter
  assert 'Base/ZoneS/ZGC/match3/PointRange' not in hdf_filter #Point range are ignored

def test_create_flow_solution_filter():
  #Don't test the value of value of dataspace, this is done by test_hdf_dataspace
  yt = """
Base CGNSBase_t [3,3]:
  Zone Zone_t [[27],[8],[0]]:
    FSall FlowSolution_t:
      GridLocation GridLocation_t "Vertex":
      array1 DataArray_t None:
      array2 DataArray_t None:
    FSpartial FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t None:
      PointList#Size IndexArray_t [1,4]:
      array3 DataArray_t None:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t [0,4,4]:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t [12,27,27]:
      Cell DataArray_t [0,8,8]:
"""
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  hdf_filter = dict()
  cgns_subsets.create_flow_solution_filter(PT.get_all_Zone_t(size_tree)[0], "Base/Zone", hdf_filter)
  assert len(hdf_filter.keys()) == 4
  assert 'Base/Zone/FSall/array1' in hdf_filter
  assert 'Base/Zone/FSall/array2' in hdf_filter
  assert 'Base/Zone/FSpartial/array3' in hdf_filter
  assert 'Base/Zone/FSpartial/PointList' in hdf_filter

  yt = """
Base CGNSBase_t [3,3]:
  Zone Zone_t [[27],[8],[0]]:
    wrongFS FlowSolution_t:
      GridLocation GridLocation_t "FaceCenter":
      array1 DataArray_t None:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t [12,27,27]:
      Cell DataArray_t [0,8,8]:
"""
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  hdf_filter = dict()
  with pytest.raises(RuntimeError):
    cgns_subsets.create_flow_solution_filter(PT.get_all_Zone_t(size_tree)[0], "Base/Zone", hdf_filter)

def test_create_zone_subregion_filter():
  #Don't test the value of value of dataspace, this is done by test_hdf_dataspace
  yt = """
Base CGNSBase_t [3,3]:
  Zone Zone_t [[27],[8],[0]]:
    defined_ZSR ZoneSubRegion_t:
      PointList IndexArray_t None:
      PointList#Size IndexArray_t [1,10]:
      array1 DataArray_t None:
      array2 DataArray_t None:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t [5,10,10]:
    linked_ZSR ZoneSubRegion_t:
      BCRegionName Descriptor_t "bc":
      array1 DataArray_t None:
      array2 DataArray_t None:
    linked_ZSR2 ZoneSubRegion_t:
      GridConnectivityRegionName Descriptor_t "bc":
      array1 DataArray_t None:
      array2 DataArray_t None:
    ZoneBC ZoneBC_t:
      bc BC_t "farfield":
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,10]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [5,10,10]:
    ZoneGC ZoneGridConnectivity_t:
      bc GridConnectivity_t:
        PointList IndexArray_t None:
        PointList#Size IndexArray_t [1,40]:
        PointListDonor IndexArray_t None:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [20,40,40]:
"""
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  hdf_filter = dict()
  cgns_subsets.create_zone_subregion_filter(PT.get_all_Zone_t(size_tree)[0], "Base/Zone", hdf_filter)
  assert len(hdf_filter.keys()) == 7
  assert 'Base/Zone/defined_ZSR/PointList' in hdf_filter
  assert 'Base/Zone/defined_ZSR/array1' in hdf_filter
  assert 'Base/Zone/defined_ZSR/array2' in hdf_filter
  assert 'Base/Zone/linked_ZSR/array1' in hdf_filter
  assert 'Base/Zone/linked_ZSR/array2' in hdf_filter
  assert 'Base/Zone/linked_ZSR2/array2' in hdf_filter
  assert 'Base/Zone/linked_ZSR2/array2' in hdf_filter

  yt = """
Base CGNSBase_t [3,3]:
  Zone Zone_t [[27],[8],[0]]:
    unlinked_ZSR ZoneSubRegion_t:
      array1 DataArray_t None:
      array2 DataArray_t None:
"""
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  hdf_filter = dict()
  with pytest.raises(RuntimeError):
    cgns_subsets.create_zone_subregion_filter(PT.get_all_Zone_t(size_tree)[0], "Base/Zone", hdf_filter)
