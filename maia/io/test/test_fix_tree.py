import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT
from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import logging as mlog

from maia.io import fix_tree

class log_capture:
  def __init__(self):
    self.logs = ''
  def reset(self):
    self.logs = ''
  def log(self, msg):
    self.logs += msg

def test_check_datasize():
  yt = """
  Base CGNSBase_t [3,3]:
    ZoneA Zone_t I4 [[11,10,0]]:
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t:
  """
  log_collector = log_capture()
  mlog.add_printer_to_logger('maia-warnings', log_collector)

  tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.check_datasize(tree)
  assert log_collector.logs == ''
  grid_co = PT.get_node_from_name(tree, 'GridCoordinates')
  PT.new_DataArray('CoordinateY', np.arange(1000), parent=grid_co)
  fix_tree.check_datasize(tree)
  assert "Some heavy data are not distributed: ['CoordinateY']\n" in log_collector.logs

def test_fix_zone_datatype():
  yt = """
  Base CGNSBase_t [3,3]:
    ZoneA Zone_t I4 [[11,10,0]]:
    ZoneB Zone_t I4 [[11,10,0]]:
  """
  size_tree = PT.yaml.to_cgns_tree(yt)
  size_data = {'/CGNSLibraryVersion': (1, 'R4', (1,)),
               '/Base': (1, 'I4', (2,)),
               '/Base/ZoneA': (1, 'I4', (1, 3)),
               '/Base/ZoneB': (1, 'I8', (1, 3))}
  fix_tree.fix_zone_datatype(size_tree, size_data)
  assert PT.get_node_from_name(size_tree, "ZoneA")[1].dtype == np.int32
  assert PT.get_node_from_name(size_tree, "ZoneB")[1].dtype == np.int64

def test_force_periodic_dataarray_as_R4():
  yt = """
  Base CGNSBase_t [3,3]:
    ZoneA Zone_t I4 [[11,10,0]]:
      ZoneGridConnectivity ZoneGridConnectivity_t:
        GC1 GridConnectivity_t "Base/ZoneA":
          GridConnectivityProperty GridConnectivityProperty_t:
            Periodic Periodic_t:
              RotationAngle  DataArray_t R8 [0., 0. ,0.]:
              RotationCenter DataArray_t R4 [0., 0. ,0.]:
              Translation    DataArray_t R4 [1., 0. ,0.]:
        GC2 GridConnectivity1to1_t "Base/ZoneA":
          GridConnectivityProperty GridConnectivityProperty_t:
            Periodic Periodic_t:
              RotationAngle  DataArray_t R4 [0., 0. ,0.]:
              RotationCenter DataArray_t R8 [0., 0. ,0.]:
              Translation    DataArray_t R8 [-1., 0. ,0.]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.force_periodic_as_R4(tree)
  for data in PT.get_nodes_from_label(tree, "DataArray_t"):
    assert PT.get_value(data).dtype == np.float32

def test_fix_point_ranges():
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity1to1_t "ZoneB":
        PointRange IndexRange_t [[17,17],[3,9],[1,5]]:
        PointRangeDonor IndexRange_t [[7,1],[9,9],[1,5]]:
        Transform "int[IndexDimension]" [-2,-1,-3]:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity1to1_t "Base0/ZoneA":
        PointRange IndexRange_t [[7,1],[9,9],[1,5]]:
        PointRangeDonor IndexRange_t [[17,17],[3,9],[1,5]]:
        Transform "int[IndexDimension]" [-2,-1,-3]:
"""
  size_tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.fix_point_ranges(size_tree)
  gcA = PT.get_node_from_name(size_tree, 'matchAB')
  gcB = PT.get_node_from_name(size_tree, 'matchBA')
  assert (PT.get_child_from_name(gcA, 'PointRange')[1]      == [[17,17], [3,9], [1,5]]).all()
  assert (PT.get_child_from_name(gcA, 'PointRangeDonor')[1] == [[ 7, 1], [9,9], [5,1]]).all()
  assert (PT.get_child_from_name(gcB, 'PointRange')[1]      == [[ 7, 1], [9,9], [5,1]]).all()
  assert (PT.get_child_from_name(gcB, 'PointRangeDonor')[1] == [[17,17], [3,9], [1,5]]).all()

def test_fix_structured_point_range_shapes():
  yt = """
Base0 CGNSBase_t [2,2]:
  ZoneA Zone_t:
    ZoneType ZoneType_t "Structured":
    ZoneBC ZoneBC_t:
      BCA1 BC_t:
        PointRange IndexRange_t [[17,17],[3,9],[1,1]]:
      BCA2 BC_t:
        PointRange IndexRange_t [[1,1],[3,9]]:
Base1 CGNSBase_t [1,2]:
  ZoneB Zone_t:
    ZoneType ZoneType_t "Structured":
    ZoneBC ZoneBC_t:
      BCB BC_t:
        PointRange IndexRange_t [[1,3],[1,1],[1,1]]:
  ZoneC Zone_t:
    ZoneType ZoneType_t "Unstructured":
    ZoneBC ZoneBC_t:
      BCC BC_t:
        PointRange IndexRange_t [[1,17]]:
Base2 CGNSBase_t [3,3]:
  ZoneD Zone_t:
    ZoneType ZoneType_t "Structured":
    ZoneBC ZoneBC_t:
      BCD BC_t:
        PointRange IndexRange_t [[1,3],[1,1],[1,1]]:
"""
  size_tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.fix_structured_pr_shape(size_tree)
  bcA1 = PT.get_node_from_name(size_tree, 'BCA1')
  bcA2 = PT.get_node_from_name(size_tree, 'BCA2')
  bcB  = PT.get_node_from_name(size_tree, 'BCB' )
  bcC  = PT.get_node_from_name(size_tree, 'BCC' )
  bcD  = PT.get_node_from_name(size_tree, 'BCD' )
  assert (PT.get_child_from_name(bcA1, 'PointRange')[1] == [[17,17], [3,9]       ]).all()
  assert (PT.get_child_from_name(bcA2, 'PointRange')[1] == [[ 1, 1], [3,9]       ]).all()
  assert (PT.get_child_from_name(bcB,  'PointRange')[1] == [[ 1, 3]              ]).all()
  assert (PT.get_child_from_name(bcC,  'PointRange')[1] == [[ 1,17]              ]).all()
  assert (PT.get_child_from_name(bcD,  'PointRange')[1] == [[ 1, 3], [1,1], [1,1]]).all()

def test_fix_structured_pr_gridloc():
  yt = """
Base CGNSBase_t [3,3]:
  ZoneU Zone_t:
    ZoneType ZoneType_t "Unstructured":
    ZoneBC ZoneBC_t:
      BCU BC_t:
        PointRange IndexRange_t [[1,3],[1,3],[1,1]]:
        GridLocation GridLocation_t "FaceCenter":
  ZoneS Zone_t:
    ZoneType ZoneType_t "Structured":
    ZoneBC ZoneBC_t:
      BCS1 BC_t:
        PointRange IndexRange_t [[1,3],[1,3],[1,1]]:
        GridLocation GridLocation_t "FaceCenter":
        BCDS1 BCDataSet_t:
          GridLocation GridLocation_t "FaceCenter":
      BCS2 BC_t:
        PointRange IndexRange_t [[1,3],[1,3],[1,1]]:
        GridLocation GridLocation_t "Vertex":
        BCDS2 BCDataSet_t:
          PointRange IndexRange_t [[1,2],[1,2],[1,1]]:
          GridLocation GridLocation_t "FaceCenter":
      BCS3 BC_t:
        PointRange IndexRange_t [[1,3],[1,2],[1,1]]:
        GridLocation GridLocation_t "Vertex":
        BCDS3 BCDataSet_t:
          PointRange IndexRange_t [[1,2],[1,1],[1,1]]:
          GridLocation GridLocation_t "FaceCenter":
      BCS4 BC_t:
        PointRange IndexRange_t [[1,3],[1,3],[1,1]]:
        GridLocation GridLocation_t "FaceCenter":
        BCDS4 BCDataSet_t:
    ZGC ZoneGridConnectivity_t:
      GCS1 GridConnectivity_t:
        PointRange IndexRange_t [[1,3],[1,1],[1,3]]:
        GridLocation GridLocation_t "FaceCenter":
    ZSR1 ZoneSubRegion_t:
      PointRange IndexRange_t [[1,1],[1,3],[1,3]]:
      GridLocation GridLocation_t "FaceCenter":
    ZSR2 ZoneSubRegion_t:
      GridLocation GridLocation_t "FaceCenter":
      GridConnectivityRegionName Descriptor_t "GCS1":
    ZSR3 ZoneSubRegion_t:
      GridLocation GridLocation_t "FaceCenter":
      BCRegionName Descriptor_t "BCS1":
    FS1 FlowSolution_t:
      PointRange IndexRange_t [[1,1],[1,3],[1,3]]:
      GridLocation GridLocation_t "FaceCenter":
"""
  size_tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.fix_structured_pr_gridloc(size_tree)
  zone_u = PT.get_node_from_path(size_tree, 'Base/ZoneU')
  assert len(PT.get_all_subsets(zone_u, 'FaceCenter')) == 1
  zone_s = PT.get_node_from_path(size_tree, 'Base/ZoneS')
  assert len(PT.get_all_subsets(zone_s, 'FaceCenter')) == 0
  zsr1_n = PT.get_node_from_path(zone_s, 'ZSR1')
  assert PT.Container.GridLocation(zsr1_n, zone_s) == "IFaceCenter"
  zsr2_n = PT.get_node_from_path(zone_s, 'ZSR2')
  assert PT.Container.GridLocation(zsr2_n, zone_s) == "JFaceCenter"
  zsr3_n = PT.get_node_from_path(zone_s, 'ZSR3')
  assert PT.Container.GridLocation(zsr3_n, zone_s) == "KFaceCenter"

def test_ensure_symmetric_gc1to1():
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t:
    :CGNS#Distribution UserDefinedData_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity1to1_t "ZoneB":
        PointRange IndexRange_t [[17,17],[3,9],[1,5]]:
        PointRangeDonor IndexRange_t [[7,1],[9,9],[1,5]]:
  ZoneB Zone_t:
    :CGNS#Distribution UserDefinedData_t:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity1to1_t "Base0/ZoneA":
        PointRange IndexRange_t [[1,7],[9,9],[1,5]]:
        PointRangeDonor IndexRange_t [[17,17],[9,3],[1,5]]:
"""
  tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.ensure_symmetric_gc1to1(tree)
  gcA = PT.get_node_from_name(tree, 'matchAB')
  gcB = PT.get_node_from_name(tree, 'matchBA')
  assert (PT.get_child_from_name(gcA, 'PointRange')[1]      == [[17,17], [3,9], [1,5]]).all()
  assert (PT.get_child_from_name(gcA, 'PointRangeDonor')[1] == [[ 7, 1], [9,9], [1,5]]).all()
  assert (PT.get_child_from_name(gcB, 'PointRange')[1]      == PT.get_child_from_name(gcA, 'PointRangeDonor')[1]).all()
  assert (PT.get_child_from_name(gcB, 'PointRangeDonor')[1] == PT.get_child_from_name(gcA, 'PointRange')[1]).all()

def test_add_missing_pr_in_dataset():
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t:
    ZoneType ZoneType_t "Structured":
    ZBC ZoneBC_t:
      BCA1 BC_t:
        PointRange IndexRange_t [[1,1],[1,29],[1,85]]:
        BCDS BCDataSet_t:
          GridLocation GridLocation_t "FaceCenter":
      BCA2 BC_t:
        PointRange IndexRange_t [[1,1],[1,29],[1,85]]:
        BCDS BCDataSet_t:
          GridLocation GridLocation_t "IFaceCenter":
      BCA3 BC_t:
        PointRange IndexRange_t [[1,29],[1,1],[1,85]]:
        BCDS BCDataSet_t:
          GridLocation GridLocation_t "JFaceCenter":
      BCA4 BC_t:
        PointRange IndexRange_t [[1,29],[1,85],[1,1]]:
        BCDS BCDataSet_t:
          GridLocation GridLocation_t "KFaceCenter":
      BCB BC_t:
        PointRange IndexRange_t [[1,1],[1,3],[1,2]]:
        GridLocation GridLocation_t "FaceCenter":
        BCDS BCDataSet_t:
          GridLocation GridLocation_t "FaceCenter":
      BCC BC_t:
        PointRange IndexRange_t [[1,1],[1,3],[1,2]]:
        BCDS BCDataSet_t:
      BCD BC_t:
        PointList IndexArray_t [[1,1,1,1],[1,1,1,1],[1,2,3,4]]: #PL should be ignored
        BCDS BCDataSet_t:
  ZoneB Zone_t:
    ZoneType ZoneType_t "Unstructured":
    ZBC ZoneBC_t:
      BC BC_t:
        PointRange IndexRange_t [[1,1],[1,3],[1,2]]:
        BCDS BCDataSet_t:
          GridLocation GridLocation_t "FaceCenter":
"""
  size_tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.add_missing_pr_in_bcdataset(size_tree)
  bcA1 = PT.get_node_from_name(size_tree, 'BCA1')
  bcdsA1 = PT.get_child_from_label(bcA1, 'BCDataSet_t')
  assert (PT.get_child_from_name(bcdsA1, 'PointRange')[1] == [[1,1], [1,28], [1,84]]).all()
  bcA2 = PT.get_node_from_name(size_tree, 'BCA2')
  bcdsA2 = PT.get_child_from_label(bcA2, 'BCDataSet_t')
  assert (PT.get_child_from_name(bcdsA2, 'PointRange')[1] == [[1,1], [1,28], [1,84]]).all()
  bcA3 = PT.get_node_from_name(size_tree, 'BCA3')
  bcdsA3 = PT.get_child_from_label(bcA3, 'BCDataSet_t')
  assert (PT.get_child_from_name(bcdsA3, 'PointRange')[1] == [[1,28], [1,1], [1,84]]).all()
  bcA4 = PT.get_node_from_name(size_tree, 'BCA4')
  bcdsA4 = PT.get_child_from_label(bcA4, 'BCDataSet_t')
  assert (PT.get_child_from_name(bcdsA4, 'PointRange')[1] == [[1,28], [1,84], [1,1]]).all()
  bcB = PT.get_node_from_name(size_tree, 'BCB')
  bcdsB = PT.get_child_from_label(bcB, 'BCDataSet_t')
  assert (PT.get_child_from_name(bcdsB, 'PointRange') is None)
  bcC = PT.get_node_from_name(size_tree, 'BCB')
  bcdsC = PT.get_child_from_label(bcC, 'BCDataSet_t')
  assert (PT.get_child_from_name(bcdsC, 'PointRange') is None)
  bc = PT.get_node_from_name(size_tree, 'BCB')
  bcds = PT.get_child_from_label(bc, 'BCDataSet_t')
  assert (PT.get_child_from_name(bcds, 'PointRange') is None)

def test_enforce_pdm_dtype():
  wrong_pdm_type = np.int64 if pdm_dtype == np.int32 else np.int32
  wrong_type = 'I8' if pdm_dtype == np.int32 else 'I4'
  yt = f"""
  Base CGNSBase_t [3,3]:
    Zone Zone_t {wrong_type} [[11,10,0]]:
      NGon Elements_t [22,0]:
        ElementRange IndexRange_t [1, 3]:
        ElementConnectivity DataArray_t {wrong_type} [1,2,3,4]:
        ElementStartOffset DataArray_t {wrong_type} [0,1,2]:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "ZoneB":
          PointList IndexArray_t {wrong_type} [[11,12,13]]:
          PointListDonor IndexArray_t {wrong_type} [[1,2,3]]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  assert PT.get_node_from_name(tree, 'PointList')[1].dtype == wrong_pdm_type
  assert PT.get_node_from_name(tree, 'ElementConnectivity')[1].dtype == wrong_pdm_type
  assert PT.get_node_from_name(tree, 'ElementStartOffset')[1].dtype == wrong_pdm_type
  assert PT.get_node_from_name(tree, 'ElementRange')[1].dtype == np.int32
  fix_tree._enforce_pdm_dtype(tree)
  assert PT.get_node_from_name(tree, 'PointList')[1].dtype == pdm_dtype
  assert PT.get_node_from_name(tree, 'ElementConnectivity')[1].dtype == pdm_dtype
  assert PT.get_node_from_name(tree, 'ElementStartOffset')[1].dtype == pdm_dtype
  assert PT.get_node_from_name(tree, 'ElementRange')[1].dtype == pdm_dtype

def test_ensure_PE_global_indexing():

  def create_tree(elts):
    tree = PT.new_CGNSTree()
    base = PT.new_CGNSBase(parent=tree)
    zone = PT.new_Zone(type='Unstructured', parent=base)
    for elt in elts:
      PT.add_child(zone, elt)
    return tree

  ngon = PT.new_Elements('WrongNGon', 'NGON_n', erange=[1,4])
  pe   = PT.new_DataArray('ParentElements', [[1,2],[3,0],[1,0],[2,4]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(create_tree([ngon]))
  assert (pe[1] == [[5,6],[7,0],[5,0],[6,8]]).all()

  ngon = PT.new_Elements('GoodNGon', 'NGON_n', erange=[1,4])
  pe   = PT.new_DataArray('ParentElements', [[5,6],[7,0],[5,0],[6,8]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(create_tree([ngon]))
  assert (pe[1] == [[5,6],[7,0],[5,0],[6,8]]).all()

  nface = PT.new_Elements('FirstNace', 'NFACE_n', erange=[1,2])
  ngon = PT.new_Elements('SecondNGon', 'NGON_n', erange=[3,6])
  pe   = PT.new_DataArray('ParentElements', [[1,0],[1,0],[1,2],[2,0]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(create_tree([ngon]))
  assert (pe[1] == [[1,0],[1,0],[1,2],[2,0]]).all()

  ngon = PT.new_Elements('EmptyNGon', 'NGON_n', erange=[1,4])
  pe   = PT.new_DataArray('ParentElements', np.empty((0,2), order='F'), parent=ngon)
  fix_tree.ensure_PE_global_indexing(create_tree([ngon]))

  with pytest.raises(RuntimeError):
    ngon = PT.new_Elements('NGon', 'NGON_n')
    fix_tree.ensure_PE_global_indexing(create_tree([ngon, ngon]))
  with pytest.raises(RuntimeError):
    ngon = PT.new_NGonElements(erange=[1,4], pe=np.empty((4,2), order='F'))
    tri = PT.new_Elements('Tri', 'TRI_3', erange=[5, 10])
    fix_tree.ensure_PE_global_indexing(create_tree([ngon, tri]))

@pytest_parallel.mark.parallel(1)
def test_ensure_signed_nface_connectivity(comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  maia.algo.pe_to_nface(tree, comm)
  tree_bck = PT.deep_copy(tree)

  assert fix_tree.ensure_signed_nface_connectivity(tree, comm) == 0
  assert PT.is_same_tree(tree, tree_bck)

  # Force unsigned connectivity
  nface_ec = PT.get_node_from_path(tree, 'Base/zone/NFaceElements/ElementConnectivity')
  nface_ec[1] = np.abs(nface_ec[1])
  assert fix_tree.ensure_signed_nface_connectivity(tree, comm) == 1
  assert PT.is_same_tree(tree, tree_bck)

def test_rm_legacy_nodes():
  yt = f"""
  ZoneA Zone_t [[11,10,0]]:
    ZoneType ZoneType_t "Unstructured":
    :elsA#Hybrid UserDefinedData_t:
      SortedCrossTable DataArray_t:
      IndexNGONCrossTable DataArray_t:
    .cedre#Geometry UserDefinedData_t:
  ZoneB Zone_t [[11,10,0]]:
    FlowSol FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      GoodArray DataArray_t:
      GoodArray#Size DataArray_t [10]:
      WrongArray DataArray_t:
  ZoneC Zone_t [[11,10,0]]:
    :elsA#Hybrid UserDefinedData_t:
    .cedre#Geometry UserDefinedData_t:
    ZoneGridConnectivity ZoneGridConnectivity_t:
      GridConnectivity GridConnectivity_t:
        GridConnectivityType GridConnectivityType_t "Abutting":
        PointRange IndexRange_t [[1,1], [1,10], [1,3]]:
        PointListDonor IndexArray_t:
        PointListDonor#Size DataArray_t [3, 1]:
        UserDefinedData UserDefinedData_t:
          NMRatio DataArray_t [1., 2., 1.]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.rm_legacy_nodes(tree)
  assert PT.get_node_from_name(tree, ':elsA#Hybrid') is None
  assert PT.get_node_from_name(tree, '.cedre#Geometry') is None
  
  assert PT.get_node_from_name(tree, 'GoodArray') is not None
  assert PT.get_node_from_name(tree, 'WrongArray') is None

  assert PT.get_node_from_name(tree, 'PointListDonor*') is None

def test_corr_index_range_names():
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t:
    ZBC ZoneBC_t:
      BCA BC_t:
        ElementRange IndexRange_t:
      BCB BC_t:
        WrongName IndexRange_t:
"""
  size_tree = PT.yaml.to_cgns_tree(yt)
  fix_tree.corr_index_range_names(size_tree)
  bcA = PT.get_node_from_name(size_tree, 'BCA')
  bcB = PT.get_node_from_name(size_tree, 'BCB')
  irA = PT.get_node_from_label(bcA, 'IndexRange_t')
  irB = PT.get_node_from_label(bcB, 'IndexRange_t')
  assert PT.get_name(irA) == 'PointRange'
  assert PT.get_name(irB) == 'WrongName'

def test_check_namings():
  tree = PT.yaml.to_cgns_tree("""
  ZoneA Zone_t:
    ZoneGridConnectivity_t ZoneGridConnectivity_t:
      match1 GridConnectivity_t:
  """)
  base = PT.get_node_from_label(tree, 'CGNSBase_t')
  gc   = PT.get_node_from_label(tree, 'GridConnectivity_t')

  log_collector = log_capture()
  mlog.add_printer_to_logger('maia-warnings', log_collector)

  fix_tree.check_namings(tree)
  assert len(log_collector.logs) == 0

  PT.new_Zone('Zone.P6.N0', type='Unstructured', parent=base)
  fix_tree.check_namings(tree)
  assert len(log_collector.logs) > 0

  log_collector.reset()
  PT.rm_nodes_from_name(tree, 'Zone.P6.N0')
  PT.set_name(gc, 'JN.P5.N2.LT.P12.N5')
  fix_tree.check_namings(tree)
  assert len(log_collector.logs) > 0