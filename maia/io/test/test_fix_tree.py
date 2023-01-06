import pytest
import numpy as np

import maia.pytree as PT
from maia.pytree.yaml  import parse_yaml_cgns
from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.io import fix_tree

def test_fix_zone_datatype():
  yt = """
  Base CGNSBase_t [3,3]:
    ZoneA Zone_t I4 [[11,10,0]]:
    ZoneB Zone_t I4 [[11,10,0]]:
  """
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  size_data = {'/CGNSLibraryVersion': (1, 'R4', (1,)),
               '/Base': (1, 'I4', (2,)),
               '/Base/ZoneA': (1, 'I4', (1, 3)),
               '/Base/ZoneB': (1, 'I8', (1, 3))}
  fix_tree.fix_zone_datatype(size_tree, size_data)
  assert PT.get_node_from_name(size_tree, "ZoneA")[1].dtype == np.int32
  assert PT.get_node_from_name(size_tree, "ZoneB")[1].dtype == np.int64

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
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  fix_tree.fix_point_ranges(size_tree)
  gcA = PT.get_node_from_name(size_tree, 'matchAB')
  gcB = PT.get_node_from_name(size_tree, 'matchBA')
  assert (PT.get_child_from_name(gcA, 'PointRange')[1]      == [[17,17], [3,9], [1,5]]).all()
  assert (PT.get_child_from_name(gcA, 'PointRangeDonor')[1] == [[ 7, 1], [9,9], [5,1]]).all()
  assert (PT.get_child_from_name(gcB, 'PointRange')[1]      == [[ 7, 1], [9,9], [5,1]]).all()
  assert (PT.get_child_from_name(gcB, 'PointRangeDonor')[1] == [[17,17], [3,9], [1,5]]).all()

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
  ZoneB Zone_t:
    ZoneType ZoneType_t "Unstructured":
    ZBC ZoneBC_t:
      BC BC_t:
        PointRange IndexRange_t [[1,1],[1,3],[1,2]]:
        BCDS BCDataSet_t:
          GridLocation GridLocation_t "FaceCenter":
"""
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
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

#def test_load_grid_connectivity_property():
  #Besoin de charger depuis un fichier, comment tester ?

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
  tree = parse_yaml_cgns.to_cgns_tree(yt)
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
  ngon = PT.new_Elements('WrongNGon', 'NGON_n', erange=[1,4])
  pe   = PT.new_DataArray('ParentElements', [[1,2],[3,0],[1,0],[2,4]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(PT.new_node('Zone', 'Zone_t', children=[ngon]))
  assert (pe[1] == [[5,6],[7,0],[5,0],[6,8]]).all()

  ngon = PT.new_Elements('GoodNGon', 'NGON_n', erange=[1,4])
  pe   = PT.new_DataArray('ParentElements', [[5,6],[7,0],[5,0],[6,8]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(PT.new_node('Zone', 'Zone_t', children=[ngon]))
  assert (pe[1] == [[5,6],[7,0],[5,0],[6,8]]).all()

  nface = PT.new_Elements('FirstNace', 'NFACE_n', erange=[1,2])
  ngon = PT.new_Elements('SecondNGon', 'NGON_n', erange=[3,6])
  pe   = PT.new_DataArray('ParentElements', [[1,0],[1,0],[1,2],[2,0]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(PT.new_node('Zone', 'Zone_t', children=[ngon]))
  assert (pe[1] == [[1,0],[1,0],[1,2],[2,0]]).all()

  ngon = PT.new_Elements('EmptyNGon', 'NGON_n', erange=[1,4])
  pe   = PT.new_DataArray('ParentElements', np.empty((0,2), order='F'), parent=ngon)
  fix_tree.ensure_PE_global_indexing(PT.new_node('Zone', 'Zone_t', children=[ngon]))

  with pytest.raises(RuntimeError):
    ngon = PT.new_Elements('NGon', 'NGON_n')
    fix_tree.ensure_PE_global_indexing(PT.new_node('Zone', 'Zone_t', children=[ngon,ngon]))
  with pytest.raises(RuntimeError):
    ngon = PT.new_Elements('NGon', 'NGON_n', erange=[1,4])
    tri = PT.new_Elements('Tri', 'TRI_3')
    fix_tree.ensure_PE_global_indexing(PT.new_node('Zone', 'Zone_t', children=[ngon,tri]))

def test_rm_legacy_nodes():
  yt = f"""
  ZoneA Zone_t [[11,10,0]]:
    ZoneType ZoneType_t "Unstructured":
    :elsA#Hybrid UserDefinedData_t:
      SortedCrossTable DataArray_t:
      IndexNGONCrossTable DataArray_t:
  ZoneB Zone_t [[11,10,0]]:
  ZoneC Zone_t [[11,10,0]]:
    :elsA#Hybrid UserDefinedData_t:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  fix_tree.rm_legacy_nodes(tree)
  assert PT.get_node_from_name(tree, ':elsA#Hybrid') is None

