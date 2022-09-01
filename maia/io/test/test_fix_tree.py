import pytest
import numpy as np
import Converter.Internal as I
import maia.pytree as PT
from maia.utils.yaml  import parse_yaml_cgns
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
  ngon = I.newElements('WrongNGon', 'NGON', erange=[1,4])
  pe   = I.newDataArray('ParentElements', [[1,2],[3,0],[1,0],[2,4]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(I.createNode('Zone', 'Zone_t', children=[ngon]))
  assert (pe[1] == [[5,6],[7,0],[5,0],[6,8]]).all()

  ngon = I.newElements('GoodNGon', 'NGON', erange=[1,4])
  pe   = I.newDataArray('ParentElements', [[5,6],[7,0],[5,0],[6,8]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(I.createNode('Zone', 'Zone_t', children=[ngon]))
  assert (pe[1] == [[5,6],[7,0],[5,0],[6,8]]).all()

  nface = I.newElements('FirstNace', 'NFACE', erange=[1,2])
  ngon = I.newElements('SecondNGon', 'NGON', erange=[3,6])
  pe   = I.newDataArray('ParentElements', [[1,0],[1,0],[1,2],[2,0]], parent=ngon)
  fix_tree.ensure_PE_global_indexing(I.createNode('Zone', 'Zone_t', children=[ngon]))
  assert (pe[1] == [[1,0],[1,0],[1,2],[2,0]]).all()

  ngon = I.newElements('EmptyNGon', 'NGON', erange=[1,4])
  pe   = I.newDataArray('ParentElements', np.empty((0,2), order='F'), parent=ngon)
  fix_tree.ensure_PE_global_indexing(I.createNode('Zone', 'Zone_t', children=[ngon]))

  with pytest.raises(RuntimeError):
    ngon = I.newElements('NGon', 'NGON')
    fix_tree.ensure_PE_global_indexing(I.createNode('Zone', 'Zone_t', children=[ngon,ngon]))
  with pytest.raises(RuntimeError):
    ngon = I.newElements('NGon', 'NGON', erange=[1,4])
    tri = I.newElements('Tri', 'TRI')
    fix_tree.ensure_PE_global_indexing(I.createNode('Zone', 'Zone_t', children=[ngon,tri]))

