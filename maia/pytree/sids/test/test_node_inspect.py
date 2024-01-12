import pytest
import numpy              as np

from maia.pytree      import node as N
from maia.pytree      import walk as W

from maia.pytree.sids import node_inspect as SIDS

from maia.pytree.yaml   import parse_yaml_cgns

def test_find_connected_zones():
  yt = """
  BaseA CGNSBase_t:
    Zone1 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone2 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone4":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "BaseA/Zone1":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
        match2 GridConnectivity_t "BaseB/Zone6":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone4 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone2":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  BaseB CGNSBase_t:
    Zone5 Zone_t:
    Zone6 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "BaseA/Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  connected_path = SIDS.Tree.find_connected_zones(tree)
  assert len(connected_path) == 3
  for zones in connected_path:
    if len(zones) == 1:
      assert zones == ['BaseB/Zone5']
    if len(zones) == 2:
      assert sorted(zones) == ['BaseA/Zone2', 'BaseA/Zone4']
    if len(zones) == 3:
      assert sorted(zones) == ['BaseA/Zone1', 'BaseA/Zone3', 'BaseB/Zone6']

def test_find_periodic_jns():
  yt = """
  BaseA CGNSBase_t:
    Zone1 Zone_t:
      ZGC ZoneGridConnectivity_t:
    Zone2 Zone_t:
      ZGC ZoneGridConnectivity_t:
  BaseB CGNSBase_t:
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
    Zone4 Zone_t:
      ZGC ZoneGridConnectivity_t:
  """
  import maia.pytree as PT
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  zgc_s = PT.get_nodes_from_label(tree, "ZoneGridConnectivity_t")

  rot1 = {'rotation_angle' : [10., 0., 0.]}
  rot2 = {'rotation_angle' : [10., 0., 0.], 'rotation_center' : [1., 1., 1.]}

  PT.new_GridConnectivityProperty(rot1, parent=PT.new_GridConnectivity('matchA', type='Abutting1to1', parent=zgc_s[0]))
  PT.new_GridConnectivityProperty(rot2, parent=PT.new_GridConnectivity('matchB', type='Abutting1to1', parent=zgc_s[0]))
  PT.new_GridConnectivityProperty(rot2, parent=PT.new_GridConnectivity('matchB', type='Abutting1to1', parent=zgc_s[1]))
  PT.new_GridConnectivityProperty(rot2, parent=PT.new_GridConnectivity('matchB', type='Abutting1to1', parent=zgc_s[2]))
  PT.new_GridConnectivityProperty(rot2, parent=PT.new_GridConnectivity('matchB', type='Abutting1to1', parent=zgc_s[3]))
  PT.new_GridConnectivityProperty(rot1, parent=PT.new_GridConnectivity('matchA', type='Abutting1to1', parent=zgc_s[1]))
  PT.new_GridConnectivity('matchnoperio1', type='Abutting1to1', parent=zgc_s[0])
  PT.new_GridConnectivity('matchnoperio2', type='Abutting1to1', parent=zgc_s[2])

  perio_vals, perio_paths = SIDS.Tree.find_periodic_jns(tree)
  assert len(perio_vals) == len(perio_paths) == 2

  expected_0 = (np.array([0., 0., 0.]), np.array([10., 0., 0.]), np.zeros(3))
  expected_1 = (np.array([1., 1., 1.]), np.array([10., 0., 0.]), np.zeros(3))

  assert all([(a==b).all() for a,b in zip(perio_vals[0], expected_0)])
  assert all([(a==b).all() for a,b in zip(perio_vals[1], expected_1)])
  
  assert perio_paths[0] == ['BaseA/Zone1/ZGC/matchA', 'BaseA/Zone2/ZGC/matchA']
  assert perio_paths[1] == ['BaseA/Zone1/ZGC/matchB', 'BaseA/Zone2/ZGC/matchB', 'BaseB/Zone3/ZGC/matchB', 'BaseB/Zone4/ZGC/matchB']


def test_ZoneType():
  #With numpy arrays
  zone_u = N.new_Zone('ZoneU', type='Unstructured')
  zone_s = N.new_Zone('ZoneS', type='Structured')
  assert SIDS.Zone.Type(zone_u) == 'Unstructured'
  assert SIDS.Zone.Type(zone_s) == 'Structured'
  #With strings
  W.get_child_from_label(zone_u, 'ZoneType_t')[1] = 'Unstructured'
  W.get_child_from_label(zone_s, 'ZoneType_t')[1] = 'Structured'
  assert SIDS.Zone.Type(zone_u) == 'Unstructured'
  assert SIDS.Zone.Type(zone_s) == 'Structured'

def test_IndexDimension():
  assert SIDS.Zone.IndexDimension(N.new_Zone('ZoneU', type='Unstructured', size=[[11,10,0]])) == 1
  assert SIDS.Zone.IndexDimension(N.new_Zone('ZoneS', type='Structured', size=[[10,9,0],[5,4,0],[2,1,0]])) == 3
  assert SIDS.Zone.IndexDimension(N.new_Zone('ZoneS', type='Structured', size=[[10,9,0],[5,4,0]])) == 2
  assert SIDS.Zone.IndexDimension(N.new_Zone('ZoneS', type='Structured', size=[[10,9,0]])) == 1

def test_NGonNode():
  zone = N.new_Zone('Zone', size=[[100, 36, 0]], type='Unstructured')
  with pytest.raises(RuntimeError):
    SIDS.Zone.NGonNode(zone)
  N.new_Elements('NGon',  type='NGON_n',  parent=zone)
  N.new_Elements('NFace', type='NFACE_n', parent=zone)
  ngon = SIDS.Zone.NGonNode(zone)
  assert N.get_name(ngon) == 'NGon' and N.get_value(ngon)[0] == 22
  N.new_Elements('NGon2', type='NGON_n', parent=zone)
  with pytest.raises(RuntimeError):
    SIDS.Zone.NGonNode(zone)

def test_ElementSize():
  elt1 = N.new_Elements(type='QUAD_4', erange=[1,100])
  elt2 = N.new_Elements(type='QUAD_4', erange=[15,15])
  assert SIDS.Element.Size(elt1) == 100
  assert SIDS.Element.Size(elt2) == 1

def test_ElementType():
  elt1 = N.new_Elements(type='QUAD_4', erange=[1,100])
  elt_type = SIDS.Element.Type(elt1)
  assert  isinstance(elt_type, int) and elt_type == 7

def test_ElementCGNSName():
  assert SIDS.Element.CGNSName(N.new_node("Toto", "Elements_t", [22, 0])) == "NGON_n"
  assert SIDS.Element.CGNSName(N.new_node("Toto", "Elements_t", [42, 0])) == "TRI_15"

def test_ElementDimension():
  assert SIDS.Element.Dimension(N.new_node("Toto", "Elements_t", [22, 0])) == 2
  assert SIDS.Element.Dimension(N.new_node("Toto", "Elements_t", [42, 0])) == 2
  assert SIDS.Element.Dimension(N.new_node("Toto", "Elements_t", [34, 0])) == 3

def test_ElementNVtx():
  assert SIDS.Element.NVtx(N.new_node("Toto", "Elements_t", [22, 0])) == None
  assert SIDS.Element.NVtx(N.new_node("Toto", "Elements_t", [42, 0])) == 15

def test_GridLocation():
  bc_no_loc = N.new_BC()
  bc_loc    = N.new_BC()
  N.new_GridLocation('JFaceCenter', bc_loc)
  assert SIDS.Subset.GridLocation(bc_no_loc) == 'Vertex'
  assert SIDS.Subset.GridLocation(bc_loc   ) == 'JFaceCenter'

def test_GridConnectivity_Type():
  gc = N.new_node("gc", "GridConnectivity1to1_t")
  assert SIDS.GridConnectivity.Type(gc) == "Abutting1to1"
  gc = N.new_node("gc", "GridConnectivity_t", 
      children=[N.new_node('GridConnectivityType', 'GridConnectivityType_t', "Abutting")])
  assert SIDS.GridConnectivity.Type(gc) == "Abutting"
  bc = N.new_BC("bc")
  with pytest.raises(Exception):
    SIDS.GridConnectivity.Type(bc)

def test_GridConnectivity_isperiodic():
  gc = N.new_GridConnectivity()
  assert SIDS.GridConnectivity.isperiodic(gc) == False
  N.new_GridConnectivityProperty(periodic={'translation' : [0., 0., 1.]}, parent=gc)
  assert SIDS.GridConnectivity.isperiodic(gc) == True

def test_GridConnectivity_periodic_values():
  gc = N.new_GridConnectivity()
  assert SIDS.GridConnectivity.periodic_values(gc) == (None, None, None)
  N.new_GridConnectivityProperty(parent=gc)
  assert SIDS.GridConnectivity.periodic_values(gc) == (None, None, None)
  W.rm_children_from_label(gc, 'GridConnectivityProperty_t')
  N.new_GridConnectivityProperty(periodic={'translation' : [0., 0., 1.]},parent=gc)
  assert (SIDS.GridConnectivity.periodic_values(gc).RotationCenter == [0., 0., 0.]).all()
  assert (SIDS.GridConnectivity.periodic_values(gc).RotationAngle == [0., 0., 0.]).all()
  assert (SIDS.GridConnectivity.periodic_values(gc).Translation == [0., 0., 1.]).all()

def test_zone_u_size():
  #Simulate a 10*5*2 vtx zone
  zone_u = N.new_Zone('Zone', size=[[100, 36, 0]], type='Unstructured')

  assert SIDS.Zone.VertexSize(zone_u) == 10*5*2
  assert SIDS.Zone.CellSize(zone_u) == 9*4*1
  assert SIDS.Zone.VertexBoundarySize(zone_u) == 0

  assert SIDS.Zone.n_vtx(zone_u) == 10*5*2
  assert SIDS.Zone.n_cell(zone_u) == 9*4*1
  assert SIDS.Zone.n_vtx_bnd(zone_u) == 0

def test_zone_s_size():
  #Simulate a 10*5*2 vtx zone
  zone_s = N.new_Zone('Zone', size=[[10,9,0], [5,4,0], [2,1,0]], type='Structured')

  assert np.all(SIDS.Zone.VertexSize(zone_s) == [10,5,2])
  assert np.all(SIDS.Zone.CellSize(zone_s) == [9,4,1])
  assert np.all(SIDS.Zone.VertexBoundarySize(zone_s) == [0,0,0])

  assert SIDS.Zone.n_vtx(zone_s) == 10*5*2
  assert SIDS.Zone.n_cell(zone_s) == 9*4*1
  assert SIDS.Zone.n_vtx_bnd(zone_s) == 0

def test_get_ordered_elements():
  zone = N.new_Zone()
  N.new_Elements('ElemA', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', erange=[1, 10], parent=zone)
  N.new_Elements('ElemC', erange=[54,60], parent=zone)

  sorted_elems = SIDS.Zone.get_ordered_elements(zone)
  assert [N.get_name(elem) for elem in sorted_elems] == ['ElemB', 'ElemA', 'ElemC']

def test_has_ngon_elements():
  zone = N.new_Zone()
  N.new_Elements('ElemA', erange=[11, 53], parent=zone)
  assert not SIDS.Zone.has_ngon_elements(zone)
  N.new_Elements('ElemB', type='NGON_n', erange=[1, 11], parent=zone)
  assert SIDS.Zone.has_ngon_elements(zone)

def test_has_ngon_elements():
  zone = N.new_Zone()
  N.new_Elements('ElemA', type='NGON_n', erange=[1, 11], parent=zone)
  assert not SIDS.Zone.has_nface_elements(zone)
  N.new_Elements('ElemB', type='NFACE_n', erange=[12, 23], parent=zone)
  assert SIDS.Zone.has_nface_elements(zone)
    
def test_get_ordered_elements_per_dim():
  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)

  sorted_elems_per_dim = SIDS.Zone.get_ordered_elements_per_dim(zone)
  assert sorted_elems_per_dim[0] == []
  assert sorted_elems_per_dim[1] == []
  assert [N.get_name(elem) for elem in sorted_elems_per_dim[2]] == ['ElemC']
  assert [N.get_name(elem) for elem in sorted_elems_per_dim[3]] == ['ElemB', 'ElemA']
    
def test_get_elt_range_per_dim():
  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.get_elt_range_per_dim(zone) == [[0,0], [0,0], [54,60], [1,53]]

  zone = N.new_Zone()
  N.new_Elements('ElemA', type='QUAD_4', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.get_elt_range_per_dim(zone) == [[0,0], [0,0], [11,60], [1,10]]

  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='TRI_3',  erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  with pytest.raises(RuntimeError):
    SIDS.Zone.get_elt_range_per_dim(zone)

def test_elt_ordering_by_dim():
  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == -1

  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == 0

  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[18, 60], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[8, 17],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[1,7],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == 1

def test_zone_dim():
  zone = N.new_Zone()
  with pytest.raises(ValueError):
    SIDS.Zone.Dimension(zone)
  N.new_Elements('Elem0', type='NODE' ,  erange=[ 1,10],  parent=zone)
  assert SIDS.Zone.Dimension(zone) == 0
  N.new_Elements('Elem1', type='BAR_2',  erange=[31,40],  parent=zone)
  assert SIDS.Zone.Dimension(zone) == 1
  N.new_Elements('Elem2', type='TRI_3',  erange=[11,20],  parent=zone)
  assert SIDS.Zone.Dimension(zone) == 2
  N.new_Elements('Elem3', type='HEXA_8', erange=[21,30], parent=zone)
  assert SIDS.Zone.Dimension(zone) == 3

def test_PointRange():
  pr = N.new_IndexRange('StandardPR', [1,3, 3,5, 1,3])
  assert (SIDS.PointRange.SizePerIndex(pr) == [3,3,3]).all()
  assert (SIDS.PointRange.n_elem(pr) == 3*3*3)

  pr = N.new_IndexRange('GCLikePR', [7,1, 9,9, 5,1])
  assert (SIDS.PointRange.SizePerIndex(pr) == [7,1,5]).all()
  assert (SIDS.PointRange.n_elem(pr) == 7*1*5)

  pr = N.new_IndexRange('ULike', [[1,15]]) # PR must be 2d
  assert (SIDS.PointRange.SizePerIndex(pr) == [15]).all()
  assert (SIDS.PointRange.n_elem(pr) == 15)

def test_PointList():
  pl = N.new_IndexArray('StandartPL', [[1,6,12]])
  assert SIDS.PointList.n_elem(pl) == 3

  pl = N.new_IndexArray('SLike', [[1,1,1,1,1], [1,1,1,2,2], [1,3,5,7,9]])
  assert SIDS.PointList.n_elem(pl) == 5

def test_Subset():
  sol = N.new_FlowSolution(loc='Vertex')
  pl = N.new_IndexArray('PointList', [[1,6,12]], parent=sol)

  assert SIDS.Subset.GridLocation(sol) == 'Vertex'
  assert SIDS.Subset.getPatch(sol) is pl
  assert SIDS.Subset.n_elem(sol) == SIDS.PointList.n_elem(pl)

  with pytest.raises(AssertionError):
    pr = N.new_IndexRange('PointRange', [[1,15]], parent=sol)
    patch = SIDS.Subset.getPatch(sol)


def test_getZoneDonorPath():
  jn1 = N.new_node('match', 'GridConnectivity1to1_t', value='BaseXX/ZoneYY')
  jn2 = N.new_node('match', 'GridConnectivity1to1_t', value='ZoneYY')
  assert SIDS.GridConnectivity.ZoneDonorPath(jn1, 'BaseXX') == 'BaseXX/ZoneYY'
  assert SIDS.GridConnectivity.ZoneDonorPath(jn2, 'BaseXX') == 'BaseXX/ZoneYY'


def test_getSubregionExtent():
  yt = """
Zone Zone_t:
  ZoneBC ZoneBC_t:
    BC BC_t:
    BC2 BC_t:
  ZGC ZoneGridConnectivity_t:
    GCA GridConnectivity_t:
    GCB GridConnectivity_t:
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
  import maia.pytree as PT
  zone = parse_yaml_cgns.to_node(yt)

  assert SIDS.Subset.ZSRExtent(W.get_node_from_name(zone, 'UnLinkedZSR'), zone) == 'UnLinkedZSR'
  assert SIDS.Subset.ZSRExtent(W.get_node_from_name(zone, 'BCLinkedZSR'), zone) == 'ZoneBC/BC2'
  assert SIDS.Subset.ZSRExtent(W.get_node_from_name(zone, 'GCLinkedZSR'), zone) == 'ZGC/GC1to1B'

  with pytest.raises(ValueError):
    SIDS.Subset.ZSRExtent(W.get_node_from_name(zone, 'OrphelanZSR'), zone)
  with pytest.raises(PT.CGNSLabelNotEqualError):
    SIDS.Subset.ZSRExtent(W.get_node_from_name(zone, 'WrongZSR'), zone)