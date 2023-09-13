import pytest
import pytest_parallel
import os
from mpi4py import MPI
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia.utils         import test_utils as TU
from maia.pytree.yaml   import parse_yaml_cgns
from maia.factory import generate_dist_block
from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.factory import dist_from_part as DFP
dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

@pytest_parallel.mark.parallel(3)
class Test_discover_nodes_from_matching:
  pt = [\
  """
Zone.P0.N0 Zone_t:
  ZBC ZoneBC_t:
    BCA BC_t "wall":
      Family FamilyName_t "myfamily":
  """,
  """
Zone.P1.N0 Zone_t:
  ZGC ZoneGridConnectivity_t:
    match.0 GridConnectivity_t:
    match.1 GridConnectivity_t:
  """,
  """
Zone.P2.N0 Zone_t:
  ZBC ZoneBC_t:
    BCB BC_t "farfield":
      GridLocation GridLocation_t "FaceCenter":
Zone.P2.N1 Zone_t:
  ZBC ZoneBC_t:
    BCA BC_t "wall":
      Family FamilyName_t "myfamily":
  """]
  def test_simple(self, comm):
    part_zones = parse_yaml_cgns.to_nodes(self.pt[comm.Get_rank()])

    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, part_zones, 'ZoneBC_t/BC_t', comm)
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCB')) == "BC_t")

    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, part_zones, ['ZoneBC_t', 'BC_t'], comm)
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCB')) == "BC_t")

    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, part_zones, ["ZoneBC_t", 'BC_t'], comm)
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCB')) == "BC_t")

    dist_zone = PT.new_Zone('Zone')
    queries = ["ZoneBC_t", lambda n : PT.get_label(n) == "BC_t" and PT.get_name(n) != "BCA"]
    DFP.discover_nodes_from_matching(dist_zone, part_zones, queries, comm)
    assert (PT.get_node_from_path(dist_zone, 'ZBC/BCA') == None)
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCB')) == "BC_t")

  def test_short(self, comm):
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[comm.Get_rank()])
    part_nodes = [PT.get_node_from_path(zone, 'ZBC') for zone in PT.get_all_Zone_t(part_tree)\
      if PT.get_node_from_path(zone, 'ZBC') is not None]

    dist_node = PT.new_node('SomeName', 'UserDefinedData_t')
    DFP.discover_nodes_from_matching(dist_node, part_nodes, 'BC_t', comm)
    assert PT.get_node_from_path(dist_node, 'BCA') is not None
    assert PT.get_node_from_path(dist_node, 'BCB') is not None

    dist_node = PT.new_node('SomeName', 'UserDefinedData_t')
    queries = [lambda n : PT.get_label(n) == "BC_t" and PT.get_name(n) != "BCA"]
    DFP.discover_nodes_from_matching(dist_node, part_nodes, queries, comm)
    assert PT.get_node_from_path(dist_node, 'BCA') is None
    assert (PT.get_label(PT.get_node_from_path(dist_node, 'BCB')) == "BC_t")

  def test_getvalue(self, comm):
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[comm.Get_rank()])
    for zbc in PT.get_nodes_from_name(part_tree, 'ZBC'):
      PT.set_value(zbc, 'test')

    # get_value as a string
    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), 'ZoneBC_t/BC_t', comm)
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC')) == 'test'
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == None
    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), 'ZoneBC_t/BC_t', comm, get_value='none')
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC')) == None
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == None
    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), 'ZoneBC_t/BC_t', comm, get_value='all')
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC')) == 'test'
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == 'wall'
    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), 'ZoneBC_t/BC_t', comm, get_value='ancestors')
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC')) == 'test'
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == None
    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), 'ZoneBC_t/BC_t', comm, get_value='leaf')
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC')) == None
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == 'wall'

    # get_value as a list
    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), 'ZoneBC_t/BC_t', comm, get_value=[False, False])
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC')) == None
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == None
    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), 'ZoneBC_t/BC_t', comm, get_value=[True, False])
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC')) == 'test'
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCA')) == None

    # get_value and search with predicate as lambda
    dist_zone = PT.new_Zone('Zone')
    queries = ["ZoneBC_t", lambda n : PT.get_label(n) == "BC_t" and PT.get_name(n) != "BCA"]
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), queries, comm, get_value='all')
    assert PT.get_node_from_path(dist_zone, 'BCA') is None
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC')) == 'test'
    assert PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCB')) == 'farfield'

  def test_with_childs(self, comm):
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[comm.Get_rank()])

    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), 'ZoneBC_t/BC_t', comm,
                                child_list=['FamilyName_t', 'GridLocation'])
    assert (PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCA/Family')) == "myfamily")
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCA/Family')) == "FamilyName_t")
    assert (PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCB/GridLocation')) == "FaceCenter")
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCB/GridLocation')) == "GridLocation_t")

    dist_zone = PT.new_Zone('Zone')
    queries = ["ZoneBC_t", lambda n : PT.get_label(n) == "BC_t" and PT.get_name(n) != "BCA"]
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), queries, comm,
                                      child_list=['FamilyName_t', 'GridLocation'])
    assert (PT.get_value(PT.get_node_from_path(dist_zone, 'ZBC/BCB/GridLocation')) == "FaceCenter")
    assert (PT.get_label(PT.get_node_from_path(dist_zone, 'ZBC/BCB/GridLocation')) == "GridLocation_t")

  def test_with_rule(self, comm):
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[comm.Get_rank()])

    # Exclude from node name
    dist_zone = PT.new_Zone('Zone')
    queries = ["ZoneBC_t", lambda n : PT.get_label(n) == "BC_t" and not 'A' in PT.get_name(n)]
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), queries, comm,
                                      child_list=['FamilyName_t', 'GridLocation'])
    assert PT.get_node_from_path(dist_zone, 'ZBC/BCA') is None
    assert PT.get_node_from_path(dist_zone, 'ZBC/BCB') is not None

    # Exclude from node content
    dist_zone = PT.new_Zone('Zone')
    queries = ["ZoneBC_t", lambda n : PT.get_label(n) == "BC_t" and PT.get_child_from_label(n, 'FamilyName_t') is not None]
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), queries, comm,
                                      child_list=['FamilyName_t', 'GridLocation'])
    assert PT.get_node_from_path(dist_zone, 'ZBC/BCA') is not None
    assert PT.get_node_from_path(dist_zone, 'ZBC/BCB') is None

  def test_multiple(self, comm):
    gc_path = 'ZoneGridConnectivity_t/GridConnectivity_t'
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[comm.Get_rank()])

    dist_zone = PT.new_Zone('Zone')
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), gc_path, comm)
    assert PT.get_node_from_path(dist_zone, 'ZGC/match.0') is not None
    assert PT.get_node_from_path(dist_zone, 'ZGC/match.1') is not None

    dist_zone = PT.new_Zone('Zone')
    queries = ["ZoneGridConnectivity_t", "GridConnectivity_t"]
    DFP.discover_nodes_from_matching(dist_zone, PT.get_all_Zone_t(part_tree), queries, comm,\
        merge_rule=lambda path : MT.conv.get_split_prefix(path))
    assert PT.get_node_from_path(dist_zone, 'ZGC/match.0') is None
    assert PT.get_node_from_path(dist_zone, 'ZGC/match.1') is None
    assert PT.get_node_from_path(dist_zone, 'ZGC/match') is not None

  def test_zones(self, comm):
    part_tree = PT.new_CGNSTree()
    if comm.Get_rank() == 0:
      part_base = PT.new_CGNSBase('BaseA', parent=part_tree)
      PT.new_Zone('Zone.P0.N0', parent=part_base)
    elif comm.Get_rank() == 1:
      part_base = PT.new_CGNSBase('BaseB', parent=part_tree)
      PT.new_Zone('Zone.withdot.P1.N0', parent=part_base)
    elif comm.Get_rank() == 2:
      part_base = PT.new_CGNSBase('BaseA', parent=part_tree)
      PT.new_Zone('Zone.P2.N0', parent=part_base)
      PT.new_Zone('Zone.P2.N1', parent=part_base)

    dist_tree = PT.new_CGNSTree()
    DFP.discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t/Zone_t', comm,\
        merge_rule=lambda zpath : MT.conv.get_part_prefix(zpath))

    assert len(PT.get_all_Zone_t(dist_tree)) == 2
    assert PT.get_node_from_path(dist_tree, 'BaseA/Zone') is not None
    assert PT.get_node_from_path(dist_tree, 'BaseB/Zone.withdot') is not None

@pytest_parallel.mark.parallel(2)
def test_get_parts_per_blocks(comm):
  if comm.Get_rank() == 0:
    pt = """
    BaseI CGNSBase_t:
      ZoneA.P0.N0 Zone_t:
      ZoneA.P0.N1 Zone_t:
      ZoneB.P0.N0 Zone_t:
    """
  elif comm.Get_rank() == 1:
    pt = """
    BaseI  CGNSBase_t:
      ZoneA.P1.N0 Zone_t:
    BaseII CGNSBase_t:
      ZoneA.P1.N0 Zone_t:
    """
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)
  part_per_blocks = DFP.get_parts_per_blocks(part_tree, comm)
  if comm.Get_rank() == 0:
    assert PT.get_names(part_per_blocks['BaseI/ZoneA']) == ['ZoneA.P0.N0', 'ZoneA.P0.N1']
    assert PT.get_names(part_per_blocks['BaseI/ZoneB']) == ['ZoneB.P0.N0']
    assert PT.get_names(part_per_blocks['BaseII/ZoneA']) == []
  elif comm.Get_rank() == 1:
    assert PT.get_names(part_per_blocks['BaseI/ZoneA']) == ['ZoneA.P1.N0']
    assert PT.get_names(part_per_blocks['BaseI/ZoneB']) == []
    assert PT.get_names(part_per_blocks['BaseII/ZoneA']) == ['ZoneA.P1.N0']

@pytest_parallel.mark.parallel(1)
def test_get_joins_dist_tree(comm):
  pt = """
  BaseI CGNSBase_t:
    ZoneA.P0.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      ZGC ZoneGridConnectivity_t:
        matchAB.0 GridConnectivity_t "ZoneB.P0.N0":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
          GridConnectivityDonorName Descriptor_t "matchBA.0":
          GridLocation GridLocation_t "Vertex":
          PointList IndexArray_t [[8,3,5]]:
          :CGNS#GlobalNumbering UserDefinedData_t:
            Index DataArray_t [3,1,2]: 
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t [10,20,30,40,50,60,70,80,90,100]: 
  """
  expected_dt =  f"""
  BaseI CGNSBase_t:
    ZoneA Zone_t:
      ZoneType ZoneType_t "Unstructured":
      ZGC ZoneGridConnectivity_t:
        matchAB GridConnectivity_t "ZoneB":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
          GridConnectivityDonorName Descriptor_t "matchBA":
          GridLocation GridLocation_t "Vertex":
          PointList IndexArray_t [[30,50,80]]:
          :CGNS#Distribution UserDefinedData_t:
            Index DataArray_t {dtype} [0,3,3]: 
  """
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)
  expected_base = parse_yaml_cgns.to_node(expected_dt)
  dist_tree_jn = DFP.get_joins_dist_tree(part_tree, comm)
  assert PT.is_same_tree(PT.get_all_CGNSBase_t(dist_tree_jn)[0], expected_base)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("idx_dim", [3, 2])
def test_recover_dist_block_size(idx_dim, comm):
  if comm.Get_rank() == 0:
    pt = """
    Zone.P0.N0 Zone_t [[2,1,0], [3,2,0], [4,3,0]]: #Middle
      ZoneType ZoneType_t "Structured":
      ZoneGridConnectivity ZoneGridConnectivity_t:
        JN.P0.N0.LT.P0.N1 GridConnectivity1to1_t "Zone.P0.N1":
          PointRange IndexRange_t [[1,1], [1,3], [1,4]]:
        JN.P0.N0.LT.P1.N0 GridConnectivity1to1_t "Zone.P1.N0":
          PointRange IndexRange_t [[2,2], [1,3], [1,4]]:
    Zone.P0.N1 Zone_t [[2,1,0], [3,2,0], [4,3,0]]: #Left
      ZoneType ZoneType_t "Structured":
      ZoneGridConnectivity ZoneGridConnectivity_t:
        JN.P0.N1.LT.P0.N0 GridConnectivity1to1_t "Zone.P0.N0":
          PointRange IndexRange_t [[2,2], [1,3], [1,4]]:
    """
  elif comm.Get_rank() == 1:
    pt = """
    Zone.P1.N0 Zone_t [[2,1,0], [3,2,0], [4,3,0]]: #Right
      ZoneType ZoneType_t "Structured":
      ZoneGridConnectivity ZoneGridConnectivity_t:
        JN.P1.N0.LT.P0.N0 GridConnectivity1to1_t "Zone.P0.N0":
          PointRange IndexRange_t [[1,1], [1,3], [1,4]]:
    """
  part_zones = parse_yaml_cgns.to_nodes(pt)
  expected = np.array([[4,3,0],[3,2,0],[4,3,0]])

  if idx_dim == 2:
    for zone in part_zones:
      zone[1] = zone[1][0:2,:]
      for pr in PT.get_nodes_from_label(zone, 'IndexRange_t'):
        pr[1] = pr[1][0:2,:]
    expected = expected[0:2,:]

  dist_size = DFP._recover_dist_block_size(part_zones, comm)
  assert np.array_equal(dist_size, expected)

@pytest_parallel.mark.parallel(3)
def test_recover_dist_tree_ngon(comm):
  # Value test is already performed in subfunction tests
  part_tree = PT.new_CGNSTree()
  if comm.Get_rank() < 2:
    part_base = PT.new_CGNSBase(parent=part_tree)
    distri_ud = MT.newGlobalNumbering()
    if comm.Get_rank() == 0:
      # part_zone = G.cartNGon((0,0,0), (.5,.5,.5), (3,3,3))
      part_zone = PT.get_all_Zone_t(generate_dist_block(3, 'Poly', MPI.COMM_SELF))[0]
      PT.rm_children_from_label(part_zone, 'ZoneBC_t')
      PT.rm_nodes_from_name(part_zone, ':CGNS#Distribution')

      vtx_gnum = np.array([1,2,3,6,7,8,11,12,13,16,17,18,21,22,23,26,27,28,31,32,33,36,37,38,41,42,43], pdm_dtype)
      cell_gnum = np.array([1,2,5,6,9,10,13,14], pdm_dtype)
      ngon_gnum = np.array([1,2,3,6,7,8,11,12,13,16,17,18,21,22,25,26,29,30,33,34,37,38,41,42,45,46,49,
                            50,53,54,57,58,61,62,65,66], pdm_dtype)
      zbc = PT.new_ZoneBC(parent=part_zone)
      bc = PT.new_BC(type='BCWall', point_list=[[1,4,2,3]], parent=zbc)
      PT.new_GridLocation('FaceCenter', bc)
      MT.newGlobalNumbering({'Index' : np.array([1,2,3,4], pdm_dtype)}, parent=bc)
    else:
      # part_zone = G.cartNGon((1,0,0), (.5,.5,.5), (3,3,3))
      part_zone = PT.get_all_Zone_t(generate_dist_block(3, 'Poly', MPI.COMM_SELF, origin=[1., 0., 0.]))[0]
      PT.rm_children_from_label(part_zone, 'ZoneBC_t')
      PT.rm_nodes_from_name(part_zone, ':CGNS#Distribution')
      vtx_gnum =  np.array([3,4,5, 8,9,10,13,14,15,18,19,20,23,24,25,28,29,30,33,34,35,38,39,40,43,44,45], pdm_dtype)
      cell_gnum = np.array([3,4,7,8,11,12,15,16], pdm_dtype)
      ngon_gnum = np.array([3,4,5,8,9,10,13,14,15,18,19,20,23,24,27,28,31,32,35,36,39,40,43,44,
                            47,48,51,52,55,56,59,60,63,64,67,68], pdm_dtype)

    ngon = PT.get_node_from_path(part_zone, 'NGonElements')
    MT.newGlobalNumbering({'Element' : ngon_gnum}, parent=ngon)

    PT.new_DataArray('Vertex', vtx_gnum,  parent=distri_ud)
    PT.new_DataArray('Cell',   cell_gnum, parent=distri_ud)

    part_zone[0] = "Zone.P{0}.N0".format(comm.Get_rank())
    PT.add_child(part_base, part_zone)
    PT.add_child(part_zone, distri_ud)
    maia.algo.pe_to_nface(part_zone)

  dist_tree = DFP.recover_dist_tree(part_tree, comm)

  dist_zone = PT.get_node_from_name(dist_tree, 'Zone')
  assert (dist_zone[1] == [[45,16,0]]).all()
  assert (PT.get_node_from_path(dist_zone, 'NGonElements/ElementRange')[1] == [1,68]).all()
  assert (PT.get_node_from_path(dist_zone, 'NFaceElements/ElementRange')[1] == [69,84]).all()
  assert (PT.get_value(PT.get_node_from_path(dist_zone, 'ZoneBC/BC')) == "BCWall")

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("void_part", [True, False])
def test_recover_dist_tree_elt(void_part, comm):
  mesh_file = os.path.join(TU.mesh_dir, 'hex_prism_pyra_tet.yaml')
  dist_tree_bck = maia.io.file_to_dist_tree(mesh_file, comm)

  if void_part:
    weights = [1.] if comm.rank == 1 else []
  else:
    weights = [.5]
  zone_to_parts = {'Base/Zone' : weights}
  part_tree = maia.factory.partition_dist_tree(dist_tree_bck, comm, zone_to_parts=zone_to_parts)
  maia.transfer.dist_tree_to_part_tree_all(dist_tree_bck, part_tree, comm)

  dist_tree = DFP.recover_dist_tree(part_tree, comm)

  dist_zone = PT.get_node_from_name(dist_tree, 'Zone')
  assert (dist_zone[1] == [[11,4,0]]).all()
  assert (PT.get_node_from_path(dist_zone, 'Tris/ElementRange')[1] == [7,12]).all()
  assert (PT.get_node_from_path(dist_zone, 'Tets/ElementRange')[1] == [16,16]).all()
  assert len(PT.get_nodes_from_label(dist_zone, 'BC_t')) == 6
  assert len(PT.get_nodes_from_label(dist_zone, 'ZoneGridConnectivity_t')) == 0

  for elt in PT.get_nodes_from_label(dist_tree_bck, 'Elements_t'):
    PT.rm_node_from_path(elt, ':CGNS#Distribution/ElementConnectivity')
  
  assert PT.is_same_tree(dist_tree_bck, dist_tree, type_tol=True) #Input tree is pdm dtype

@pytest_parallel.mark.parallel(3)
def test_recover_dist_tree_s(comm):
  mesh_file = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dist_tree_bck = maia.io.file_to_dist_tree(mesh_file, comm)

  part_tree = maia.factory.partition_dist_tree(dist_tree_bck, comm)

  dist_tree = DFP.recover_dist_tree(part_tree, comm)

  # Force GridLocation to appear on dtree bck for comparison
  for bc in PT.get_nodes_from_label(dist_tree_bck, 'BC_t'):
    if PT.get_child_from_name(bc, 'GridLocation') is None:
      PT.new_GridLocation('Vertex', bc)

  assert PT.is_same_tree(dist_tree_bck, dist_tree, type_tol=True) #Recover create I4 zones

