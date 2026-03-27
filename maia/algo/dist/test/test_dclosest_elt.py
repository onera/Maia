import pytest
import pytest_parallel

import maia
import maia.pytree as PT

from maia.algo.dist import closest_elt as dCLO
from maia.algo.part import closest_elt as pCLO

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('elt_kind', ['NGON_n', 'TRI_3'])
def test_find_closest_elt(elt_kind, comm):
  
  surf = maia.factory.generate_dist_sphere(5, elt_kind, comm)
  pts = maia.factory.generate_dist_block(14, 'Poly', comm, origin=[-.5, -.5, -.5])

  dCLO.find_closest_element(surf, pts, 'Vertex', comm)

  # Compare use part. implementation, which is well tested
  psurf = maia.factory.partition_dist_tree(surf, comm)
  ppts = maia.factory.partition_dist_tree(pts, comm)
  PT.rm_nodes_from_label(ppts, 'DiscreteData_t')
  pCLO.find_closest_element(psurf, ppts, 'Vertex', comm)
  res = PT.find_node_from_name(ppts, 'ClosestElement')
  PT.set_name(res, 'pClosestElement')

  maia.transfer.part_tree_to_dist_tree_all(pts, ppts, comm)
  
  dclo = PT.find_node_from_name(pts,  'ClosestElement')
  pclo = PT.find_node_from_name(pts, 'pClosestElement')
  PT.set_name(pclo, 'ClosestElement') # Needed to compare but same name in tree
  assert PT.is_same_tree(dclo, pclo, abs_tol=1E-15)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('elt_kind', ['NFACE_n', 'TETRA_4'])
def test_find_closest_bnd(elt_kind, comm):
  
  vol = maia.factory.generate_dist_sphere(5, elt_kind, comm)
  pts = maia.factory.generate_dist_block(11, 'Poly', comm, origin=[-.5, -.5, -.5])

  dCLO.find_closest_boundary(vol, pts, 'Vertex', comm)

  # Compare use part. implementation, which is well tested
  pvol = maia.factory.partition_dist_tree(vol, comm)
  ppts = maia.factory.partition_dist_tree(pts, comm)
  PT.rm_nodes_from_label(ppts, 'DiscreteData_t')
  pCLO.find_closest_boundary(pvol, ppts, 'Vertex', comm)
  res = PT.find_node_from_name(ppts, 'ClosestElement')
  PT.set_name(res, 'pClosestElement')

  maia.transfer.part_tree_to_dist_tree_all(pts, ppts, comm)
  
  dclo = PT.find_node_from_name(pts,  'ClosestElement')
  pclo = PT.find_node_from_name(pts, 'pClosestElement')
  PT.set_name(pclo, 'ClosestElement') # Needed to compare but same name in tree
  assert PT.is_same_tree(dclo, pclo, abs_tol=1E-15)

@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(1)
def test_find_closest_bnd_propagation(comm):
  # Method is not reproductible in // (unless we use compute2 algo) -> test serial only
  vol = maia.factory.generate_dist_sphere(5, 'NFACE_n', comm)

  dCLO.find_closest_boundary_propagation(vol, comm)

  # Compare use part. implementation, which is well tested
  pvol = maia.factory.partition_dist_tree(vol, comm)
  PT.rm_nodes_from_label(pvol, 'DiscreteData_t')
  pCLO.find_closest_boundary_propagation(pvol, comm)
  res = PT.find_node_from_name(pvol, 'ClosestElement')
  PT.set_name(res, 'pClosestElement')

  maia.transfer.part_tree_to_dist_tree_all(vol, pvol, comm)

  dclo = PT.find_node_from_name(vol,  'ClosestElement')
  pclo = PT.find_node_from_name(vol, 'pClosestElement')
  PT.set_name(pclo, 'ClosestElement') # Needed to compare but same name in tree
  assert PT.is_same_tree(dclo, pclo, abs_tol=1E-15)


@pytest_parallel.mark.parallel(2)
def test_find_closest_bnd_S(comm):
  treeL = maia.factory.generate_dist_block(4, 'S', comm, origin=(-1,0,0)) 
  zone = PT.find_node_from_label(treeL, 'Zone_t')
  PT.set_name(zone, 'Left')
  treeR = maia.factory.generate_dist_block(3, 'S', comm)
  zone = PT.get_node_from_label(treeR, 'Zone_t')
  PT.set_name(zone, 'Right')
  tree = PT.union(treeL, treeR)
  
  pts_full = PT.yaml.to_cgns_tree("""
  Base CGNSBase_t [3, 2]:
    Zone Zone_t [[4, 0, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [-1, 0.01, -.5, .95]:
        CoordinateY DataArray_t R8 [0.05, 0.01, 0.5, .95]:
        CoordinateZ DataArray_t R8 [0, .5, 1, .25]:
  """)
  pts = maia.factory.full_to_dist_tree(pts_full, comm)

  dCLO.find_closest_boundary(tree, pts, 'Vertex', comm, surf_predicate=PT.pred.name_is('Zmax'))
  
  expected_dist   = [1, .5]   if comm.rank == 0 else [0, 0.75]
  expected_dom_id = [0, 1]    if comm.rank == 0 else [0, 1]
  expected_gnum   = [100, 33] if comm.rank == 0 else [104, 36]
  expected_dom_list = ['Base/Left', 'Base/Right']

  assert (PT.find_node_from_name(pts, 'Distance')[1] == expected_dist).all()
  assert (PT.find_node_from_name(pts, 'ClosestEltDomId')[1] == expected_dom_id).all()
  assert (PT.find_node_from_name(pts, 'ClosestEltGnum')[1] == expected_gnum).all()
  assert PT.get_str_value(PT.find_node_from_name(pts, 'DomainList')).split('\n') == expected_dom_list
