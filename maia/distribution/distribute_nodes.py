import Converter.Internal      as I
import maia.sids.Internal_ext  as IE

from maia.sids import sids
from maia import npy_pdm_gnum_dtype as pdm_dtype
from . import distribution_function as DIF

def distribute_pl_node(node, comm):
  """
  Distribute a standard node having a PointList (and its childs) over several processes,
  using uniform distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  dist_node = I.copyTree(node)
  n_elem = sids.Subset.n_elem(dist_node)
  distri = DIF.uniform_distribution(n_elem, comm).astype(pdm_dtype)

  #PL and PLDonor
  for array_n in IE.getNodesByMatching(dist_node, ['IndexArray_t']):
    array_n[1] = array_n[1][0][distri[0]:distri[1]].reshape(1,-1, order='F')
  #Data Arrays
  has_subset = lambda n : I.getNodeFromName1(n, 'PointList') is not None or I.getNodeFromName1(n, 'PointRange') is not None
  bcds_without_pl = lambda n : I.getType(n) == 'BCDataSet_t' and not has_subset(n)
  bcds_without_pl_query = [bcds_without_pl, 'BCData_t', 'DataArray_t']
  for array_path in ['DataArray_t', 'BCData_t/DataArray_t', bcds_without_pl_query]:
    for array_n in IE.getNodesByMatching(dist_node, array_path):
      array_n[1] = array_n[1][distri[0]:distri[1]]

  #Additionnal treatement for subnodes with PL (eg bcdataset)
  has_pl = lambda n : I.getName(n) not in ['PointList', 'PointRange'] and has_subset(n)
  for child in [node for node in I.getChildren(dist_node) if has_pl(node)]:
    dist_child = distribute_pl_node(child, comm)
    child[2] = dist_child[2]

  IE.newDistribution({'Index' : distri}, dist_node)

  return dist_node

def distribute_data_node(node, comm):
  """
  Distribute a standard node having arrays supported by allCells or allVertices over several processes,
  using uniform distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  dist_node = I.copyTree(node)
  assert I.getNodeFromName(dist_node, 'PointList') is None

  for array in I.getNodesFromType1(dist_node, 'DataArray_t'):
    distri = DIF.uniform_distribution(array[1].size, comm)
    array[1] = array[1].reshape(-1, order='F')[distri[0] : distri[1]]

  return dist_node

def distribute_element_node(node, comm):
  """
  Distribute a standard element node over several processes, using uniform distribution.
  Mainly useful for unit tests. Node must be know by each process.
  """
  assert I.getType(node) == 'Elements_t'
  assert sids.ElementCGNSName(node) != "MIXED", "Mixed elements are not supported"
  dist_node = I.copyTree(node)

  n_elem = sids.ElementSize(node)
  distri = DIF.uniform_distribution(n_elem, comm).astype(pdm_dtype)
  IE.newDistribution({'Element' : distri}, dist_node)

  ec = I.getNodeFromName1(dist_node, 'ElementConnectivity')
  if sids.ElementCGNSName(node) in ['NGON_n', 'NFACE_n']:
    eso = I.getNodeFromName1(dist_node, 'ElementStartOffset')
    distri_ec = eso[1][[distri[0], distri[1], -1]]
    ec[1] = ec[1][distri_ec[0] : distri_ec[1]]
    eso[1] = eso[1][distri[0]:distri[1]+1]

    IE.newDistribution({'ElementConnectivity' : distri_ec.astype(pdm_dtype)}, dist_node)
  else:
    n_vtx = sids.ElementNVtx(node)
    ec[1] = ec[1][n_vtx*distri[0] : n_vtx*distri[1]]
    IE.newDistribution({'ElementConnectivity' : n_vtx*distri}, dist_node)
  
  pe = I.getNodeFromName1(dist_node, 'ParentElements')
  if pe is not None:
    pe[1] = (pe[1][distri[0] : distri[1]]).copy(order='F') #Copy is needed to have contiguous memory
  
  return dist_node

def distribute_tree(tree, comm, owner=None):
  """
  Distribute a standard cgns tree over several processes, using uniform distribution.
  Mainly useful for unit tests. If owner is None, tree must be know by each process;
  otherwise, tree is broadcasted from the owner process
  """

  if owner is not None:
    tree = comm.bcast(tree, root=owner)

  # Do a copy to capture all original nodes
  dist_tree = I.copyTree(tree)
  for zone in IE.getNodesByMatching(dist_tree, ['CGNSBase_t', 'Zone_t']):
    # > Cell & Vertex distribution
    n_vtx  = sids.Zone.n_vtx(zone)
    n_cell = sids.Zone.n_cell(zone)
    zone_distri = {'Vertex' : DIF.uniform_distribution(n_vtx , comm).astype(pdm_dtype),
                   'Cell'   : DIF.uniform_distribution(n_cell, comm).astype(pdm_dtype)}
    if sids.Zone.Type(zone) == 'Structured':
      zone_distri['Face'] = DIF.uniform_distribution(sids.Zone.n_face(zone), comm).astype(pdm_dtype)

    IE.newDistribution(zone_distri, zone)

    # > Coords
    grid_coords = I.getNodesFromType1(zone, 'GridCoordinates_t')
    for grid_coord in grid_coords:
      I._rmNode(zone, grid_coord)
      I._addChild(zone, distribute_data_node(grid_coord, comm))

    # > Elements
    elts = I.getNodesFromType1(zone, 'Elements_t')
    for elt in elts:
      I._rmNode(zone, elt)
      I._addChild(zone, distribute_element_node(elt, comm))

    # > Flow Solutions
    sols = I.getNodesFromType1(zone, 'FlowSolution_t') + I.getNodesFromType1(zone, 'DiscreteData_t')
    for sol in sols:
      I._rmNode(zone, sol)
      if I.getNodeFromName1(sol, 'PointList') is None:
        I._addChild(zone, distribute_data_node(sol, comm))
      else:
        I._addChild(zone, distribute_pl_node(sol, comm))

    # > BCs
    zonebcs = I.getNodesFromType1(zone, 'ZoneBC_t')
    for zonebc in zonebcs:
      I._rmNode(zone, zonebc)
      dist_zonebc = I.createChild(zone, I.getName(zonebc), 'ZoneBC_t')
      for bc in I.getNodesFromType1(zonebc, 'BC_t'):
        I._addChild(dist_zonebc, distribute_pl_node(bc, comm))

    # > GCs
    zonegcs = I.getNodesFromType1(zone, 'ZoneGridConnectivity_t')
    for zonegc in zonegcs:
      I._rmNode(zone, zonegc)
      dist_zonegc = I.createChild(zone, I.getName(zonegc), 'ZoneGridConnectivity_t')
      for gc in I.getNodesFromType1(zonegc, 'GridConnectivity_t') + I.getNodesFromType1(zonegc, 'GridConnectivity1to1_t'):
        I._addChild(dist_zonegc, distribute_pl_node(gc, comm))

    # > ZoneSubRegion
    zone_subregions = I.getNodesFromType1(zone, 'ZoneSubRegion_t')
    for zone_subregion in zone_subregions:
      # Trick if related to an other node -> add pl
      matching_region_path = IE.getSubregionExtent(zone_subregion, zone)
      if matching_region_path != I.getName(zone_subregion):
        I._addChild(zone_subregion, I.getNodeFromPath(zone, matching_region_path + '/PointList'))
        I._addChild(zone_subregion, I.getNodeFromPath(zone, matching_region_path + '/PointRange'))
      dist_zone_subregion = distribute_pl_node(zone_subregion, comm)
      if matching_region_path != I.getName(zone_subregion):
        I._rmNodesByName(dist_zone_subregion, 'PointList')
        I._rmNodesByName(dist_zone_subregion, 'PointRange')
        I._rmNode(dist_zone_subregion, IE.getDistribution(dist_zone_subregion))

      I._addChild(zone, dist_zone_subregion)
      I._rmNode(zone, zone_subregion)

  return dist_tree
