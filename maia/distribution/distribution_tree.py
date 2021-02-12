import Converter.Internal as     I

import maia.sids.sids as SIDS
from   .distribution_function                 import create_distribution_node

def compute_plist_or_prange_distribution(node, comm):
  """
  Compute the distribution for a given node using its PointList or PointRange child
  If a PointRange node is found, the total lenght is getted from the product
  of the differences for each direction (cgns convention (cgns convention :
  first and last are included).
  If a PointList node is found, the total lenght is getted from the product of
  PointList#Size arrays, which store the size of the PL in each direction.
  """

  pr_n = I.getNodeFromName1(node, 'PointRange')
  pl_n = I.getNodeFromName1(node, 'PointList')

  if(pr_n):
    pr_lenght = SIDS.point_range_n_elt(pr_n)
    create_distribution_node(pr_lenght, comm, 'Index', node)

  if(pl_n):
    pls_n   = I.getNodeFromName1(node, 'PointList#Size')
    pl_size = I.getValue(pls_n).prod()
    create_distribution_node(pl_size, comm, 'Index', node)

def compute_elements_distribution(zone, comm):
  """
  """
  if SIDS.ZoneType(zone) == 'Structured':
    pass
  else:
    elts = I.getNodesFromType1(zone, 'Elements_t')

    for elt in elts:
      er = I.getNodeFromName(elt, 'ElementRange')
      n_tot_elmt = er[1][1] - er[1][0] + 1
      create_distribution_node(n_tot_elmt, comm, 'Element', elt)

def compute_zone_distribution(zone, comm):
  """
  """
  n_vtx  = SIDS.zone_n_vtx (zone)
  n_cell = SIDS.zone_n_cell(zone)

  distrib_vtx  = create_distribution_node(n_vtx  , comm, 'Vertex', zone)
  distrib_cell = create_distribution_node(n_cell , comm, 'Cell'  , zone)

  compute_elements_distribution(zone, comm)

  for zone_subregion in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
    compute_plist_or_prange_distribution(zone_subregion, comm)

  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      compute_plist_or_prange_distribution(bc, comm)
      for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
        compute_plist_or_prange_distribution(bcds, comm)

  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    gcs = I.getNodesFromType1(zone_gc, 'GridConnectivity_t') + I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t')
    for gc in gcs:
      compute_plist_or_prange_distribution(gc, comm)

def add_distribution_info(dist_tree, comm, distribution_policy='uniform'):
  """
  """
  for base in I.getNodesFromType1(dist_tree, 'CGNSBase_t'):
    for zone in I.getNodesFromType1(base, 'Zone_t'):
      compute_zone_distribution(zone, comm)

def clean_distribution_info(dist_tree):
  """
  Remove the node related to distributio info from the dist_tree :
  :CGNS#Distribution, PointList#Size, ElementConnectivity#Size
  """
  for base in I.getNodesFromType1(dist_tree, 'CGNSBase_t'):
    for zone in I.getNodesFromType1(base, 'Zone_t'):
      I._rmNodesByName1(zone, ':CGNS#Distribution')
      for elmt in I.getNodesFromType1(zone, 'Elements_t'):
        I._rmNodesByName1(elmt, ':CGNS#Distribution')
        I._rmNodesByName1(elmt, 'ElementConnectivity#Size')
      for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
        for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
          I._rmNodesByName2(bc, ':CGNS#Distribution')
          I._rmNodesByName2(bc, 'PointList#Size')
      for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
        for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity_t') + \
                  I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t'):
          I._rmNodesByName1(gc, ':CGNS#Distribution')
          I._rmNodesByName1(gc, 'PointList#Size')
      for zone_subregion in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
        I._rmNodesByName1(zone_subregion, ':CGNS#Distribution')
        I._rmNodesByName1(zone_subregion, 'PointList#Size')
