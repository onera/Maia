from   mpi4py             import MPI
import Converter.Internal as     I
import Converter.PyTree   as     C

from .distribution_zone import compute_zone_distribution


#
# uniform = compute_proc_indices

# def master_of_each_node():
#   sub_comm = master_of_each_node(comm)
#   compute_proc_indices(sub_comm)
#   #
# Heterogene : Element / BC / Join


def add_distribution_info(dist_tree, comm, distribution_policy='uniform'):
  """
  """
  for base in I.getNodesFromType1(dist_tree, 'CGNSBase_t'):
  # DFAM.compute_family_distribution(dist_tree)
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
