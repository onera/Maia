from   mpi4py             import MPI
import Converter.PyTree   as     C
import Converter.Filter   as     CFilter
import Converter.Internal as     I

def correct_point_range(size_tree, size_data):
  """
  Performs adaptation or correction in order to be properly setup for other algorithm
  """

  print("correct_point_range")

def load_grid_connectivity_property(filename, tree):
  """
  Load the GridConnectivityProperty_t nodes that may be present in joins.
  Because the transformation data is stored as numpy array, these nodes
  are not loaded on the previous step.
  """
  # Prepare pathes
  gc_prop_pathes = []
  for base in I.getBases(tree):
    for zone in I.getZones(base):
      for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
        for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity_t'):
          gc_prop = I.getNodeFromType1(gc, 'GridConnectivityProperty_t')
          if gc_prop is not None:
            gc_prop_path = '/'.join([base[0], zone[0], zone_gc[0], gc[0], gc_prop[0]])
            gc_prop_pathes.append(gc_prop_path)

  # Load
  gc_prop_nodes = CFilter.readNodesFromPaths(filename, gc_prop_pathes)

  # Replace with loaded data
  for path, gc_prop in zip(gc_prop_pathes, gc_prop_nodes):
    gc_node_path = '/'.join(path.split('/')[:-1])
    gc_node = I.getNodeFromPath(tree, gc_node_path)
    I._rmNodesByType(gc_node, 'GridConnectivityProperty_t')
    I._addChild(gc_node, gc_prop)
