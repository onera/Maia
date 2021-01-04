from   mpi4py             import MPI
import numpy              as     np
import Converter.PyTree   as     C
import Converter.Filter   as     CFilter
import Converter.Internal as     I

def correct_point_range(size_tree, size_data):
  """
  Permute start and end of PointRange or PointRangeDonor nodes found in GridConnectivity1to1_t
  in order to
  a. be consistent with the transform node
  b. keep the symmetry PR|a->b = PRDonor|b->a
  """
  for base in I.getBases(size_tree):
    base_name = I.getName(base)
    for zone in I.getZones(base):
      zone_name = I.getName(zone)
      for zone_gc in I.getNodesFromType1(zone, "ZoneGridConnectivity_t"):
        for gc in I.getNodesFromType1(zone_gc, "GridConnectivity1to1_t"):
          gc_path     = base_name + '/' + zone_name
          gc_opp_path = I.getValue(gc)
          if not '/' in gc_opp_path:
            gc_opp_path = base_name + '/' + gc_opp_path

          transform     = I.getValue(I.getNodeFromName1(gc, 'Transform'))
          point_range   = I.getValue(I.getNodeFromName1(gc, 'PointRange'))
          point_range_d = I.getValue(I.getNodeFromName1(gc, 'PointRangeDonor'))

          donor_dir    = abs(transform) - 1
          nb_points    = point_range[:,1] - point_range[:,0]
          nb_points_d  = np.sign(transform)*(point_range_d[donor_dir,1] - point_range_d[donor_dir,0])
          dir_to_swap  = (nb_points != nb_points_d)

          if gc_path < gc_opp_path:
            point_range_d[dir_to_swap, 0], point_range_d[dir_to_swap, 1] = \
                point_range_d[dir_to_swap, 1], point_range_d[dir_to_swap, 0]
          elif gc_path > gc_opp_path:
            point_range[dir_to_swap, 0], point_range[dir_to_swap, 1] = \
                point_range[dir_to_swap, 1], point_range[dir_to_swap, 0]
          # If same base/zone, transform should be 1, 2, 3
          else:
            assert (dir_to_swap == False).all()

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
