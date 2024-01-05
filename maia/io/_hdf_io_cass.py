import Converter
import Converter.Filter
import Converter.PyTree   as C

import maia.pytree        as PT

from .fix_tree import fix_point_ranges, ensure_symmetric_gc1to1, fix_zone_datatype,\
                      rm_legacy_nodes, add_missing_pr_in_bcdataset

def add_sizes_to_zone_tree(zone, zone_path, size_data):
  """
  Creates the MyArray#Size node using the size_data dict on the given zone
  for the following nodes:
  - ElementConnectivity array of Element_t nodes
  - PointList (or Unstr PointRange) array of BC_t
  - PointList array of GC_t, GC1to1_t, BCDataSet_t and ZoneSubRegion_t nodes
  """
  for elmt in PT.iter_children_from_label(zone, 'Elements_t'):
    elmt_path = zone_path+"/"+elmt[0]
    ec_path   = elmt_path+"/ElementConnectivity"
    if PT.get_child_from_name(elmt, 'ElementStartOffset') is not None:
      PT.new_IndexArray('ElementConnectivity#Size', value=size_data[ec_path][2], parent=elmt)

  for zone_bc in PT.iter_children_from_label(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in PT.iter_children_from_label(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]
      if PT.get_child_from_name(bc, 'PointList') is not None:
        pl_path = bc_path+"/PointList"
        PT.new_IndexArray('PointList#Size', value=size_data[pl_path][2], parent=bc)
      for bcds in PT.iter_children_from_label(bc, 'BCDataSet_t'):
        if PT.get_child_from_name(bcds, 'PointList') is not None:
          pl_path = bc_path+"/"+bcds[0]+"/PointList"
          PT.new_IndexArray('PointList#Size', value=size_data[pl_path][2], parent=bcds)
        for bcdata, array in PT.get_children_from_predicates(bcds, 'BCData_t/DataArray_t', ancestors=True):
          data_path = '/'.join([bc_path, bcds[0], bcdata[0], array[0]])
          PT.new_IndexArray(f'{array[0]}#Size', value=size_data[data_path][2], parent=bcdata)

  for zone_gc in PT.iter_children_from_label(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+zone_gc[0]
    gcs = PT.get_children_from_label(zone_gc, 'GridConnectivity_t') \
        + PT.get_children_from_label(zone_gc, 'GridConnectivity1to1_t')
    for gc in gcs:
      gc_path = zone_gc_path+"/"+gc[0]
      if PT.get_child_from_name(gc, 'PointList') is not None:
        pl_path = gc_path+"/PointList"
        PT.new_IndexArray('PointList#Size', value=size_data[pl_path][2], parent=gc)
      if PT.get_child_from_name(gc, 'PointListDonor') is not None:
        pld_path = gc_path+"/PointListDonor"
        PT.new_IndexArray('PointListDonor#Size', value=size_data[pl_path][2], parent=gc)
        assert size_data[pld_path][2] == size_data[pl_path][2]

  for zone_subregion in PT.iter_children_from_label(zone, 'ZoneSubRegion_t'):
    zone_subregion_path = zone_path+"/"+zone_subregion[0]
    if PT.get_child_from_name(zone_subregion, 'PointList') is not None:
      pl_path = zone_subregion_path+"/PointList"
      PT.new_IndexArray('PointList#Size', value=size_data[pl_path][2], parent=zone_subregion)

  for flow_sol in PT.iter_children_from_label(zone, 'FlowSolution_t'):
    sol_path = zone_path + "/" + PT.get_name(flow_sol)
    if PT.get_child_from_name(flow_sol, 'PointList') is not None:
      pl_path = sol_path+"/PointList"
      PT.new_IndexArray('PointList#Size', value=size_data[pl_path][2], parent=flow_sol)
    for array in PT.get_children_from_label(flow_sol, 'DataArray_t'):
      #This one is DataArray to be detected in fix_tree.rm_legacy_nodes
      if size_data[sol_path+'/'+array[0]][1] != 'MT':
        PT.new_DataArray(f'{array[0]}#Size', value=size_data[sol_path+'/'+array[0]][2], parent=flow_sol)

def add_sizes_to_tree(size_tree, size_data):
  """
  Convience function which loops over zones to add size
  data in each one.
  """
  for base in PT.iter_children_from_label(size_tree, 'CGNSBase_t'):
    base_path = '/'+base[0]
    for zone in PT.iter_all_Zone_t(base):
      zone_path = base_path+"/"+zone[0]
      add_sizes_to_zone_tree(zone, zone_path, size_data)


def load_size_tree(filename, comm):
  """
    Load on all ranks a "size tree"
    a size tree is a partial tree that contains only the data needed to distribute the tree:
      nb of nodes, nb of elements, size of bcs and gcs...
    Convention:
      when we load the dimensions of an array "MyArray" without loading the array,
      then the dimensions are kept in a "MyArray#Size" node,
      at the same level as the array node would be
  """
  skeleton_depth  = -1
  skeleton_n_data = 4

  # In order to avoid filesystem overload only 1 proc reads the squeleton, then we broadcast
  if(comm.Get_rank() == 0):
    size_data = dict()
    assert Converter.checkFileType(filename) == "bin_hdf"
    size_tree = Converter.PyTree.convertFile2PyTree(filename,
                                                    skeletonData=[skeleton_n_data, skeleton_depth],
                                                    dataShape=size_data,
                                                    format='bin_hdf')
    fix_zone_datatype(size_tree, size_data)
    add_sizes_to_tree(size_tree, size_data)
    fix_point_ranges(size_tree)
    pred_1to1 = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity1to1_t'
    if PT.get_node_from_predicates(size_tree, pred_1to1) is not None:
      ensure_symmetric_gc1to1(size_tree)
    add_missing_pr_in_bcdataset(size_tree)
    rm_legacy_nodes(size_tree)
  else:
    size_tree = None

  size_tree = comm.bcast(size_tree, root=0)

  return size_tree

def load_partial(filename, dist_tree, hdf_filter, comm):
  hdf_filter = {f'/{key}' : data for key, data in hdf_filter.items()}
  partial_dict_load = C.convertFile2PartialPyTreeFromPath(filename, hdf_filter, comm)

  for path, data in partial_dict_load.items():
    if path.startswith('/'):
      path = path[1:]
    node = PT.get_node_from_path(dist_tree, path)
    node[1] = data

def load_grid_connectivity_property(filename, tree):
  """
  Load the GridConnectivityProperty_t nodes that may be present in joins.
  Because the transformation data is stored as a numpy array, these nodes
  are not loaded on the previous step.
  """
  # Prepare paths
  zgc_t_path = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t'
  is_gc = lambda n : PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  gc_prop_paths = []
  for base,zone,zone_gc in PT.iter_children_from_predicates(tree, zgc_t_path, ancestors=True):
    for gc in PT.iter_children_from_predicate(zone_gc, is_gc):
      gc_prop = PT.get_child_from_label(gc, 'GridConnectivityProperty_t')
      if gc_prop is not None:
        gc_prop_path = '/'.join([base[0], zone[0], zone_gc[0], gc[0], gc_prop[0]])
        gc_prop_paths.append(gc_prop_path)

  # Load
  gc_prop_nodes = Converter.Filter.readNodesFromPaths(filename, gc_prop_paths)

  # Replace with loaded data
  for path, gc_prop in zip(gc_prop_paths, gc_prop_nodes):
    gc_node_path = '/'.join(path.split('/')[:-1])
    gc_node = PT.get_node_from_path(tree, gc_node_path)
    PT.rm_children_from_label(gc_node, 'GridConnectivityProperty_t')
    PT.add_child(gc_node, gc_prop)


def write_partial(filename, dist_tree, hdf_filter, comm):
  hdf_filter = {f'/{key}' : data for key, data in hdf_filter.items()}
  C.convertPyTree2FilePartial(dist_tree, filename, comm, hdf_filter, ParallelHDF=True)

def read_full(filename):
  return C.convertFile2PyTree(filename)

def write_full(filename, tree, links=[]):
  C.convertPyTree2File(tree, filename, links=links)
