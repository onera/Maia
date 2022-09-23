import maia.pytree        as PT

from .fix_tree import fix_point_ranges, fix_zone_datatype, load_grid_connectivity_property

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
      PT.new_PointList('ElementConnectivity#Size', value=size_data[ec_path][2], parent=elmt)

  for zone_bc in PT.iter_children_from_label(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in PT.iter_children_from_label(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]
      if PT.get_child_from_name(bc, 'PointList') is not None:
        pl_path = bc_path+"/PointList"
        PT.new_PointList('PointList#Size', value=size_data[pl_path][2], parent=bc)
      for bcds in PT.iter_children_from_label(bc, 'BCDataSet_t'):
        if PT.get_child_from_name(bcds, 'PointList') is not None:
          pl_path = bc_path+"/"+bcds[0]+"/PointList"
          PT.new_PointList('PointList#Size', value=size_data[pl_path][2], parent=bcds)

  for zone_gc in PT.iter_children_from_label(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+zone_gc[0]
    gcs = PT.get_children_from_label(zone_gc, 'GridConnectivity_t') \
        + PT.get_children_from_label(zone_gc, 'GridConnectivity1to1_t')
    for gc in gcs:
      gc_path = zone_gc_path+"/"+gc[0]
      if PT.get_child_from_name(gc, 'PointList') is not None:
        pl_path = gc_path+"/PointList"
        PT.new_PointList('PointList#Size', value=size_data[pl_path][2], parent=gc)
      if PT.get_child_from_name(gc, 'PointListDonor') is not None:
        pld_path = gc_path+"/PointListDonor"
        assert size_data[pld_path][2] == size_data[pl_path][2]

  for zone_subregion in PT.iter_children_from_label(zone, 'ZoneSubRegion_t'):
    zone_subregion_path = zone_path+"/"+zone_subregion[0]
    if PT.get_child_from_name(zone_subregion, 'PointList') is not None:
      pl_path = zone_subregion_path+"/PointList"
      PT.new_PointList('PointList#Size', value=size_data[pl_path][2], parent=zone_subregion)

  for flow_sol in PT.iter_children_from_label(zone, 'FlowSolution_t'):
    sol_path = zone_path + "/" + PT.get_name(flow_sol)
    if PT.get_child_from_name(flow_sol, 'PointList') is not None:
      pl_path = sol_path+"/PointList"
      PT.new_PointList('PointList#Size', value=size_data[pl_path][2], parent=flow_sol)

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


def load_collective_size_tree_cassiopee(filename, comm):
  """
    Load on all ranks a "size tree"
    a size tree is a partial tree that contains only the data needed to distribute the tree:
      nb of nodes, nb of elements, size of bcs and gcs...
    Convention:
      when we load the dimensions of an array "MyArray" without loading the array,
      then the dimensions are kept in a "MyArray#Size" node,
      at the same level as the array node would be
  """
  import Converter
  skeleton_depth  = 7
  skeleton_n_data = 3

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
    load_grid_connectivity_property(filename, size_tree)
  else:
    size_tree = None

  size_tree = comm.bcast(size_tree, root=0)

  return size_tree

def load_collective_size_tree_h5py(filename, comm):
  from .hdf._hdf_cgns import load_lazy_wrapper

  if comm.Get_rank() == 0:

    #Define the function used to skip or not data
    def skip_data(pname_and_label, name_and_label):
      if pname_and_label[1] == 'UserDefinedData_t':
        return False
      if name_and_label[1] in ['IndexArray_t']:
        return True
      if name_and_label[1] in ['DataArray_t']:
        if pname_and_label[1] not in ['Periodic_t', 'ReferenceState_t']:
          return True
      return False

    size_tree = load_lazy_wrapper(filename, skip_data)

    fix_point_ranges(size_tree)
  else:
    size_tree = None

  size_tree = comm.bcast(size_tree, root=0)

  return size_tree

def load_collective_size_tree(filename, comm, legacy):
  if legacy:
    return load_collective_size_tree_cassiopee(filename, comm)
  else:
    return load_collective_size_tree_h5py(filename, comm)
