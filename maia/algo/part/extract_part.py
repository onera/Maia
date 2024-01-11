import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.utils.logging as mlog
from   maia.factory       import dist_from_part
from   maia.utils         import np_utils
from   .extract_part_s    import exchange_field_s, extract_part_one_domain_s
from   .extract_part_u    import exchange_field_u, extract_part_one_domain_u
from   .extraction_utils  import LOC_TO_DIM

import numpy as np

import Pypdm.Pypdm as PDM


def set_transfer_dataset(bc_n, zsr_bc_n, zone_type):

  if zone_type=='Structured':
    unwanted_type = 'IndexArray_t'
    required_name = 'PointRange'
  else:
    unwanted_type = 'IndexRange_t'
    required_name = 'PointList'
  there_is_dataset = False
  assert PT.get_child_from_predicates(bc_n, f'BCDataSet_t/{unwanted_type}') is None,\
                 'BCDataSet_t with PointList aren\'t managed'
  ds_arrays = PT.get_children_from_predicates(bc_n, 'BCDataSet_t/BCData_t/DataArray_t')
  for ds_array in ds_arrays:
    PT.new_DataArray(name=PT.get_name(ds_array), value=PT.get_value(ds_array), parent=zsr_bc_n)
  if len(ds_arrays) != 0:
    there_is_dataset = True
    # PL and Location is needed for data exchange, but this should be done in ZSR func
    for name in [required_name, 'GridLocation']:
      PT.add_child(zsr_bc_n, PT.get_child_from_name(bc_n, name))
  return there_is_dataset

class Extractor:
  def __init__( self,
                part_tree, patch, location, comm,
                # equilibrate=True,
                graph_part_tool="hilbert"):

    self.part_tree     = part_tree
    self.exch_tool_box = dict()
    self.comm          = comm

    # Get zones by domains
    part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)
    # Check : monodomain
    assert len(part_tree_per_dom.values()) == 1

    # > Check if U or S (working because monodomain)
    zone_type = PT.get_node_from_name(part_tree, 'ZoneType')
    is_struct = PT.get_value(zone_type)=='Structured' if zone_type is not None else False
    self.is_struct = comm.allreduce(is_struct)
    # if self.is_struct and equilibrate:
    #   raise NotImplementedError('Structured Extractor with equilibrate=True option is not yet implemented.')
    # if not self.is_struct and not equilibrate:
    #   raise NotImplementedError('Unstructured Extractor with equilibrate=False option is not yet implemented.')

    # ExtractPart dimension
    self.location = location
    self.dim = LOC_TO_DIM[location]
    assert self.dim in [0,2,3], "[MAIA] Error : dimensions 1 not yet implemented"
    #CGNS does not support 0D, so keep input dim in this case (which is 3 since 2d is not managed)
    if location == 'Vertex':
      if self.is_struct:
        cell_dim = -1
        for domain_prs in patch:
          for part_pr in domain_prs:
            if part_pr.size!=0:
              size_per_dim = np.diff(part_pr)[:,0]
              idx = np.where(size_per_dim!=0)[0]
              cell_dim = idx.size
        cell_dim = comm.allreduce(cell_dim, op=MPI.MAX)
      else:
        cell_dim = 3    
    else:
      cell_dim = self.dim 
    
    assert graph_part_tool in ["hilbert","parmetis","ptscotch"]
    assert not( (self.dim==0) and graph_part_tool in ['parmetis', 'ptscotch']),\
           '[MAIA] Vertex extraction not available with parmetis or ptscotch partitioning. Please check your script.' 

    # ExtractPart CGNSTree
    extract_tree = PT.new_CGNSTree()
    extract_base = PT.new_CGNSBase('Base', cell_dim=cell_dim, phy_dim=3, parent=extract_tree)
    # Compute extract part of each domain
    for i_domain, dom_part_zones in enumerate(part_tree_per_dom.items()):
      dom_path   = dom_part_zones[0]
      part_zones = dom_part_zones[1]
      if self.is_struct:
        extract_zones, etb = extract_part_one_domain_s(part_zones, patch[i_domain], self.location, comm)
      else:
        extract_zones, etb = extract_part_one_domain_u(part_zones, patch[i_domain], self.location, comm,
                                                      # equilibrate=equilibrate,
                                                      graph_part_tool=graph_part_tool)
      self.exch_tool_box[dom_path] = etb
      for extract_zone in extract_zones:
        if PT.Zone.n_vtx(extract_zone)!=0:
          PT.add_child(extract_base, extract_zone)

      # > Clean orphan GC
      if self.is_struct:
        all_zone_name_l = PT.get_names(PT.get_children_from_label(extract_base, 'Zone_t'))
        all_zone_name_l = comm.allgather(all_zone_name_l)
        all_zone_name = list(np.concatenate(all_zone_name_l))

        for zone_n in PT.get_children_from_label(extract_base, 'Zone_t'):
          for zgc_n in PT.get_children_from_label(zone_n, 'ZoneGridConnectivity_t'):
            for gc_n in PT.get_children_from_label(zgc_n, 'GridConnectivity1to1_t'):
              matching_zone_name = PT.get_value(gc_n)
              if matching_zone_name not in all_zone_name:
                PT.rm_child(zgc_n, gc_n)
            if len(PT.get_children_from_label(zgc_n, 'GridConnectivity1to1_t'))==0:
              PT.rm_child(zone_n, zgc_n)

    self.extract_tree = extract_tree

  def exchange_fields(self, fs_container):
    exchange_fld_func = exchange_field_s if self.is_struct else exchange_field_u
    exchange_fld_func(self.part_tree,  self.extract_tree , self.dim, self.exch_tool_box,\
          fs_container, self.comm)

  def get_extract_part_tree(self) :
    return self.extract_tree


def extract_part_from_zsr(part_tree, zsr_name, comm,
                          transfer_dataset=True,
                          containers_name=[], **options):
  """Extract the submesh defined by the provided ZoneSubRegion from the input volumic
  partitioned tree.

  Dimension of the output mesh is set up accordingly to the GridLocation of the ZoneSubRegion.
  Submesh is returned as an independant partitioned CGNSTree and includes the relevant connectivities.

  Fields found under the ZSR node are transfered to the extracted mesh if ``transfer_dataset`` is set to True.
  In addition, additional containers specified in ``containers_name`` list are transfered to the extracted tree.
  Containers to be transfered can be either of label FlowSolution_t or ZoneSubRegion_t.

  Args:
    part_tree       (CGNSTree)    : Partitioned tree from which extraction is computed. U-Elts
      connectivities are *not* managed.
    zsr_name        (str)         : Name of the ZoneSubRegion_t node
    comm            (MPIComm)     : MPI communicator
    transfer_dataset(bool)        : Transfer (or not) fields stored in ZSR to the extracted mesh (default to ``True``)
    containers_name (list of str) : List of the names of the fields containers to transfer
                                    on the output extracted tree.
    **options: Options related to the extraction.
  Returns:
    extract_tree (CGNSTree)  : Extracted submesh (partitioned)

  Extraction can be controled by the optional kwargs:

    - ``graph_part_tool`` (str) -- Partitioning tool used to balance the extracted zones.
      Admissible values are ``hilbert, parmetis, ptscotch``. Note that
      vertex-located extractions require hilbert partitioning. Defaults to ``hilbert``.
  
  Important:
    - Input tree must have a U-NGon or Structured connectivity
    - Partitions must come from a single initial domain on input tree.
  
  See also:
    :func:`create_extractor_from_zsr` takes the same parameters, excepted ``containers_name``,
    and returns an Extractor object which can be used to exchange containers more than once through its
    ``Extractor.exchange_fields(container_name)`` method.
  
  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_zsr@start
      :end-before:  #extract_from_zsr@end
      :dedent: 2
  """

  start = time.time()
  extractor = create_extractor_from_zsr(part_tree, zsr_name, comm, **options)

  l_containers_name = [name for name in containers_name]
  if transfer_dataset and zsr_name not in l_containers_name:
    l_containers_name += [zsr_name]
  if l_containers_name:
    extractor.exchange_fields(l_containers_name)
  end = time.time()

  extract_tree = extractor.get_extract_part_tree()

  # Print some light stats
  elts_kind = ['vtx', 'edges', 'faces', 'cells'][extractor.dim]
  if extractor.dim == 0:
    n_cell = sum([PT.Zone.n_vtx(zone) for zone in PT.iter_all_Zone_t(extract_tree)])
  else:
    n_cell = sum([PT.Zone.n_cell(zone) for zone in PT.iter_all_Zone_t(extract_tree)])
  n_cell_all = comm.allreduce(n_cell, MPI.SUM)
  mlog.info(f"Extraction from ZoneSubRegion \"{zsr_name}\" completed ({end-start:.2f} s) -- "
            f"Extracted tree has locally {mlog.size_to_str(n_cell)} {elts_kind} "
            f"(Σ={mlog.size_to_str(n_cell_all)})")


  return extract_tree


def create_extractor_from_zsr(part_tree, zsr_path, comm, **options):
  """Same as extract_part_from_zsr, but return the extractor object."""
  # Get zones by domains

  graph_part_tool = options.get("graph_part_tool", "hilbert")

  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  # Get patch for each partitioned zone and group it by domain
  patch = list()
  location = ''
  for domain, part_zones in part_tree_per_dom.items():
    patch_domain = list()
    for part_zone in part_zones:
      zsr_node = PT.get_node_from_path(part_zone, zsr_path)
      if zsr_node is not None:
        #Follow BC or GC link
        related_node = PT.Subset.ZSRExtent(zsr_node, part_zone)
        zsr_node     = PT.get_node_from_path(part_zone, related_node)
        patch_domain.append(PT.Subset.getPatch(zsr_node)[1])
        location = PT.Subset.GridLocation(zsr_node)
      else: # ZSR does not exists on this partition
        patch_domain.append(np.empty((1,0), np.int32))
    patch.append(patch_domain)

  # Get location if proc has no zsr
  location = comm.allreduce(location, op=MPI.MAX)

  return Extractor(part_tree, patch, location, comm,
                   graph_part_tool=graph_part_tool)



def extract_part_from_bc_name(part_tree, bc_name, comm,
                              transfer_dataset=True,
                              containers_name=[],
                              **options):
  """Extract the submesh defined by the provided BC name from the input volumic
  partitioned tree.

  Behaviour and arguments of this function are similar to those of :func:`extract_part_from_zsr`:
  ``zsr_name`` becomes ``bc_name`` and optional ``transfer_dataset`` argument allows to 
  transfer BCDataSet from BC to the extracted mesh (default to ``True``).

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_bc_name@start
      :end-before:  #extract_from_bc_name@end
      :dedent: 2
  """

  # Local copy of the part_tree to add ZSR 
  l_containers_name = [name for name in containers_name]
  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)

  # Adding ZSR to tree
  there_is_bcdataset = False
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:
      bc_n = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t') 
      if bc_n is not None:
        zsr_bc_n  = PT.new_ZoneSubRegion(name=bc_name, bc_name=bc_name, parent=part_zone)
        if transfer_dataset:
          there_is_bcdataset = set_transfer_dataset(bc_n, zsr_bc_n, PT.Zone.Type(part_zone))

  if transfer_dataset and comm.allreduce(there_is_bcdataset, MPI.LOR):
    l_containers_name.append(bc_name) # not to change the initial containers_name list


  return extract_part_from_zsr(local_part_tree, bc_name, comm,
                               transfer_dataset=False,
                               containers_name=l_containers_name,
                             **options)



def extract_part_from_family(part_tree, family_name, comm,
                             transfer_dataset=True,
                             containers_name=[],
                             **options):
  """Extract the submesh defined by the provided family name from the input volumic
  partitioned tree.
  
  Family related nodes can be labelled either as BC_t or ZoneSubRegion_t, but their
  GridLocation must have the same value. They generate a merged output on the resulting extracted tree.

  Behaviour and arguments of this function are similar to those of :func:`extract_part_from_zsr`.

  Warning:
    Only U-NGon meshes are managed in this function.

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_family@start
      :end-before:  #extract_from_family@end
      :dedent: 2
  """

  if PT.get_value(PT.get_node_from_name(part_tree, 'ZoneType'))=='Structured':
    raise RuntimeError(f'extract_part_from_family function is not implemented for Structured meshes.')

  # Local copy of the part_tree to add ZSR 
  l_containers_name = [name for name in containers_name]
  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)

  # > Discover family related nodes
  in_fam = lambda n : PT.predicate.belongs_to_family(n, family_name, True)
  is_regionname = lambda n: PT.get_name(n) in ['BCRegionName', 'GridConnectivityRegionName']
  bc_gc_in_fam = lambda n: PT.get_name(n) in region_node_names
  zsr_has_regionname = lambda n: PT.get_label(n)=="ZoneSubRegion_t" and \
                                (PT.get_child_from_name(n, 'BCRegionName')               is not None or \
                                 PT.get_child_from_name(n, 'GridConnectivityRegionName') is not None)
  fam_to_node_paths = lambda zone, family_name: PT.predicates_to_paths(zone, [lambda n: PT.get_label(n)=='ZoneSubRegion_t' and in_fam]) + \
                                                PT.predicates_to_paths(zone, ['ZoneBC_t', in_fam])


  fam_node_paths = list()
  for domain, part_zones in part_tree_per_dom.items():
    dist_zone = PT.new_Zone('Zone')
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ['ZoneSubRegion_t' and in_fam], comm, get_value='leaf', child_list=['FamilyName_t', 'GridLocation_t', 'Descriptor_t'])
    region_node_names = list()
    for zsr_with_regionname_n in PT.get_children_from_predicate(dist_zone, zsr_has_regionname):
      region_node = PT.get_child_from_predicate(zsr_with_regionname_n, is_regionname)
      region_node_names.append(PT.get_value(region_node))
    child_list = ['AdditionalFamilyName_t', 'FamilyName_t', 'GridLocation_t']
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ['ZoneBC_t', lambda n: in_fam(n) or bc_gc_in_fam(n)], comm, get_value='leaf', child_list=child_list)
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ['ZoneGridConnectivity_t', bc_gc_in_fam], comm, get_value='leaf', child_list=child_list)

    fam_node_paths.extend(fam_to_node_paths(dist_zone, family_name))

    gl_nodes = PT.get_nodes_from_label(dist_zone, 'GridLocation_t')
    location = [PT.get_value(n) for n in gl_nodes]
    if len(set(location)) > 1:
      # Not checking subregion extents, possible ?
      raise ValueError(f"Specified family refers to nodes with different GridLocation value : {set(location)}.")
     
  # Adding ZSR to tree
  there_is_bcdataset = dict((path, False) for path in fam_node_paths)
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:

      fam_pl = list()
      for path in fam_node_paths:
        fam_node = PT.get_node_from_path(part_zone, path)
        if fam_node is not None:

          if PT.get_label(fam_node)=='BC_t':
            bc_name = PT.get_name(fam_node)
            if transfer_dataset:
              zsr_bc_n = PT.new_ZoneSubRegion(name=bc_name, bc_name=bc_name)
              there_is_bcdataset[path] = set_transfer_dataset(fam_node, zsr_bc_n, PT.Zone.Type(part_zone))
              if PT.get_child_from_label(zsr_bc_n, 'DataArray_t') is not None:
                PT.add_child(part_zone, zsr_bc_n)

          if PT.get_label(fam_node)=="ZoneSubRegion_t":
            if transfer_dataset:
              if PT.get_child_from_label(fam_node, 'DataArray_t') is not None:
                there_is_bcdataset[path] = True
            related_path = PT.Subset.ZSRExtent(fam_node, part_zone)
            fam_node = PT.get_node_from_path(part_zone, related_path)
          pl_n = PT.get_child_from_name(fam_node, 'PointList')
          fam_pl.append(PT.get_value(pl_n))

      fam_pl = np_utils.concatenate_np_arrays(fam_pl)[1] if len(fam_pl)!=0 else np.zeros(0, dtype=np.int32).reshape((1,-1), order='F')
      fam_pl = np.unique(fam_pl, axis=1)
      if fam_pl.size!=0:
        zsr_n = PT.new_ZoneSubRegion(name=family_name, point_list=fam_pl, loc=location[0], parent=part_zone)

  # Synchronize container names
  for node_path, there_is in there_is_bcdataset.items():
    if transfer_dataset and comm.allreduce(there_is, MPI.LOR):
      node_name = node_path.split('/')[-1]
      if node_name not in l_containers_name:
        l_containers_name.append(node_name) # not to change the initial containers_name list

  return extract_part_from_zsr(local_part_tree, family_name, comm, 
                               transfer_dataset=False,
                               containers_name=l_containers_name,
                             **options)
