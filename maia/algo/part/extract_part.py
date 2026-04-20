import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT
import maia.utils.logging as mlog
from   maia.factory       import dist_from_part
from   maia.utils         import np_utils, par_utils
from   maia.utils         import vstride as vs
from   .extract_part_s    import exchange_field_s, extract_part_one_domain_s
from   .extract_part_u    import exchange_field_u, extract_part_one_domain_u
from   .extraction_utils  import LOC_TO_DIM
from   .utils             import _gather_containers_name
from   maia.typing        import *

import numpy as np

import Pypdm.Pypdm as PDM

def get_stats(extract_tree: CGNSTree, dim: int,
              comm: MPIComm) -> Tuple[str, int, int]:
    elts_kind = ['vtx', 'edges', 'faces', 'cells'][dim]
    if dim == 0:
      n_cell = sum([PT.Zone.n_vtx(zone) for zone in PT.iter_all_Zone_t(extract_tree)])
    else:
      n_cell = sum([PT.Zone.n_cell(zone) for zone in PT.iter_all_Zone_t(extract_tree)])
    n_cell_all = comm.allreduce(n_cell, MPI.SUM)
    return elts_kind, n_cell, n_cell_all


def set_transfer_dataset(bc_n: CGNSTree,zsr_bc_n: CGNSTree,
                         zone_type: str) -> bool:

  if zone_type=='Structured':
    unwanted_type = 'IndexArray_t'
    unwanted_name = 'PointList'
    required_name = 'PointRange'
  else:
    unwanted_type = 'IndexRange_t'
    unwanted_name = 'PointRange'
    required_name = 'PointList'
  there_is_dataset = False
  assert PT.get_child_from_predicates(bc_n, f'BCDataSet_t/{unwanted_type}') is None,\
                 f'BCDataSet_t with {unwanted_name} aren\'t managed'

  is_valid_bcds = PT.pred.label_is('BCDataSet_t') & ~PT.pred.has_child_of_name(required_name)
  ds_arrays = PT.get_children_from_predicates(bc_n, [is_valid_bcds, 'BCData_t', 'DataArray_t'])
  for ds_array in ds_arrays:
    PT.new_DataArray(name=PT.get_name(ds_array), value=PT.get_np_value(ds_array), parent=zsr_bc_n)
  if len(ds_arrays) != 0:
    there_is_dataset = True
    # PL and Location is needed for data exchange, but this should be done in ZSR func
    for name in [required_name, 'GridLocation']:
      PT.add_child(zsr_bc_n, PT.get_child_from_name(bc_n, name))
  return there_is_dataset


class Extractor:
  def __init__(self, part_tree:CGNSPartTree, patch:List[NDArray],
               location:str, comm: MPIComm,
               equilibrate:bool=True,
               graph_part_tool:str="hilbert") -> None:
    """Initialize an extractor object to perform extraction of a part of a mesh"""
    self.part_tree     = part_tree
    self.exch_tool_box = dict()
    self.comm          = comm

    # Get zones by domains
    part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)
    # Check : monodomain
    assert len(part_tree_per_dom.values()) == 1
    dom_path, part_zones = list(part_tree_per_dom.items())[0]

    # > Check if U or S (working because monodomain)
    zone_type = PT.get_node_from_name(part_tree, 'ZoneType')
    is_struct = PT.get_value(zone_type)=='Structured' if zone_type is not None else False
    self.is_struct = comm.allreduce(is_struct)

    # Get dim of input tree (monobase + all ranks should know the base)
    celldim, phydim = PT.get_np_value(PT.find_child_from_label(part_tree, 'CGNSBase_t'))

    # ExtractPart dimension
    self.location = location
    if self.location == '': # Early return if the extraction is totally empty
      self.extract_tree = PT.new_CGNSTree()
      self.tgt_dim = None
      return

    assert 2 <= celldim, f"Extraction from {celldim}d mesh is not implemented"
    self.src_dim = celldim
    self.tgt_dim = LOC_TO_DIM[celldim][location]
    self.dims = (self.src_dim, self.tgt_dim)
    if celldim == 3:
      assert self.tgt_dim != 1, f"Extraction from 3d mesh to 1d mesh is not implemented"
    #CGNS does not support 0D, so keep input dim in this case (which is 3 since 2d is not managed)
    if location == 'Vertex':
      if self.is_struct:
        cell_dim = -1
        for part_pr in patch:
          if part_pr.size!=0:
            size_per_dim = np.diff(part_pr)[:,0]
            idx = np.where(size_per_dim!=0)[0]
            cell_dim = idx.size
        cell_dim = comm.allreduce(cell_dim, op=MPI.MAX)
      else:
        cell_dim = celldim
    else:
      cell_dim = self.tgt_dim

    assert graph_part_tool in ["hilbert","parmetis","ptscotch"]
    assert not( (self.tgt_dim==0) and graph_part_tool in ['parmetis', 'ptscotch']),\
           '[MAIA] Vertex extraction not available with parmetis or ptscotch partitioning. Please check your script.'

    # ExtractPart CGNSTree
    base_name = next(iter(part_tree_per_dom.keys())).split('/')[0] #Only one base
    extract_tree = PT.new_CGNSTree()
    extract_base = PT.new_CGNSBase(base_name, cell_dim=cell_dim, phy_dim=phydim, parent=extract_tree)
    # Compute extract part of each domain
    if self.is_struct:
      extract_zones, etb = extract_part_one_domain_s(part_zones, patch, self.dims, self.location, comm)
    else:
      extract_zones, etb = extract_part_one_domain_u(part_zones, patch, self.dims, comm,
                                                     equilibrate=equilibrate,
                                                     graph_part_tool=graph_part_tool)
    etb['ExtractingCnt'] = None
    self.exch_tool_box[dom_path] = etb
    for extract_zone in extract_zones:
      if PT.Zone.n_vtx(extract_zone)!=0:
        if phydim == 2:
          PT.rm_node_from_path(extract_zone, 'GridCoordinates/CoordinateZ')
        PT.add_child(extract_base, extract_zone)

    # > Clean orphan GC
    if self.is_struct:
      all_zone_name_l = [PT.get_name(n) for n in PT.iter_all_Zone_t(extract_base)]
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

    # Copy Families existing on extracted tree
    is_family_name = PT.pred.label_in(['FamilyName_t', 'AdditionalFamilyName_t'])
    found_family_name = set([PT.get_str_value(n) for n in PT.get_nodes_from_predicate(extract_tree, is_family_name)])
    for family_name in sorted(found_family_name):
      fam_node = PT.get_node_from_name_and_label(part_tree, family_name, 'Family_t', depth=2)
      if fam_node is not None:
        PT.add_child(extract_base, PT.deep_copy(fam_node))
    self.extract_tree = extract_tree

  def all_containers(self) -> List[str]:
    LOCS = ['Vertex', 'FaceCenter', 'CellCenter']
    IS_CNT = PT.pred.label_in(['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t'])

    cnts_per_zone = list()
    for zone in PT.get_all_Zone_t(self.part_tree):
      predicate = IS_CNT \
                & PT.pred.has_child_of_label('DataArray_t') \
                & PT.pred.NodePredicate(lambda c : LOCS.index(PT.Container.GridLocation(c, zone)) <= LOCS.index(self.location))
      cnts_per_zone.append(PT.get_children_from_predicate(zone, predicate))
    return _gather_containers_name(cnts_per_zone, 'any', self.comm)

  def exchange_fields(self, fs_container: List[str]) -> None:
    """Exchange fields between partitions"""
    if self.location == '': # Nothing to do if extract_tree is None
      return
    exchange_fld_func = exchange_field_s if self.is_struct else exchange_field_u
    exchange_fld_func(self.part_tree, self.extract_tree, self.dims, self.exch_tool_box,\
          fs_container, self.comm)

  def get_extract_part_tree(self) -> CGNSPartTree:
    """Return the extracted part tree"""
    return self.extract_tree


def _extract_part_from_zsr(part_tree: CGNSPartTree,
                           zsr_name: str,
                           comm: MPIComm,
                           transfer_dataset: bool = True,
                           containers_name: Union[List[str], Literal['ALL']] = [],
                           **options: Any) -> Tuple[CGNSPartTree, Optional[int]]:
  """Internal function to extract part from ZoneSubRegion"""
  extractor = _create_extractor_from_zsr(part_tree, zsr_name, comm, **options)

  if containers_name == 'ALL':
    l_containers_name = [name for name in extractor.all_containers() if name != zsr_name]
  else:
    l_containers_name = [name for name in containers_name]
  if transfer_dataset:
    if zsr_name not in l_containers_name:
      l_containers_name += [zsr_name]
  if l_containers_name:
    extractor.exchange_fields(l_containers_name)

  extract_tree = extractor.get_extract_part_tree()

  return extract_tree, extractor.tgt_dim


def extract_part_from_zsr(part_tree: CGNSPartTree,
                          zsr_name: str,
                          comm: MPIComm,
                          transfer_dataset: bool = True,
                          containers_name: Union[List[str], Literal['ALL']] = [],
                          **options) -> CGNSPartTree:
  """Extract the submesh defined by the provided ZoneSubRegion from the input volumic
  partitioned tree.

  Dimension of the output mesh is set up accordingly to the GridLocation of the ZoneSubRegion.
  Submesh is returned as an independant partitioned CGNSTree and includes the relevant connectivities.

  Data fields existing in the volumic mesh can be transfered to the extracted mesh by two ways:

  - if ``transfer_dataset`` is set to ``True``, fields found under the ZoneSubRegion are transfered on the
    extracted mesh, where they are stored in a FlowSolution_t container since they cover all cells (or vertices). 
  - Other full or partial containers are transfered if their name is requested in the ``containers_name`` list. 
    Their dimension must be at most equal to the one of the extracted mesh.

  Args:
    part_tree       (CGNSPartTree): Partitioned tree from which extraction is computed. U-Elts
      connectivities are *not* managed.
    zsr_name        (str)         : Name of the ZoneSubRegion_t node
    comm            (MPIComm)     : MPI communicator
    transfer_dataset(bool)        : Transfer (or not) fields stored in ZSR to the extracted mesh (default to ``True``)
    containers_name (list of str or ``'ALL'``) : Name of each container node to transfer
      on the output extracted tree.
    **options: Options related to the extraction.
  Returns:
    CGNSTree: Extracted submesh (partitioned)

  Extraction can be controled by the optional kwargs (only for U meshes):

    - ``equilibrate`` (bool) -- If ``False``, the extracted entities remains on their original rank,
      which simplifies data exchanges but leads to poor load balancing. Default is ``True``.
    - ``graph_part_tool`` (str) -- Partitioning tool used to balance the extracted zones (if ``equilibrate=True``)
      Admissible values are ``hilbert, parmetis, ptscotch``. Note that
      vertex-located extractions require hilbert partitioning. Default is ``hilbert``.

  Important:
    - Input tree must have a U-NGon or Structured connectivity
    - Partitions must come from a single initial domain on input tree.

  See also:
    :func:`create_extractor_from_zsr` takes the same parameters, excepted ``containers_name`` and ``transfer_dataset``,
    and returns an Extractor object which can be used to exchange containers more than once through its
    ``Extractor.exchange_fields(container_name)`` method.

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_zsr@start
      :end-before:  #extract_from_zsr@end
      :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  start = time.time()
  extract_tree, dim = _extract_part_from_zsr(part_tree, zsr_name, comm,
                                             transfer_dataset=transfer_dataset,
                                             containers_name=containers_name, **options)
  end = time.time()

  # > Print some light stats
  if dim is not None:
    elts_kind, n_cell, n_cell_all = get_stats(extract_tree, dim, comm)
    mlog.info(f"Extraction from ZoneSubRegion \"{zsr_name}\" completed ({end-start:.2f} s) -- "
              f"Extracted tree has locally {mlog.size_to_str(n_cell)} {elts_kind} "
              f"(Σ={mlog.size_to_str(n_cell_all)})")
  else:
    mlog.warning(f"ZoneSubRegion \"{zsr_name}\" does not exist in input tree, "
                 f"an empty extracted tree is returned from extract_part_from_zsr")

  return extract_tree


def _create_extractor_from_zsr(part_tree: CGNSPartTree,
                               zsr_path: str,
                               comm: MPIComm,
                               **options) -> Extractor:
  """Create an extractor object from a ZoneSubRegion path"""
  # Get zones by domains
  if options.get("equilibrate", True) == False:
    if 'graph_part_tool' in options:
      mlog.warning("extract_part: option `graph_part_tool` is ignored when `equilibrate` is False")

  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)
  assert len(part_tree_per_dom.values()) == 1
  dist_zone_path, part_zones = list(part_tree_per_dom.items())[0]

  # Get patch for each partitioned zone
  patch = list()
  location = ''
  region_names = []
  for part_zone in part_zones:
    zsr_node = PT.get_node_from_path(part_zone, zsr_path)
    if zsr_node is not None:
      if PT.get_child_from_name(zsr_node, 'BCRegionName'):
        region_names.append(PT.get_value(PT.find_child_from_name(zsr_node, 'BCRegionName')))
      if PT.get_child_from_name(zsr_node, '__maia::RegionNames'):
        for region_name in PT.get_value(PT.find_child_from_name(zsr_node, '__maia::RegionNames')).split("\n"):
          region_names.append(region_name)
      #Follow BC or GC link
      zsr_node = PT.Container.SubsetNode(zsr_node, part_zone)
      patch.append(PT.get_np_value(PT.Subset.getPatch(zsr_node)))
      location = PT.Subset.GridLocation(zsr_node)
    else: # ZSR does not exist on this partition
      patch.append(np.empty((1,0), np.int32))

  # Get location and bcregionname if proc has no zsr
  location = comm.allreduce(location, op=MPI.MAX)
  region_names = list(set(comm.allreduce(region_names)))

  if location == 'FaceCenter':
    dist_zone_name = dist_zone_path.split('/')[1]
    fake_dist_zone = PT.new_Zone(dist_zone_name)
    fake_dist_zbc = PT.new_ZoneBC(parent=fake_dist_zone)
    bc_predicate = ['ZoneBC_t', 'BC_t']
    child_list = ['FamilyName_t', 'GridLocation_t', 'Ordinal_t', 'AdditionalFamilyName_t', 'Descriptor_t']
    dist_from_part.discover_nodes_from_matching(fake_dist_zone, part_zones, bc_predicate, comm,
                                  child_list=child_list, get_value='all')
    all_dist_bcs = [bc for bc in PT.get_nodes_from_predicates(fake_dist_zone, 'ZoneBC_t/BC_t')]

    for part_zone in part_zones:
      n_vtx = PT.Zone.n_vtx(part_zone)
      ngon = PT.Zone.NGonNode(part_zone)
      face_vtx = MT.Element.connectivity(ngon)
      fs_vtx_bc = PT.new_FlowSolution('__maia::TagBCsOnVtx', parent=part_zone)
      for d_bc in all_dist_bcs:
        bc_name = PT.get_name(d_bc)
        bc_tag = np.zeros(n_vtx)
        bc = PT.get_node_from_name(part_zone, bc_name)
        if bc is not None and bc_name not in region_names:
          pl = PT.get_value(PT.get_node_from_name(bc, 'PointList'))[0]
          vtx_ids = np.unique(np.concatenate(vs.take(face_vtx, pl-1)))
          bc_tag = np.zeros(n_vtx)
          bc_tag[vtx_ids-1] = 1
        if bc_name not in region_names:
          da = PT.new_DataArray(bc_name, value=bc_tag, parent=fs_vtx_bc)
          PT.new_Descriptor('BCValue', PT.get_value(d_bc), parent=da)
          fm = PT.get_child_from_name(d_bc, 'FamilyName')
          if fm is not None:
            PT.new_Descriptor('Family', PT.get_value(fm), parent=da)

  extractor = Extractor(part_tree, patch, location, comm, **options)

  if location == 'FaceCenter':

    extractor.exchange_fields(['__maia::TagBCsOnVtx'])

    extract_tree = extractor.get_extract_part_tree()
  
    for extract_base in PT.get_all_CGNSBase_t(extract_tree):
      if PT.get_value(extract_base)[0] == 2:
        maia.algo.ngon_to_edge_pe(extract_tree, comm)
        PT.rm_children_from_label(extract_base, 'Family_t')
        families = []
        for extract_zone in PT.get_children_from_label(extract_base, 'Zone_t'):
          extract_zbc = PT.get_node_from_name(extract_zone, 'ZoneBC')
          if extract_zbc is not None:
            ngon_elts = PT.Zone.NGonNode(extract_zone)
            ngon_er   = PT.get_value(PT.get_child_from_name(ngon_elts, 'ElementRange'))
            orig_bc_ids = -np.ones((PT.Zone.n_cell(extract_zone)), dtype=np.int32) # -1 because the extracted ZoneSubRegion can contain sone faces that aren't BC
            orig_bc_names = sorted([PT.get_name(d_bc) for d_bc in all_dist_bcs]) #Sort is needed to have // independant bc ids
            for ibc, bc_name in enumerate(orig_bc_names):
              bc = PT.get_child_from_name(extract_zbc, bc_name)
              if bc is not None:
                orig_bc_ids[PT.get_child_from_name(bc, 'PointList')[1].reshape(-1)-ngon_er[0]] = ibc
            dd = PT.new_DiscreteData('Original3DBC', loc='CellCenter', fields={'OriginalBCIds': orig_bc_ids}, parent=extract_zone)
            PT.new_Descriptor('OriginalBCNames', "\n".join(orig_bc_names), parent=dd)
            PT.rm_children_from_label(extract_zbc, 'BC_t')
          pfs_vtx_bc = PT.get_node_from_name(extract_zone, '__maia::TagBCsOnVtx')
          if pfs_vtx_bc is None: continue
          coords = PT.get_node_from_name(extract_zone, 'GridCoordinates')
          edge_elts = MT.Zone.EdgeNode(extract_zone)
          edge_co = PT.get_value(PT.get_child_from_name(edge_elts, 'ElementConnectivity'))
          edge_pe = PT.get_value(PT.get_child_from_name(edge_elts, 'ParentElements'))
          ext_edges = np.where(edge_pe[:,1]==0)[0]
          edge_gnum = PT.get_np_value(MT.get_GlobalNumbering(edge_elts, 'Element'))
          extract_zgc = PT.get_child_from_label(extract_zone, 'ZoneGridConnectivity_t')
          gc_edges = np.empty(0, dtype=PT.get_value(extract_zone).dtype)
          if extract_zgc is not None:
            GC_PRED = PT.pred.is_gc_of_kind(is_1to1=True) & PT.pred.has_location('EdgeCenter')
            for gc in PT.iter_children_from_predicate(extract_zgc, GC_PRED):
              gc_edges = np.concatenate([gc_edges,PT.get_value(PT.get_child_from_name(gc, 'PointList'))[0]])
          bc_edges = np.setdiff1d(ext_edges, gc_edges-1)
          bc_edges_n1 = edge_co[2*bc_edges]
          bc_edges_n2 = edge_co[2*bc_edges+1]
          for da in PT.get_children_from_label(pfs_vtx_bc, 'DataArray_t'):
            is_bc_n1 = PT.get_value(da)[bc_edges_n1-1]
            is_bc_n2 = PT.get_value(da)[bc_edges_n2-1]
            bc_indices = np.where((is_bc_n1==1)&(is_bc_n2==1))[0]
            if len(bc_indices)>0:
              fm = PT.get_child_from_name(da, 'Family')
              if fm is not None:
                family_name = PT.get_value(fm)
                families.append(family_name)
                extracted_bc = PT.new_BC(PT.get_name(da), 'FamilySpecified', loc="EdgeCenter", point_list=[bc_edges[bc_indices]+1], family=family_name, parent=extract_zbc)
              else:
                extracted_bc = PT.new_BC(PT.get_name(da), PT.get_value(PT.get_child_from_name(da, 'BCValue')), loc="EdgeCenter", point_list=[bc_edges[bc_indices]+1], parent=extract_zbc)
              MT.new_GlobalNumbering({'Index':edge_gnum[bc_edges[bc_indices]]}, parent=extracted_bc)
          PT.rm_children_from_name(extract_zone, '__maia::TagBCsOnVtx')
        families = comm.allgather(list(set(families)))
        for family_name in list(set(sum(families, []))):
          PT.add_child(extract_base, PT.get_node_from_predicates(part_tree, f'CGNSBase_t/{family_name}'))
    PT.rm_nodes_from_name(part_tree, '__maia::TagBCsOnVtx')
    for extract_base in PT.get_all_CGNSBase_t(extract_tree):
      if PT.get_value(extract_base)[0] == 2:
        bc_names = [PT.get_name(bc) for bc in all_dist_bcs]
        for bc_name in bc_names:
          bc_gnum_l = []
          bc_l = PT.get_nodes_from_predicates(extract_base, f'Zone_t/ZoneBC_t/{bc_name}')
          for bc in bc_l:
            bc_gnum_n = MT.get_GlobalNumbering(bc, 'Index')
            bc_gnum_l.append(PT.get_value(bc_gnum_n))
          new_gn = maia.algo.part.point_cloud_utils.create_sub_numbering(bc_gnum_l, comm)
          for b, bc in enumerate(bc_l):
            bc_gnum_n = MT.get_GlobalNumbering(bc, 'Index')
            PT.set_value(bc_gnum_n, new_gn[b])

  # This will be usefull to detect self data exchange later
  for subdict in extractor.exch_tool_box.values():
    subdict['ExtractingCnt'] = zsr_path
  return extractor

def create_extractor_from_zsr(part_tree: CGNSPartTree,
                              zsr_path : str,
                              comm: MPIComm,
                              **options) -> Extractor:
  """Same as extract_part_from_zsr, but return the extractor object."""
  # Get zones by domains
  MT.check_cgns_part_tree(part_tree)
  extractor = _create_extractor_from_zsr(part_tree, zsr_path, comm, **options)
  if extractor.location == '':
    mlog.warning(f"ZoneSubRegion \"{zsr_path}\" does not exist in input tree, "
                 f"an empty extractor is returned from create_extractor_from_zsr")
  return extractor

def extract_part_from_bc_name(part_tree: CGNSPartTree,
                              bc_name: str,
                              comm: MPIComm,
                              transfer_dataset: bool = True,
                              containers_name: Union[List[str], Literal['ALL']] = [],
                              **options) -> CGNSPartTree:
  """Extract the submesh defined by the provided BC name from the input volumic
  partitioned tree.

  Behaviour and arguments of this function are similar to those of :func:`extract_part_from_zsr`:
  ``zsr_name`` becomes ``bc_name`` and optional ``transfer_dataset`` argument allows to
  transfer BCDataSet (without PointList or PointRange) from BC to the extracted mesh (default to ``True``).

  See also:
    :func:`create_extractor_from_bc_name` takes the same parameters, excepted ``containers_name`` and ``transfer_dataset``,
    and returns an Extractor object which can be used to exchange containers more than once through its
    ``Extractor.exchange_fields(container_name)`` method.

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_bc_name@start
      :end-before:  #extract_from_bc_name@end
      :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  start = time.time()

  # Local copy of the part_tree to add ZSR
  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)

  # Adding ZSR to tree
  there_is_bcdataset = False
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:
      bc_n = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t')
      if bc_n is not None:
        zsr_bc_n  = PT.new_ZoneSubRegion(name=f'__{bc_name}', bc_name=bc_name, parent=part_zone)
        if transfer_dataset:
          there_is_bcdataset = set_transfer_dataset(bc_n, zsr_bc_n, PT.Zone.Type(part_zone))

  _transfer_dataset = False
  if transfer_dataset and comm.allreduce(there_is_bcdataset, MPI.LOR):
    _transfer_dataset = True


  extract_tree, dim = _extract_part_from_zsr(local_part_tree, f'__{bc_name}', comm,
                                             transfer_dataset=_transfer_dataset,
                                             containers_name=containers_name,
                                           **options)
  # Rename native container
  if transfer_dataset:
    for ext_zone in PT.get_all_Zone_t(extract_tree):
      cnt = PT.get_child_from_name(ext_zone, f'__{bc_name}')
      if cnt is not None:
        PT.update_node(cnt, name=bc_name)

  end = time.time()

  # > Print some light stats
  if dim is not None:
    elts_kind, n_cell, n_cell_all = get_stats(extract_tree, dim, comm)
    mlog.info(f"Extraction from BC \"{bc_name}\" completed ({end-start:.2f} s) -- "
              f"Extracted tree has locally {mlog.size_to_str(n_cell)} {elts_kind} "
              f"(Σ={mlog.size_to_str(n_cell_all)})")
  else:
    mlog.warning(f"BC \"{bc_name}\" does not exist in input tree, "
                 f"an empty extracted tree is returned from extract_part_from_bc_name")

  return extract_tree

def create_extractor_from_bc_name(part_tree: CGNSPartTree, bc_name: str,
                                  comm: MPIComm,**options) -> Extractor:
  """Create an extractor object from a BC name"""
  MT.check_cgns_part_tree(part_tree)
  # Local copy of the part_tree to add ZSR
  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)

  # Adding ZSR to tree
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:
      bc_n = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t')
      if bc_n is not None:
        PT.new_ZoneSubRegion(name=f'__{bc_name}', bc_name=bc_name, parent=part_zone)

  extractor = _create_extractor_from_zsr(local_part_tree, f'__{bc_name}', comm, **options)
  if extractor.location == '':
    mlog.warning(f"BC \"{bc_name}\" does not exist in input tree, "
                 f"an empty extractor is returned from create_extractor_from_bc_name")
  return extractor


def _prepare_extract_from_family(part_tree: CGNSPartTree, family_name: str,
                                 comm: MPIComm) -> Tuple[CGNSPartTree, List[CGNSPath]]:
  """Internal function to prepare extraction from a family name"""

  has_struct_zone = any(PT.Zone.Type(zone) == 'Structured' for zone in PT.get_all_Zone_t(part_tree))
  if comm.allreduce(has_struct_zone, MPI.LOR):
    raise RuntimeError(f'extract_part_from_family function is not implemented for Structured meshes.')

  # Local copy of the part_tree to add ZSR
  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)

  # > Discover family related nodes
  in_fam = PT.pred.belongs_to_family(family_name)
  is_regionname = PT.pred.name_in(['BCRegionName', 'GridConnectivityRegionName'])
  zsr_has_regionname = PT.pred.label_is('ZoneSubRegion_t') \
                     & (PT.pred.has_child_of_name('BCRegionName') \
                        | PT.pred.has_child_of_name('GridConnectivityRegionName'))


  fam_node_paths = list()
  region_names_by_domain = dict()
  for domain, part_zones in part_tree_per_dom.items():
    # Create a "fake" dist zone including:
    #   - ZSR belonging to provided family,
    #   - BC  belonging to the provided family OR referenced by a previoulsy found ZSR
    #   - GC  referenced by a previously found ZSR
    dist_zone = PT.new_Zone('Zone')
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, [PT.pred.label_is('ZoneSubRegion_t') & in_fam],
                                                comm, get_value='leaf', child_list=['FamilyName_t', 'GridLocation_t', 'Descriptor_t'])
    region_node_names:List[str] = list()
    for zsr_with_regionname_n in PT.get_children_from_predicate(dist_zone, zsr_has_regionname):
      region_node = PT.find_child_from_predicate(zsr_with_regionname_n, is_regionname)
      region_node_names.append(PT.get_str_value(region_node))
    region_names_by_domain[domain] = region_node_names
    child_list = ['AdditionalFamilyName_t', 'FamilyName_t', 'GridLocation_t']
    bc_gc_in_fam = PT.pred.name_in(region_node_names)
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ['ZoneBC_t', in_fam | bc_gc_in_fam], comm, get_value='leaf', child_list=child_list)
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ['ZoneGridConnectivity_t', bc_gc_in_fam], comm, get_value='leaf', child_list=child_list)

    # Add selected ZSR and BCs to fam_node_paths
    fam_node_paths.extend(PT.predicates_to_paths(dist_zone, [PT.pred.label_is("ZoneSubRegion_t")]))
    fam_node_paths.extend(PT.predicates_to_paths(dist_zone, ['ZoneBC_t', in_fam]))
    region_names_by_domain[domain].extend([path.split('/')[-1] for path in PT.predicates_to_paths(dist_zone, ['ZoneBC_t', in_fam])])

    gl_nodes = PT.get_nodes_from_label(dist_zone, 'GridLocation_t')
    location = [PT.get_str_value(n) for n in gl_nodes]
    if len(set(location)) > 1:
      # Not checking subregion extents, possible ?
      raise ValueError(f"Specified family refers to nodes with different GridLocation value : {set(location)}.")

  # Adding ZSR to tree
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:

      fam_pl = list()
      for path in fam_node_paths:
        fam_node = PT.get_node_from_path(part_zone, path)
        if fam_node is not None:

          if PT.get_label(fam_node)=="ZoneSubRegion_t":
            fam_node = PT.Container.SubsetNode(fam_node, part_zone)

          pl_n = PT.find_child_from_name(fam_node, 'PointList')
          fam_pl.append(PT.get_np_value(pl_n))

      fam_pl_cat = np_utils.concatenate_np_arrays(fam_pl)[1] if len(fam_pl)!=0 else np.zeros(0, dtype=np.int32).reshape((1,-1), order='F')
      if fam_pl_cat.size!=0:
        fam_pl_cat, pl_idx = np.unique(fam_pl_cat, return_index=True, axis=1) # If pl.size == 0, this line fails with numpy 1.17
        fake_zsr = PT.new_ZoneSubRegion(name=f"__{family_name}", point_list=fam_pl_cat, loc=location[0], parent=part_zone)
        # Store the index array to reorder associated data the same way
        PT.new_UserDefinedData("_UniqueIdx", pl_idx, parent=fake_zsr)
        # Store region names
        if region_names_by_domain[domain]:
          PT.new_Descriptor("__maia::RegionNames", "\n".join(region_names_by_domain[domain]), parent=fake_zsr)
  return local_part_tree, fam_node_paths

def _prepare_extract_from_family_fields(local_part_tree, family_name, fam_node_paths, containers_name, comm):
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)
  # First pass : collect fields name + values in nodes referenced by the input Family
  fields_per_part = list()
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:
      for i,path in enumerate(fam_node_paths):
        fam_node = PT.get_node_from_path(part_zone, path)
        if fam_node is not None:
          if PT.get_label(fam_node) == "ZoneSubRegion_t":
            fields_per_part.append({PT.get_name(n) : PT.get_np_value(n) \
                                    for n in PT.get_children_from_label(fam_node, 'DataArray_t')})
          elif PT.get_label(fam_node) == 'BC_t':
            bcname = PT.utils.path_tail(path)
            zrs_from_bc = PT.new_ZoneSubRegion(bcname, bc_name=bcname, parent=part_zone)
            set_transfer_dataset(fam_node, zrs_from_bc, PT.Zone.Type(part_zone))
            fields_per_part.append({PT.get_name(n) : PT.get_np_value(n) \
                                    for n in PT.get_children_from_label(zrs_from_bc, 'DataArray_t')})

  # Update fam_node_path to indicate created ZSRs
  for i,path in enumerate(fam_node_paths):
    if len(split := path.split('/')) > 1:
      fam_node_paths[i] = split[1]


  # Filter names to keep only mergeable arrays, ie appearing on all subsets
  field_names = [set(fields.keys()) for fields in fields_per_part]
  glo_cnt = par_utils.sets_intersection(field_names, comm)
  full_fields = sorted(glo_cnt) if glo_cnt is not None else []

  is_empty_l = np.ones(len(fam_node_paths), bool)
  is_empty_g = np.empty(len(fam_node_paths), bool)
  # Concatenate full arrays and store them in tmp ZSR for extraction
  _fields_per_part = iter(fields_per_part)
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:
      fake_zsr = PT.get_child_from_name(part_zone, f'__{family_name}')
      if fake_zsr is not None: # Cat fields
        order = PT.get_np_value(PT.find_child_from_name(fake_zsr, '_UniqueIdx'))
        gathered_fields = {key: [] for key in full_fields}
        for i,path in enumerate(fam_node_paths):
          if (cnt:=PT.get_node_from_path(part_zone, path)) is not None:
            tt = next(_fields_per_part) # Consume stored value
            for field in full_fields:
              gathered_fields[field].append(tt[field])
              PT.rm_children_from_name(cnt, field) # Remove to avoid double exchange
            is_empty_l[i] &= (len(PT.get_children_from_label(cnt, 'DataArray_t')) == 0)
        for fname, fields in gathered_fields.items():
          PT.new_DataArray(fname, np_utils.concatenate_np_arrays(fields)[1][order], parent=fake_zsr)

  transfer_dataset = len(full_fields) > 0
  # Add fam_node_paths in containers_name to have partial exchange on other fields
  # (filtering empty container)
  if containers_name == 'ALL':
    # Case 1 - ALL : we can just propagate ALL to get initial containers + created ones (BCDS)
    # Filtering will be performed by all_containers()
    _containers_name = 'ALL'
  else:
    # Case 2 - list : complete with non empty created containers
    comm.Allreduce(is_empty_l, is_empty_g, MPI.LAND)
    _containers_name = [c for c in containers_name]
    for i,name in enumerate(fam_node_paths):
      if not is_empty_g[i] and name not in _containers_name:
        _containers_name.append(name)

  return transfer_dataset, _containers_name


def extract_part_from_family(part_tree: CGNSPartTree,
                             family_name: str,
                             comm: MPIComm,
                             transfer_dataset: bool = True,
                             containers_name: Union[List[str], Literal['ALL']] = [],
                             **options) -> CGNSPartTree:
  """Extract the submesh defined by the provided family name from the input volumic
  partitioned tree.

  Family related nodes can be labelled either as BC_t or ZoneSubRegion_t, but their
  GridLocation must have the same value. They generate a merged output on the resulting extracted tree.

  Behaviour and arguments of this function are similar to those of :func:`extract_part_from_zsr`.

  Warning:
    Only U-NGon meshes are managed in this function.
  See also:
    :func:`create_extractor_from_family` takes the same parameters, excepted ``containers_name`` and ``transfer_dataset``,
    and returns an Extractor object which can be used to exchange containers more than once through its
    ``Extractor.exchange_fields(container_name)`` method.

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_family@start
      :end-before:  #extract_from_family@end
      :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  start = time.time()

  # Search and concat requested PLs to create extracting family on local_part_tree
  local_part_tree, fam_node_paths = _prepare_extract_from_family(part_tree, family_name, comm)
  if transfer_dataset:
    # If transfer_dataaset, also search associated fields; concat them in 
    # local extracting family *or* update containers_name if partial fields
    transfer_dataset, _containers_name = _prepare_extract_from_family_fields(local_part_tree, 
                                                                             family_name, 
                                                                             fam_node_paths, 
                                                                             containers_name,
                                                                             comm)
  else:
    _containers_name = containers_name # Nothing to do : if 'ALL', search is delegated to _extract_part_from_zsr

  extract_tree, dim = _extract_part_from_zsr(local_part_tree, f"__{family_name}", comm, 
                                             transfer_dataset, _containers_name, **options)
  # Rename native container
  if transfer_dataset:
    for ext_zone in PT.get_all_Zone_t(extract_tree):
      cnt = PT.get_child_from_name(ext_zone, f'__{family_name}')
      if cnt is not None:
        PT.update_node(cnt, name=family_name)

  end = time.time()


  # > Print some light stats
  if dim is not None:
    elts_kind, n_cell, n_cell_all = get_stats(extract_tree, dim, comm)
    mlog.info(f"Extraction from Family \"{family_name}\" completed ({end-start:.2f} s) -- "
              f"Extracted tree has locally {mlog.size_to_str(n_cell)} {elts_kind} "
              f"(Σ={mlog.size_to_str(n_cell_all)})")
  else:
    mlog.warning(f"Family \"{family_name}\" does not exist in input tree, "
                 f"an empty extracted tree is returned from extract_part_from_family")

  return extract_tree



def create_extractor_from_family(part_tree: CGNSPartTree, family_name: str,
                                 comm: MPIComm, **options) -> Extractor:
  """Create an extractor object from a family name"""
  MT.check_cgns_part_tree(part_tree)
  local_part_tree, _ = _prepare_extract_from_family(part_tree, family_name, comm)

  extractor = _create_extractor_from_zsr(local_part_tree, f"__{family_name}", comm, **options)
  if extractor.location == '':
    mlog.warning(f"Family \"{family_name}\" does not exist in input tree, "
                 f"an empty extractor is returned from create_extractor_from_family")
  return extractor
