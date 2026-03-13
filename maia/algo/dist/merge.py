import numpy as np
from re import sub
from Pypdm import Pypdm as PDM

from maia.typing import *
from maia.pytree.typing import Predicates
import maia.pytree        as PT
import maia.pytree.pred   as PTp
import maia.pytree.maia   as MT

import maia

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import np_utils, par_utils, as_pdm_gnum, logging
from maia.utils import vstride as vs

from maia.algo.dist import matching_jns_tools as MJT
from maia.algo.dist import concat_nodes as GN
from maia.algo.dist import vertex_list as VL
from maia.transfer  import protocols as EP

from .ngon_tools import cgns_connectivity_from_vs

HAS_POINTLIST = PTp.has_child_of_name('PointList')

def _append_or_create(d, key, val):
  try:
    d[key].append(val)
  except KeyError:
    d[key] = [val]

def camel_case(s):
  return sub(r"(_|-)+", " ", s).title().replace(" ", "")

def merge_all_zones_from_families(dist_tree: CGNSDistTree,
                                  comm: MPIComm,
                                  **kwargs) -> None:
  """Apply merge_zones_from_family to each family of the tree"""
  MT.check_cgns_dist_tree(dist_tree)
  family_names = [PT.get_name(node) for node in \
          PT.iter_nodes_from_label(dist_tree, 'Family_t', depth=2)]
  for family_name in family_names:
    merge_zones_from_family(dist_tree, family_name, comm, **kwargs)

def merge_zones_from_family(dist_tree: CGNSDistTree,
                            family_name: str,
                            comm: MPIComm,
                            **kwargs) -> None:
  """Merge the zones belonging to the given family into a single one.

  See :func:`merge_zones` for full documentation.

  Args:
    dist_tree (CGNSDistTree): Input distributed tree
    family_name (str)       : Name of the family (read from ``(Additional)FamilyName_t`` node)
        used to select the zones.
    comm (MPIComm)          : MPI communicator
    kwargs: any argument of :func:`merge_zones`, excepted output_path

  See also:
    Function ``merge_all_zones_from_families(tree, comm, **kwargs)`` does
    this operation for all the ``Family_t`` nodes of the input tree.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #merge_zones_from_family@start
        :end-before: #merge_zones_from_family@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)

  is_zone_with_fam = PTp.label_is('Zone_t') & PTp.belongs_to_family(family_name)
  zone_paths = PT.predicates_to_paths(dist_tree, ['CGNSBase_t', is_zone_with_fam])
  if zone_paths:
    base_name = zone_paths[0].split('/')[0]
    zone_name = camel_case(family_name)
    if zone_name == family_name:
      zone_name = zone_name.lower()
    merge_zones(dist_tree, zone_paths, comm, output_path=f'{base_name}/{zone_name}', **kwargs)

def merge_connected_zones(dist_tree: CGNSDistTree,
                          comm: MPIComm,
                          **kwargs) -> None:
  """Detect all the zones connected through 1to1 matching jns and merge them.

  See :func:`merge_zones` for full documentation.

  Args:
    dist_tree (CGNSDistTree): Input distributed tree
    comm (MPIComm) : MPI communicator
    kwargs: any argument of :func:`merge_zones`, excepted output_path

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #merge_connected_zones@start
        :end-before: #merge_connected_zones@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  MJT.find_joins_donor_name(dist_tree, comm)
  grouped_zone_paths = PT.Tree.find_connected_zones(dist_tree)

  for i, zone_paths in enumerate(grouped_zone_paths):
    zone_paths_u = [path for path in zone_paths \
        if PT.Zone.Type(PT.find_node_from_path(dist_tree, path)) == 'Unstructured']
    base = zone_paths[0].split('/')[0]
    merge_zones(dist_tree, zone_paths_u, comm, output_path=f'{base}/mergedZone{i}', **kwargs)

def merge_zones(dist_tree: CGNSDistTree,
                zone_paths: List[CGNSPath],
                comm: MPIComm,
                output_path: Optional[str] = None,
                subset_merge: str = 'name',
                concatenate_jns: bool = True) -> None:
  """Merge the given zones into a single one.

  Input tree is modified inplace : original zones will be removed from the tree and replaced
  by the merged zone. Merged zone is added with name *MergedZone* under the first involved Base
  except if ``output_path`` is not None : in this case, the provided path defines the base and zone name
  of the merged block.

  Subsets of the merged block can be reduced thanks to subset_merge parameter:

  - ``'none'``   : no reduction occurs : all subset of all original zones remains on merged zone, with a
    numbering suffix.
  - ``'name'`` : Subsets having the same name on the original zones (within a same label) produces
    a unique subset on the output merged zone.
  - ``'family'`` : Subsets having the same FamilyName on the original zones (within a same label) produces
    a unique subset on the output merged zone. Subsets without FamilyName fallback to ``'name'`` strategy.

  Important:
    Only unstructured polyedric trees are supported, and interfaces between the zones
    to merge must have a FaceCenter (EdgeCenter in 2D) location.

  Args:
    dist_tree (CGNSDistTree): Input distributed tree
    zone_paths (list of str): List of path (BaseName/ZoneName) of the zones to merge.
        Wildcard ``*`` are allowed in BaseName and/or ZoneName.
    comm       (MPIComm): MPI communicator
    output_path (str, optional): Path of the output merged block. Defaults to None.
    subset_merge (str, optional): Merging strategy for the subsets. Defaults to 'name'.
    concatenate_jns (bool, optional): if True, reduce the multiple 1to1 matching joins related
        to the merged_zone to a single one. Defaults to True.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #merge_zones@start
        :end-before: #merge_zones@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  # Transform wildcard into concrete path
  replace_super_wildcard = lambda p: '*/*' if p == '*' else p
  zone_paths = [replace_super_wildcard(p) for p in zone_paths]
  zone_paths = PT.utils.concretize_paths(dist_tree, zone_paths, ['CGNSBase_t', 'Zone_t'])

  assert all([PT.Zone.Type(PT.find_node_from_path(dist_tree, path)) == 'Unstructured' for path in zone_paths])
  #Those one will be needed for jn recovering
  MJT.find_joins_donor_name(dist_tree, comm)

  #Force full donor name, otherwise it is hard to reset jns
  PT.enforceDonorAsPath(dist_tree)

  # We create a tree including only the zones to merge to speed up some operations
  masked_tree = PT.new_CGNSTree()
  for zone_path in zone_paths:
    base_n, zone_n = zone_path.split('/')
    base = PT.find_child_from_name(dist_tree, base_n)
    zone = PT.find_child_from_name(base, zone_n)
    masked_base = PT.update_child(masked_tree, base_n, 'CGNSBase_t', PT.get_value(base))
    PT.add_child(masked_base, zone)

    #Remove from input tree at the same time
    PT.rm_node_from_path(dist_tree, zone_path)

  # Detect CellDim/PhyDim
  cell_dims = {PT.Base.CellDimension(b) for b in PT.iter_all_CGNSBase_t(masked_tree)}
  phy_dims  = {PT.Base.PhysicalDimension(b) for b in PT.iter_all_CGNSBase_t(masked_tree)}
  assert len(cell_dims) == 1, "Merging zone of different CellDimension is not allowed"
  assert len(phy_dims)  == 1, "Merging zone of different PhysicalDimension is not allowed"
  cell_dim = cell_dims.pop()
  phy_dim = phy_dims.pop()

  # Create NGON/ParentElements if not existing
  if cell_dim == 3:
    maia.algo.nface_to_pe(masked_tree, comm, True)
  else:
    maia.algo.ngon_to_edge_pe(masked_tree, comm, True)

  #Merge zones
  merged_zone = _merge_zones(masked_tree, comm, subset_merge)

  #Add output
  if output_path is None:
    output_base = PT.get_node_from_path(dist_tree, zone_paths[0].split('/')[0])
  else:
    output_base = PT.get_node_from_path(dist_tree, output_path.split('/')[0])
    if output_base is None:
      output_base = PT.new_CGNSBase(output_path.split('/')[0], cell_dim=cell_dim, phy_dim=phy_dim, parent=dist_tree)
    PT.set_name(merged_zone, output_path.split('/')[1])
  assert output_base is not None
  PT.add_child(output_base, merged_zone)

  #First we have to retrieve PLDonor for external jn and update opposite zones
  merged_zone_path = PT.get_name(output_base) + '/' + PT.get_name(merged_zone)
  jn_to_pl = {}
  for jn_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity_t'):
    gc = PT.find_node_from_path(dist_tree, jn_path)
    if PT.GridConnectivity.is1to1(gc):
      jn_to_pl[jn_path] = \
          (PT.find_child_from_name(gc, 'PointList')[1], PT.find_child_from_name(gc, 'PointListDonor')[1], MT.get_Distribution(gc))

  # Update opposite names when going to opp zone (intrazone have been caried before)
  for zgc, gc in PT.get_children_from_predicates(merged_zone, ['ZoneGridConnectivity_t', 'GridConnectivity_t'], ancestors=True):
    if PT.GridConnectivity.is1to1(gc) and PT.get_value(gc) not in zone_paths:
      opp_path = MJT.get_jn_donor_path(dist_tree, f"{merged_zone_path}/{zgc[0]}/{gc[0]}")
      opp_gc = PT.find_node_from_path(dist_tree, opp_path)
      opp_gc_donor_name = PT.find_child_from_name(opp_gc, 'GridConnectivityDonorName') #TODO factorize
      PT.set_value(opp_gc_donor_name, gc[0])
  #Now all donor names are OK

  for zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    is_merged_zone = zone_path == merged_zone_path
    zone = PT.find_node_from_path(dist_tree, zone_path)
    for zgc, gc in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', 'GridConnectivity_t'], ancestors=True):
      #Update name and PL
      if PT.get_value(gc) in zone_paths: #Can be: jn from non concerned zone to merged zones or periodic from merged zones
        PT.set_value(gc, merged_zone_path)
        jn_path = f"{zone_path}/{PT.get_name(zgc)}/{PT.get_name(gc)}"
        if PT.GridConnectivity.is1to1(gc):
          jn_path_opp= MJT.get_jn_donor_path(dist_tree, jn_path)
          # Copy and permute pl/pld only for all the zones != merged zone OR for one gc over two for
          # merged zone
          if not is_merged_zone or jn_path_opp < jn_path:
            PT.update_child(gc, 'PointList'     , 'IndexArray_t', jn_to_pl[jn_path_opp][1])
            PT.update_child(gc, 'PointListDonor', 'IndexArray_t', jn_to_pl[jn_path_opp][0])
            PT.rm_children_from_name(gc, ":CGNS#Distribution")
            PT.add_child(gc, jn_to_pl[jn_path_opp][2])

  if concatenate_jns:
    GN.concatenate_jns(dist_tree, comm)

  # Transfert some nodes on the merged zone, only if they exist everywhere and have same value
  merge_me = PTp.label_in(['FamilyName_t', 'AdditionalFamilyName_t'])
  if len(zone_paths) > 0:
    zone = PT.find_node_from_path(masked_tree, zone_paths[0])
    common = {(PT.get_name(n), PT.get_label(n), PT.get_value(n)) \
               for n in PT.iter_children_from_predicate(zone, merge_me)}
    for zone_path in zone_paths[1:]:
      zone = PT.find_node_from_path(masked_tree, zone_path)
      # Use set intersection to eliminate nodes that does not appear on this zone
      common = common & {(PT.get_name(n), PT.get_label(n), PT.get_value(n)) \
                         for n in PT.iter_children_from_predicate(zone, merge_me)}
    for c in sorted(common): #Sort to garantie same insertion order across mpi ranks
      PT.new_child(merged_zone, name=c[0], label=c[1], value=c[2])

  # Cleanup empty bases
  to_remove = []
  for base in PT.get_children_from_label(dist_tree, 'CGNSBase_t'):
    if len(PT.get_children_from_label(base, 'Zone_t')) == 0:
      to_remove.append(PT.get_name(base))
  for base_n in to_remove:
    PT.rm_children_from_name(dist_tree, base_n)

def _merge_zones(tree: CGNSDistTree, comm: MPIComm,
                 subset_merge_strategy: str='name') -> CGNSDistTree:
  """
  Tree must contain *only* the zones to merge. We use a tree instead of a list of zone because it's easier
  to retrieve opposites zones througt joins. Interface beetween zones shall be described by faces
  """

  zone_paths = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  n_zone = len(zone_paths)
  zones = PT.get_all_Zone_t(tree)
  assert min([PT.Zone.Type(zone) == 'Unstructured' for zone in zones]) == True
  cell_dim = PT.Zone.CellDimension(zones[0])

  if cell_dim == 3:
    expected_elt_tot = sum([PT.Zone.n_cell(z) + PT.Zone.n_face(z) for z in zones])
    _expected_eso_tot = sum(PT.get_child_from_name(PT.Zone.NGonNode(z), 'ElementStartOffset')[1][-1] for z in zones) if comm.rank == comm.size-1 else 0
    expected_eso_tot = comm.bcast(_expected_eso_tot, root=comm.size-1)
  else:
    n_edge = lambda z: MT.Element.n_elt(MT.Zone.EdgeNode(z))
    expected_elt_tot = sum([PT.Zone.n_cell(z) + n_edge(z) for z in zones])
    expected_eso_tot = 0
  output_dtype = PT.get_np_value(zones[0]).dtype
  if max(expected_elt_tot, expected_eso_tot) > np.iinfo(np.int32).max:
    if pdm_dtype == np.int32:
      msg = f"_merge_zones would overflow this I4 production of maia/ParaDiGM. "\
            f"Please try with an I8 production (using -D_PDM_ENABLE_LONG_G_NUM=ON)."
      raise OverflowError(msg)
    elif output_dtype == np.int32:
      output_dtype = np.dtype(np.int64)
      msg = f"Input meshes uses I4 integers, but result of _merge_zones would overflow it. "\
            f"Kind of output zone has thus be changed to I8."
      logging.warning(msg)

  zone_to_id = {path : i for i, path in enumerate(zone_paths)}

  loc = 'FaceCenter' if PT.Zone.CellDimension(zones[0]) == 3 else 'EdgeCenter'
  face_gc_query:Predicates = ['ZoneGridConnectivity_t', PT.pred.IS_GC & PTp.has_location(loc)]
  vtx_gc_query:Predicates  = ['ZoneGridConnectivity_t', PT.pred.IS_GC & PTp.has_location('Vertex')]

  # Move non 1to1 GC_t to ZoneBC since they have no PointListDonor
  is_not_1to1 = PTp.label_is('GridConnectivity_t') & PTp.is_gc_of_kind(is_1to1=False)
  for zone_path in zone_paths:
    zone = PT.find_node_from_path(tree, zone_path)
    for zgc in PT.get_children_from_label(zone, 'ZoneGridConnectivity_t'):
      non_abutting = PT.get_children_from_predicate(zgc, is_not_1to1)
      if len(non_abutting) > 0:
        fake_zbc = PT.new_child(zone, f'maia_{PT.get_name(zgc)}', 'ZoneBC_t')
        for gc in non_abutting:
          PT.set_label(gc, 'BC_t')
          PT.rm_child(zgc, gc)
          PT.add_child(fake_zbc, gc)

  # JNs to external zones must be excluded from vertex list computing
  tree_vl = PT.shallow_copy(tree)
  for zone in PT.iter_all_Zone_t(tree_vl):
    if PT.get_node_from_predicates(zone, vtx_gc_query) is not None:
      raise RuntimeError("Vertex located 1to1 GridConnectivity_t nodes are not supported in merge_zones." \
                         " Use a tree with FaceCenter located joins.")
  for base, zone in PT.get_children_from_predicates(tree_vl, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    for zgc, gc in PT.get_children_from_predicates(zone, face_gc_query, ancestors=True):
      if PT.GridConnectivity.ZoneDonorPath(gc, PT.get_name(base)) not in zone_to_id:
        PT.rm_child(zgc, gc)
  VL.generate_jns_vertex_list(tree_vl, comm, have_isolated_faces=True)
  #Reput in tree
  for zone_path in zone_paths:
    zone    = PT.find_node_from_path(tree, zone_path)
    zone_vl = PT.find_node_from_path(tree_vl, zone_path)
    for zgc in PT.get_children_from_label(zone, 'ZoneGridConnectivity_t'):
      zgc_vl = PT.find_child_from_name(zone_vl, PT.get_name(zgc))
      for gc_vl in PT.get_children_from_predicate(zgc_vl, PTp.label_is('GridConnectivity_t') \
          & PTp.has_location('Vertex')):
        PT.add_child(zgc, gc_vl)

  # Collect interface data
  interface_dn_f = []
  interface_ids_f = []
  interface_dom = []
  interface_dn_v = []
  interface_ids_v = []
  for zone_path, zone in zip(zone_paths, zones):
    base_name, zone_name = zone_path.split('/')
    for zgc, gc in PT.get_children_from_predicates(zone, face_gc_query, ancestors=True):
      opp_zone_path = PT.GridConnectivity.ZoneDonorPath(gc, base_name)
      if opp_zone_path in zone_to_id:
        if PT.GridConnectivity.isperiodic(gc):
          PT.new_node('__maia_jn_update__', 'Descriptor_t', value=str(zone_to_id[opp_zone_path]), parent=gc)
        else:
          PT.new_node('__maia_merge__', 'Descriptor_t', parent=gc)
          gc_path = f"{zone_path}/{PT.get_name(zgc)}/{PT.get_name(gc)}"
          gc_path_opp = MJT.get_jn_donor_path(tree, gc_path)
      if PT.get_child_from_name(gc, '__maia_merge__') is not None and gc_path < gc_path_opp:
        interface_dom.append((zone_to_id[zone_path], zone_to_id[opp_zone_path]))

        pl  = as_pdm_gnum(PT.get_np_value(PT.find_child_from_name(gc, 'PointList'))[0])
        pld = as_pdm_gnum(PT.get_np_value(PT.find_child_from_name(gc, 'PointListDonor'))[0])

        interface_dn_f.append(pl.size)
        interface_ids_f.append(np_utils.interweave_arrays([pl,pld]))

        # Find corresponding vertex
        if len(PT.get_name(gc)) < 28:
          pred = PT.pred.name_is(f'{PT.get_name(gc)}#Vtx')
        else:
          sub_pred = PT.pred.name_is('OriginalName') & PT.pred.value_is(f'{PT.get_name(gc)}#Vtx')
          pred = PT.pred.IS_GC & PT.pred.NodePredicate(lambda n : PT.get_child_from_predicate(n, sub_pred) is not None)
        gc_vtx = PT.find_child_from_predicate(zgc, pred)
        pl_v  = as_pdm_gnum(PT.get_np_value(PT.find_child_from_name(gc_vtx, 'PointList'))[0])
        pld_v = as_pdm_gnum(PT.get_np_value(PT.find_child_from_name(gc_vtx, 'PointListDonor'))[0])
        interface_dn_v.append(pl_v.size)
        interface_ids_v.append(np_utils.interweave_arrays([pl_v,pld_v]))

  # Generate interfaces
  graph_idx, graph_ids, graph_dom = PDM.interface_to_graph(\
      len(interface_dn_v), False, interface_dn_v, interface_ids_v, interface_dom, comm)
  graph_dict_v = {'graph_idx' : graph_idx, 'graph_ids' : graph_ids, 'graph_dom' : graph_dom}

  graph_idx, graph_ids, graph_dom = PDM.interface_to_graph(\
      len(interface_dn_f), False, interface_dn_f, interface_ids_f, interface_dom, comm)
  graph_dict_f = {'graph_idx' : graph_idx, 'graph_ids' : graph_ids, 'graph_dom' : graph_dom}

  # Empty graph for cells
  graph_dict_c = {'graph_idx' : np.array([0], np.int32),
                  'graph_ids' : np.empty(0, pdm_dtype),
                  'graph_dom' : np.empty(0, np.int32)}

  # Collect distributions
  entities = ['Vertex', 'Face', 'Cell']
  blocks_distri_l:Dict[str, List[NDArray]] = {entity : [] for entity in entities}
  selected_l:Dict[str, List[NDArray]]      = {entity : [] for entity in entities}
  for zone in zones:
    for entity in entities:
      if entity == 'Vertex':
        distri = as_pdm_gnum(MT.Zone.vtx_distribution(zone))
      elif entity == 'Face':
        elt_node = PT.Zone.NGonNode(zone) if cell_dim == 3 else MT.Zone.EdgeNode(zone)
        distri = as_pdm_gnum(MT.Element.distribution(elt_node))
      elif entity == 'Cell':
        distri = as_pdm_gnum(MT.Zone.cell_distribution(zone))
      blocks_distri_l[entity].append(par_utils.partial_to_full_distribution(distri, comm))
      selected_l[entity].append(np.arange(distri[0], distri[1], dtype=pdm_dtype)+1)

  # Create merge protocols
  # In 2D, we adopt the convention Cell = surfacic elts, Face = lineic elts, Vertex = vertices
  # for paradigm MbM objects
  mbm_vtx  = PDM.MultiBlockMerge(n_zone, blocks_distri_l['Vertex'], selected_l['Vertex'], graph_dict_v, comm)
  mbm_face = PDM.MultiBlockMerge(n_zone, blocks_distri_l['Face'  ], selected_l['Face'  ], graph_dict_f, comm)
  mbm_cell = PDM.MultiBlockMerge(n_zone, blocks_distri_l['Cell'  ], selected_l['Cell'  ], graph_dict_c, comm)
  all_mbm = {'Vertex' : mbm_vtx, 'Face' : mbm_face, 'Cell' : mbm_cell}

  merged_distri_vtx  = mbm_vtx .get_merged_distri()
  merged_distri_face = mbm_face.get_merged_distri()
  merged_distri_cell = mbm_cell.get_merged_distri()

  zone_dims = np.array([[merged_distri_vtx[-1], merged_distri_cell[-1], 0]], dtype=output_dtype, order='F')
  merged_zone = PT.new_Zone('MergedZone', size=zone_dims, type='Unstructured')

  # NGon
  _merge_ngon(all_mbm, tree, merged_zone, comm)

  # Generate NFace (TODO)
  pass


  # Merge all mesh data
  vtx_data_queries = [
                      ['GridCoordinates_t'],
                      [PTp.label_is('FlowSolution_t') & PTp.has_location('Vertex') & ~HAS_POINTLIST],
                      [PTp.label_is('DiscreteData_t') & PTp.has_location('Vertex') & ~HAS_POINTLIST],
                     ]
  cell_data_queries = [
                       [PTp.label_is('FlowSolution_t') & PTp.has_location('CellCenter') & ~HAS_POINTLIST],
                       [PTp.label_is('DiscreteData_t') & PTp.has_location('CellCenter') & ~HAS_POINTLIST],
                      ]
  _merge_allmesh_data(mbm_vtx,  zones, merged_zone, vtx_data_queries)
  _merge_allmesh_data(mbm_cell, zones, merged_zone, cell_data_queries)

  _merge_pls_data(all_mbm, zones, merged_zone, comm, subset_merge_strategy)

  MT.new_Distribution({'Vertex' : par_utils.full_to_partial_distribution(merged_distri_vtx, comm),
                      'Cell'   : par_utils.full_to_partial_distribution(merged_distri_cell, comm)},
                     merged_zone)

  # Move back non 1to1 GC_t to ZoneGridConnectivity
  for zbc in PT.get_children_from_label(merged_zone, 'ZoneBC_t'):
    zbc_name = PT.get_name(zbc)
    if zbc_name.startswith('maia_'): # This is a fake ZBC
      zgc = PT.update_child(merged_zone, zbc_name[5:], 'ZoneGridConnectivity_t')
      for child in PT.get_children(zbc):
        PT.set_label(child, 'GridConnectivity_t')
        PT.add_child(zgc, child)
      PT.rm_child(merged_zone, zbc)


  return merged_zone

def _merge_allmesh_data(mbm, zones, merged_zone, data_queries):
  """
  Merge the all DataArray supported by allCells or allVertex (depending on query and mbm),
  found under each of data_query (query must start from zone node), from input zones
  to merged_zone.
  """

  to_merge  = dict()

  for query in data_queries:
    for zone in zones:
      #For global data, we should have only one parent
      for node, data in PT.get_children_from_predicates(zone, query + ['DataArray_t'], ancestors=True):
        dic_path = PT.get_name(node) + '/' + PT.get_name(data)
        _append_or_create(to_merge, dic_path, data[1])

  merged = {key : mbm.merge_field(datas) for key, datas in to_merge.items()}


  additional_types = ['GridLocation_t', 'Descriptor_t', 'DataClass_t', 'DimensionalUnits_t']
  for query in data_queries:
    #Use zone 0 to get node type and value. Nodes must be know in every zone
    for node in PT.get_children_from_predicates(zones[0], query):
      m_node = PT.update_child(merged_zone, PT.get_name(node), PT.get_label(node), PT.get_value(node))
      for data in PT.iter_children_from_label(node, 'DataArray_t'):
        PT.new_DataArray(data[0], merged[PT.get_name(node) + '/' + PT.get_name(data)], parent=m_node)
      for type in additional_types:
        for sub_node in PT.iter_children_from_label(node, type):
          PT.add_child(m_node, sub_node)

def gather_subsets(zones, query, merge_strategy, is_bcds=False):
  """
  Create a dict mapping new node path to the list of subset nodes to merge,
  depending of merge strategy.
  """
  subset_groups = {}

  for i,zone in enumerate(zones):
    for path in PT.predicates_to_paths(zone, query):

      if merge_strategy == 'name':
        common_path = path
      elif merge_strategy == 'family':
        nodepath = PT.utils.path_head(path) if is_bcds else path # To get BC node is case of BCDS
        node = PT.get_node_from_path(zone, nodepath)
        famnode = PT.get_child_from_label(node, 'FamilyName_t')
        if famnode is not None:
          pos = 1 if is_bcds else -1
          common_path = PT.utils.update_path_elt(path, pos, lambda s : PT.get_value(famnode))
        else: # Fallback if node has no Family
          common_path = path
      else:
        pos = 1 if is_bcds else -1
        common_path = PT.utils.update_path_elt(path, pos, lambda s: s + f'.{i}')

      if common_path not in subset_groups:
        subset_groups[common_path] = len(zones) * [None]
      subset_groups[common_path][i] = PT.get_node_from_path(zone, path)

  return subset_groups

def pre_merge_families_per_zone(zone, query, comm):
  """ Pre merge the subset according to their family, for a given zone.
  This is because merge_pl_data only support one node per zone after.
  """
  bcds_pl    = PTp.label_is('BCDataSet_t') &  HAS_POINTLIST
  bcds_no_pl = PTp.label_is('BCDataSet_t') & ~HAS_POINTLIST
  fam_to_merge = {}
  for node_list in PT.get_children_from_predicates(zone, query, ancestors=True):
    node = node_list[-1]
    if PT.get_child_from_label(node, 'FamilyName_t') is not None:
      fam = PT.get_value(PT.get_child_from_label(node, 'FamilyName_t'))
      if fam in fam_to_merge:
        fam_to_merge[fam].append(node)
      else:
        fam_to_merge[fam] = [node]

  for fam, nodes in fam_to_merge.items():
    if len(nodes) > 1:
      parent = node_list[-2] if len(node_list) > 1 else zone
      merged_node = GN.concatenate_subset_nodes(nodes, comm, output_name=f'merged{fam}',
                                                additional_data_queries=[[bcds_no_pl, 'BCData_t', 'DataArray_t']],
                                                additional_child_queries=['AdditionalFamilyName_t', 'FamilyName_t'])

      # Also merge BCDS with pl if node is a BC
      if PT.get_label(merged_node) == 'BC_t':
        ds_names = [PT.get_name(n) for n in PT.get_children_from_predicate(nodes[0], bcds_pl)]
        for node in nodes: # Check : data should be the same on each node
          assert [PT.get_name(n) for n in PT.get_children_from_predicate(node, bcds_pl)] == ds_names
        for ds_name in ds_names:
          ds_nodes = [PT.get_child_from_name(node, ds_name) for node in nodes]
          merged_ds = GN.concatenate_subset_nodes(ds_nodes, comm, output_name=ds_name, additional_data_queries=['BCData_t/DataArray_t'])
          PT.add_child(merged_node, merged_ds)

      PT.add_child(parent, merged_node)
      for node in nodes:
        PT.rm_child(parent, node)

def _merge_pls_data(all_mbm, zones, merged_zone, comm, merge_strategy='name'):
  """
  Wrapper to perform a merge of the following subset nodes (when having a PointList) :
    FlowSolution_t, DiscreteData_t, ZoneSubRegion_t, BC_t, GridConnectivity_t, BCDataSet_t
  from the input zones to the merged zone merged_zone.

  If merged_strategy=='name', the nodes subset set having the same name on different zones are merged
  into a single one.
  Otherwise, force to keep one subset_node per input zone to let the user manage its merge.
  Merging by name is not performed for GridConnectivity_t
  """
  #In each case, we need to collect all the nodes, since some can be absent of a given zone
  loc = 'FaceCenter' if PT.Zone.CellDimension(zones[0]) == 3 else 'EdgeCenter'
  jn_to_keep = PTp.label_is('GridConnectivity_t') & PTp.has_location(loc) & ~PTp.has_child_of_name('__maia_merge__')

  #Order : FlowSolution/DiscreteData/ZoneSubRegion, BC, BCDataSet, GridConnectivity_t,
  all_subset_queries = [
      [PTp.label_in(['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t']) & HAS_POINTLIST],
      ['ZoneBC_t', 'BC_t'],
      ['ZoneBC_t', 'BC_t', PTp.label_is('BCDataSet_t') & HAS_POINTLIST],
      ['ZoneGridConnectivity_t', jn_to_keep]
      ]

  all_data_queries = [
      ['DataArray_t'],
      [PTp.label_is('BCDataSet_t') & ~HAS_POINTLIST, 'BCData_t', 'DataArray_t'],
      ['BCData_t', 'DataArray_t'],
      ['PointListDonor'],
      ]

  zones = [PT.shallow_copy(z) for z in zones] # We may modify zones structure

  # Trick to avoid spectific treatment of ZoneSubRegions (add PL)
  for zone in zones:
    for zsr in PT.iter_children_from_label(zone, 'ZoneSubRegion_t'):
      #Copy PL when related to bc/gc to avoid specific treatement
      related = PT.Container.SubsetNode(zsr, zone)
      if related is not zsr:
        PT.add_child(zsr, PT.find_child_from_name(related, 'PointList'))

  i_query = 0
  for query, rules in zip(all_subset_queries, all_data_queries):

    _merge_strategy = None if query[0] == 'ZoneGridConnectivity_t' else merge_strategy

    if _merge_strategy == 'family':
      # We can merge at most one node per zone, so in family case we may need to concatenate first
      for zone in zones:
        pre_merge_families_per_zone(zone, query, comm)

    subset_groups = gather_subsets(zones, query, _merge_strategy, i_query==2)

    #Merge and add to output
    for path,subset_nodes in subset_groups.items():
      #We have to retrieve a zone knowing this node to deduce the kind of parent nodes and gridLocation
      master_idx = [n is not None for n in subset_nodes].index(True)
      master_zone = zones[master_idx]

      location = PT.Subset.GridLocation(subset_nodes[master_idx])
      key = location.split('Center')[0]
      if key == 'Edge':
        assert PT.Zone.CellDimension(merged_zone) == 2
        key = 'Face'
      mbm = all_mbm[key]
      merged_pl = _merge_pl_data(mbm, zones, subset_nodes, location, rules, comm)
      # Enforce zone dtype for output PL
      for pl in PT.get_children_from_name(merged_pl, 'PointList*'):
        pl[1] = np_utils.safe_int_cast(pl[1], merged_zone[1].dtype)

      #Rebuild structure until last node
      parent = merged_zone
      path_split = path.split('/')
      master_nodes = PT.get_child_from_predicates(master_zone, query[:-1], ancestors=True)
      for i, master_node in enumerate(master_nodes):
        parent = PT.update_child(parent, path_split[i], PT.get_label(master_node), PT.get_value(master_node))

      PT.set_name(merged_pl, PT.utils.path_tail(path))

      # If internal perio jns, update GCDonorName on merged zone
      if PT.get_child_from_name(merged_pl, '__maia_jn_update__') is not None:
        opp_domain_id = PT.get_value(PT.get_child_from_name(merged_pl, '__maia_jn_update__'))
        donor_name_node = PT.get_child_from_name(merged_pl, "GridConnectivityDonorName")
        PT.set_value(donor_name_node, f"{PT.get_value(donor_name_node)}.{opp_domain_id}")
        PT.rm_children_from_name(merged_pl, '__maia_jn_update__')

      PT.add_child(parent, merged_pl)
    i_query += 1

  # Since link may be broken in merged zone, it is safer to remove it
  for zsr in PT.iter_children_from_label(merged_zone, 'ZoneSubRegion_t'):
    PT.rm_children_from_name(zsr, 'BCRegionName')
    PT.rm_children_from_name(zsr, 'GridConnectivityRegionName')

def _equilibrate_data(data, comm, distri=None, distri_full=None):
  if distri_full is None:
    if distri is None:
      first = next(iter(data.values()))
      distri_full = par_utils.gather_and_shift(first.size, comm, pdm_dtype)
    else:
      distri_full = par_utils.partial_to_full_distribution(distri, comm)

  ideal_distri = par_utils.uniform_distribution(distri_full[-1], comm)
  dist_data = EP.block_to_block(data, distri_full, ideal_distri, comm)

  return ideal_distri, dist_data


def _merge_pl_data(mbm, zones, subset_nodes, loc, data_query, comm):
  """
  Internal function used by _merge_zones to produce a merged node from the zones to merge
  and the path to a subset node (having a PL)
  Subset nodes comming from different zones with the same path will be merged
  Also merge all the nodes found under the data_query query (starting from subset_node)
  requested in data_queries list

  Return the merged subset node
  """

  ref_node = None

  has_data  = []
  strides   = []
  all_datas = {}
  assert len(zones) == len(subset_nodes)
  for zone, node in zip(zones, subset_nodes):
    if loc == 'Vertex':
      distri_ptb = MT.Zone.vtx_distribution(zone)
    elif loc == 'EdgeCenter':
      distri_ptb = MT.Element.distribution(MT.Zone.EdgeNode(zone))
    elif loc == 'FaceCenter':
      distri_ptb = MT.Element.distribution(PT.Zone.NGonNode(zone))
    elif loc == 'CellCenter':
      distri_ptb = MT.Zone.cell_distribution(zone)
    if node is not None:
      ref_node = node #Take any node as reference, to build name/type/value of merged node

      pl = PT.get_np_value(PT.find_child_from_name(node, 'PointList'))[0]
      part_data = {}
      for nodes in PT.get_children_from_predicates(node, data_query, ancestors=True):
        path =  '/'.join([PT.get_name(node) for node in nodes])
        data_n = nodes[-1]
        data = data_n[1]
        if data_n[1].ndim > 1:
          assert data_n[1].ndim == 2 and PT.get_label(data_n) == 'IndexArray_t'
          for dim in range(data_n[1].shape[0]): #Manage U (1,N) or S (3,N) PL
            _append_or_create(part_data, f'{path}_{dim}', np.ascontiguousarray(data_n[1][dim]))
        else:
          _append_or_create(part_data, path, data)
      #TODO maybe it is just a BtB -- nope because we want to reorder; but we could do one with all pl at once
      stride = np.zeros(distri_ptb[1] - distri_ptb[0], np.int32)
      GI = EP.GlobalIndexer(distri_ptb, pl-1, comm)
      mask = GI.access_counts > 0
      stride[mask] = 1
      dist_data = {key: GI.Put(pdata[0])[mask] for key, pdata in part_data.items()}
      dist_data['PL'] = np.flatnonzero(mask) + distri_ptb[0] + 1

      has_data.append(True)
      strides.append(stride)
      for data_path, data in dist_data.items():
        _append_or_create(all_datas, data_path, data)

    else:
      has_data.append(False)
      strides.append(np.zeros(distri_ptb[1] - distri_ptb[0], np.int32))

  #Fill data for void zones
  for data_path, datas in all_datas.items():
    if len(datas) != sum(has_data):
      missing = [PT.get_name(subset) for subset in subset_nodes if \
                 subset is not None and PT.get_node_from_path(subset, data_path) is None]
      raise RuntimeError(f"Data {data_path} is defined in some subsets, but missing in {missing}")
    zero_data = np.empty(0, datas[0].dtype)
    data_it = iter(datas)
    updated_data = [next(data_it) if _has_data else zero_data for i,_has_data in enumerate(has_data)]
    all_datas[data_path] = updated_data

  pl_data = all_datas.pop('PL')
  _, merged_pl = mbm.merge_and_update(mbm, [as_pdm_gnum(pl) for pl in pl_data], strides)
  merged_data = {'PointList' : merged_pl}

  # For periodic jns of zones to merge, PointListDonor must be transported and updated.
  # Otherwise, it must just be transported to new zone
  if PT.get_node_from_name(ref_node, '__maia_jn_update__') is not None:
    opp_dom = int(PT.get_value(PT.get_node_from_name(ref_node, '__maia_jn_update__')))
    pld_data = all_datas.pop('PointListDonor_0') #Since we merge U zones we should have only 1D-PL
    block_datas   = [as_pdm_gnum(pld) for pld in pld_data]
    block_domains = [opp_dom*np.ones(pld.size, np.int32) for pld in pld_data]
    merged_data['PointListDonor_0'] = mbm.merge_and_update(mbm, block_datas, strides, block_domains)[1]

  merged_data.update({path : mbm.merge_field(datas, strides)[1] for path, datas in all_datas.items()})

  # Data is merged, but distributed using pl distri. We do a BtB to re equilibrate it
  merged_pl_distri, merged_data = _equilibrate_data(merged_data, comm)

  #Creation of node
  merged_node = PT.new_node(PT.get_name(ref_node), PT.get_label(ref_node), PT.get_value(ref_node))
  PT.new_IndexArray(value=merged_data['PointList'].reshape((1, -1), order='F'), parent=merged_node)

  for nodes in PT.get_children_from_predicates(ref_node, data_query, ancestors=True):
    path =  '/'.join([PT.get_name(node) for node in nodes])
    # #Rebuild structure if any
    sub_ref = ref_node
    merged_parent = merged_node
    for node in nodes[:-1]:
      sub_ref = PT.get_child_from_name(sub_ref, PT.get_name(node))
      merged_parent = PT.update_child(merged_parent, PT.get_name(sub_ref), PT.get_label(sub_ref), PT.get_value(sub_ref))
    if PT.get_label(nodes[-1]) == 'IndexArray_t':
      # Recombine (1,N) or (3,N) array
      keys = [f'{path}_{idim}' for idim in range(3)]
      to_combine = [merged_data[key] for key in keys if key in merged_data]
      combined = np.empty((len(to_combine), to_combine[0].size), to_combine[0].dtype, order='F')
      for i, array in enumerate(to_combine):
        combined[i,:] = array
      PT.new_IndexArray(PT.get_name(nodes[-1]), combined, merged_parent)
    else:
      PT.new_DataArray(PT.get_name(nodes[-1]), merged_data[path], parent=merged_parent)

  # Add these nodes taking value for any input node
  additional_types = ['GridLocation_t', 'GridConnectivityType_t', 'GridConnectivityProperty_t']
  for type in additional_types:
    for sub_node in PT.iter_children_from_label(ref_node, type):
      PT.add_child(merged_node, sub_node)

  # Add these nodes only if same name / value on all input nodes
  merge_me = PTp.label_in(['AdditionalFamilyName_t', 'FamilyName_t', 'Descriptor_t'])
  common = {(PT.get_name(n), PT.get_label(n), PT.get_value(n)) \
             for n in PT.iter_children_from_predicate(ref_node, merge_me)}
  for subset_node in [x for x in subset_nodes if x is not None]:
    # Use set intersection to eliminate nodes that does not appear on this subset
    common = common & {(PT.get_name(n), PT.get_label(n), PT.get_value(n)) \
                        for n in PT.iter_children_from_predicate(subset_node, merge_me)}
  for c in sorted(common): #Sort to garantie same insertion order across mpi ranks
    PT.new_child(merged_node, name=c[0], label=c[1], value=c[2])

  MT.new_Distribution({'Index' : merged_pl_distri}, merged_node)

  return merged_node

def _merge_ngon(all_mbm, tree, merged_zone, comm):
  """
  Internal function used by _merge_zones to create the merged NGonNode
  """
  out_dtype = merged_zone[1].dtype

  zone_paths = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  zone_to_id = {path : i for i, path in enumerate(zone_paths)}
  dim = PT.Zone.CellDimension(PT.find_node_from_path(tree, zone_paths[0]))
  get_face_node = PT.Zone.NGonNode if dim == 3 else MT.Zone.EdgeNode

  # Create working data
  for zone_path, dom_id in zone_to_id.items():
    zone = PT.get_node_from_path(tree, zone_path)
    face_node = get_face_node(zone)
    pe_bck = PT.get_child_from_name(face_node, 'ParentElements')[1]
    pe = pe_bck.copy()
    # If NGon are first, then PE indexes cell, we must shift : PDM expect cell starting at 1
    if PT.Element.Range(face_node)[0] == 1:
      np_utils.shift_nonzeros(pe, -PT.Element.Size(face_node))
    PT.new_DataArray('UpdatedPE', pe, parent=face_node)
    PT.new_DataArray('PEDomain',  dom_id * np.ones_like(pe_bck, dtype=np.int32), parent=face_node)

  # First, we need to update the PE node to include cells of opposite zone
  query = PT.pred.IS_GC & PTp.has_child_of_name('__maia_merge__')

  for zone_path_send in zone_paths:
    base_n = zone_path_send.split('/')[0]
    dom_id_send = zone_to_id[zone_path_send]
    zone_send = PT.get_node_from_path(tree, zone_path_send)
    face_send = get_face_node(zone_send)
    face_distri_send = MT.Element.distribution(face_send)
    pe_send          = PT.get_child_from_name(face_send, 'UpdatedPE')[1]

    gcs = PT.get_nodes_from_predicate(zone_send, query, depth=2)
    all_pls = [PT.get_child_from_name(gc, 'PointList')[1][0]-1 for gc in gcs]
    part_pe = EP.block_to_part(pe_send[:,0], face_distri_send, all_pls, comm)
    for i, gc in enumerate(gcs):

      pld = PT.get_child_from_name(gc, 'PointListDonor')[1][0]

      #This is the left cell of the join face present in PL. Send it to opposite zone
      part_pe_gc = part_pe[i]

      # Get send data on the opposite zone and update PE
      zone_path = PT.GridConnectivity.ZoneDonorPath(gc, base_n)
      zone = PT.find_node_from_path(tree, zone_path)
      face_node = get_face_node(zone)
      pe      = PT.get_np_value(PT.find_child_from_name(face_node, 'UpdatedPE'))
      pe_dom  = PT.get_np_value(PT.find_child_from_name(face_node, 'PEDomain'))
      face_distri = MT.Element.distribution(face_node)

      GI = EP.GlobalIndexer(face_distri, pld-1, comm)
      local_faces = GI.access_counts > 0
      assert np.max(pe[local_faces, 1], initial=0) == 0 #Initial = trick to admit empty array
      GI.Put(part_pe_gc, pe[:,1])
      pe_dom[local_faces, 1] = dom_id_send

  #PE are ready, collect data
  ec_l = []
  ec_stride_l = []
  pe_l = []
  pe_stride_l = []
  pe_dom_l = []
  for zone_path in zone_paths:
    zone = PT.get_node_from_path(tree, zone_path)
    face_node = get_face_node(zone)
    face_node_cnt = MT.Element.connectivity(face_node) # face_vtx or edge_vtx
    pe     = as_pdm_gnum(PT.get_child_from_name(face_node, 'UpdatedPE')[1])
    pe_dom = PT.get_child_from_name(face_node, 'PEDomain')[1]

    ec_l.append(as_pdm_gnum(face_node_cnt.values))
    ec_stride_l.append(np_utils.safe_int_cast(face_node_cnt.counts, np.int32))

    #We have to detect and remove bnd faces from PE to use PDM stride
    bnd_faces = np.where(pe == 0)[0]
    stride = 2*np.ones(pe.shape[0], dtype=np.int32)
    stride[bnd_faces] = 1
    pe_stride_l.append(stride)
    #Also remove 0 from pe and pe_domain
    pe_l.append(np.delete(pe.reshape(-1), 2*bnd_faces+1))
    pe_dom_l.append(np_utils.safe_int_cast(np.delete(pe_dom.reshape(-1), 2*bnd_faces+1), np.int32))

  # Now merge and update
  merged_ec_stri, merged_ec = all_mbm['Face'].merge_and_update(all_mbm['Vertex'], ec_l, ec_stride_l)
  merged_pe_stri, merged_pe = all_mbm['Face'].merge_and_update(all_mbm['Cell'],   pe_l, pe_stride_l, pe_dom_l)
  merged_distri_face = all_mbm['Face'].get_merged_distri()

  # Reshift ESO to make it global
  merged_cnt = vs.from_counts(merged_ec_stri, merged_ec)

  #Post treat PE : we need to reintroduce 0 on boundary faces (TODO : could avoid tmp array ?)
  bnd_faces = np.where(merged_pe_stri == 1)[0]
  merged_pe_idx  = np_utils.sizes_to_indices(merged_pe_stri)
  merged_pe_full = np.insert(merged_pe, merged_pe_idx[bnd_faces]+1, 0)
  assert (merged_pe_full.size == 2*merged_pe_stri.size)
  pe = np.empty((merged_pe_stri.size, 2), order='F', dtype=out_dtype)
  pe[:,0] = merged_pe_full[0::2]
  pe[:,1] = merged_pe_full[1::2]
  np_utils.shift_nonzeros(pe, merged_distri_face[-1])

  # Finally : create ngon node
  erange = np.array([1, merged_distri_face[-1]], out_dtype)
  if dim == 3:
    eso, merged_ec = cgns_connectivity_from_vs(merged_cnt, comm, out_dtype)
    merged_face_node = PT.new_NGonElements(erange=erange, eso=eso, ec=merged_ec, pe=pe)
  else:
    merged_ec = np_utils.safe_int_cast(merged_cnt.values, out_dtype)
    merged_face_node = PT.new_Elements('EdgeElements', 'BAR_2', erange=erange, econn=merged_ec, pe=pe)
  # HERE
  MT.new_Distribution({'Element' : par_utils.full_to_partial_distribution(merged_distri_face, comm)},
                       merged_face_node)
  PT.add_child(merged_zone, merged_face_node)

