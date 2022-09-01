import numpy as np
from Pypdm import Pypdm as PDM

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.sids   as sids
import maia.pytree.maia   as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import py_utils, np_utils, par_utils

from maia.algo.dist import matching_jns_tools as MJT
from maia.algo.dist import concat_nodes as GN
from maia.algo.dist import vertex_list as VL
from maia.transfer.dist_to_part import data_exchange as MBTP
from maia.transfer.part_to_dist import data_exchange as MPTB

def merge_connected_zones(tree, comm, **kwargs):
  """Detect all the zones connected through 1to1 matching jns and merge them.

  See :func:`merge_zones` for full documentation.
  
  Args:
    tree (CGNSTree): Input distributed tree
    comm (MPIComm) : MPI communicator
    kwargs: any argument of :func:`merge_zones`, excepted output_path

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #merge_connected_zones@start
        :end-before: #merge_connected_zones@end
        :dedent: 2
  """
  grouped_zones_paths = sids.find_connected_zones(tree)

  for i, zones_path in enumerate(grouped_zones_paths):
    zones_path_u = [path for path in zones_path \
        if sids.Zone.Type(I.getNodeFromPath(tree, path)) == 'Unstructured']
    base = zones_path[0].split('/')[0]
    merge_zones(tree, zones_path_u, comm, output_path=f'{base}/mergedZone{i}', **kwargs)

def merge_zones(tree, zones_path, comm, output_path=None, subset_merge='name', concatenate_jns=True):
  """Merge the given zones into a single one.

  Input tree is modified inplace : original zones will be removed from the tree and replaced
  by the merged zone. Merged zone is added with name *MergedZone* under the first involved Base
  except if output_path is not None : in this case, the provided path defines the base and zone name
  of the merged block.

  Subsets of the merged block can be reduced thanks to subset_merge parameter:
  
  - None   => no reduction occurs : all subset of all original zones remains on merged zone, with a
    numbering suffix.
  - 'name' => Subset having the same name on the original zones (within a same label) produces
    and unique subset on the output merged zone.

  Only unstructured-NGon trees are supported, and interfaces between the zones
  to merge must have a FaceCenter location.

  Args:
    tree (CGNSTree): Input distributed tree
    zones_path (list of str): List of pathes (BaseName/ZoneName) of the zones to merge
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
  assert all([sids.Zone.Type(I.getNodeFromPath(tree, path)) == 'Unstructured' for path in zones_path])
  #Those one will be needed for jn recovering
  MJT.add_joins_donor_name(tree, comm)

  #Force full donor name, otherwise it is hard to reset jns
  sids.enforceDonorAsPath(tree)

  # We create a tree including only the zones to merge to speed up some operations
  masked_tree = I.newCGNSTree()
  for zone_path in zones_path:
    base_n, zone_n = zone_path.split('/')
    masked_base = I.createUniqueChild(masked_tree, base_n, 'CGNSBase_t')
    I._addChild(masked_base, I.getNodeFromPath(tree, zone_path))

    #Remove from input tree at the same time
    I._rmNodeByPath(tree, zone_path)

  #Merge zones
  merged_zone = _merge_zones(masked_tree, comm, subset_merge)

  #Add output
  if output_path is None:
    output_base = I.getNodeFromPath(tree, zones_path[0].split('/')[0])
  else:
    output_base = I.getNodeFromPath(tree, output_path.split('/')[0])
    if output_base is None:
      output_base = I.newCGNSBase(output_path.split('/')[0], 3, 3, parent=tree)
    I.setName(merged_zone, output_path.split('/')[1])
  I._addChild(output_base, merged_zone)

  #First we have to retrieve PLDonor for external jn and update opposite zones
  merged_zone_path = I.getName(output_base) + '/' + I.getName(merged_zone)
  jn_to_pl = {}
  for jn_path in PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity_t'):
    gc = I.getNodeFromPath(tree, jn_path)
    jn_to_pl[jn_path] = \
        (PT.get_child_from_name(gc, 'PointList')[1], PT.get_child_from_name(gc, 'PointListDonor')[1], MT.getDistribution(gc))

  # Update opposite names when going to opp zone (intrazone have been caried before)
  for zgc, gc in PT.get_children_from_predicates(merged_zone, ['ZoneGridConnectivity_t', 'GridConnectivity_t'], ancestors=True):
    if I.getValue(gc) not in zones_path:
      opp_path = MJT.get_jn_donor_path(tree, f"{merged_zone_path}/{zgc[0]}/{gc[0]}")
      opp_gc = I.getNodeFromPath(tree, opp_path)
      opp_gc_donor_name = PT.get_child_from_name(opp_gc, 'GridConnectivityDonorName') #TODO factorize
      I.setValue(opp_gc_donor_name, gc[0])
  #Now all donor names are OK

  for zone_path in PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t'):
    is_merged_zone = zone_path == merged_zone_path
    zone = I.getNodeFromPath(tree, zone_path)
    for zgc, gc in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', 'GridConnectivity_t'], ancestors=True):
      #Update name and PL
      if I.getValue(gc) in zones_path: #Can be: jn from non concerned zone to merged zones or periodic from merged zones
        I.setValue(gc, merged_zone_path)
        jn_path = f"{zone_path}/{I.getName(zgc)}/{I.getName(gc)}"
        jn_path_opp= MJT.get_jn_donor_path(tree, jn_path)
        # Copy and permute pl/pld only for all the zones != merged zone OR for one gc over two for
        # merged zone
        if not is_merged_zone or jn_path_opp < jn_path:
          I.newIndexArray('PointList'     , jn_to_pl[jn_path_opp][1], parent=gc)
          I.newIndexArray('PointListDonor', jn_to_pl[jn_path_opp][0], parent=gc)
          PT.rm_children_from_name(gc, ":CGNS#Distribution")
          I._addChild(gc, jn_to_pl[jn_path_opp][2])

  if concatenate_jns:
    GN.concatenate_jns(tree, comm)

def _add_zone_suffix(zones, query):
  """Util function prefixing all the nodes founds by a query by the number of the zone"""
  for izone, zone in enumerate(zones):
    for node in PT.get_children_from_predicates(zone, query):
      I.setName(node, I.getName(node) + f".{izone}")
      # Also update internal (periodic) joins with new donor name
      if PT.get_child_from_name(node, '__maia_jn_update__') is not None:
        opp_domain_id = PT.get_child_from_name(node, '__maia_jn_update__')[1][0]
        donor_name_node = PT.get_child_from_name(node, "GridConnectivityDonorName")
        I.setValue(donor_name_node, f"{I.getValue(donor_name_node)}.{opp_domain_id}")

def _rm_zone_suffix(zones, query):
  """Util function removing the number of the zone in all the nodes founds by a query
  Use it to cleanup after _add_zone_suffix has been applied"""
  for zone in zones:
    for node in PT.get_children_from_predicates(zone, query):
      I.setName(node, '.'.join(I.getName(node).split('.')[:-1]))

def _merge_zones(tree, comm, subset_merge_strategy='name'):
  """
  Tree must contain *only* the zones to merge. We use a tree instead of a list of zone because it's easier
  to retrieve opposites zones througt joins. Interface beetween zones shall be described by faces
  """

  zones_path = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  n_zone = len(zones_path)
  zones = I.getZones(tree)
  assert min([sids.Zone.Type(zone) == 'Unstructured' for zone in zones]) == True

  zone_to_id = {path : i for i, path in enumerate(zones_path)}

  is_perio = lambda n : PT.get_child_from_label(n, 'GridConnectivityProperty_t') is not None
  gc_query = ['ZoneGridConnectivity_t', \
                lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] 
                and sids.Subset.GridLocation(n) == 'FaceCenter']

  # JNs to external zones must be excluded from vertex list computing
  tree_vl = I.copyRef(tree)
  for base, zone in PT.get_children_from_predicates(tree_vl, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    for zgc, gc in PT.get_children_from_predicates(zone, gc_query, ancestors=True):
      if sids.getZoneDonorPath(I.getName(base), gc) not in zone_to_id:
        I.rmNode(zgc, gc)
  VL.generate_jns_vertex_list(tree_vl, comm, have_isolated_faces=True)
  #Reput in tree
  for zone_path in zones_path:
    zone    = I.getNodeFromPath(tree, zone_path)
    zone_vl = I.getNodeFromPath(tree_vl, zone_path)
    for zgc in PT.get_children_from_label(zone, 'ZoneGridConnectivity_t'):
      zgc_vl = PT.get_child_from_name(zone_vl, I.getName(zgc))
      for gc_vl in PT.get_children_from_predicate(zgc_vl, lambda n: I.getType(n) == 'GridConnectivity_t' \
          and sids.Subset.GridLocation(n) == 'Vertex'):
        I._addChild(zgc, gc_vl)
  
  # Collect interface data
  interface_dn_f = []
  interface_ids_f = []
  interface_dom = []
  interface_dn_v = []
  interface_ids_v = []
  for zone_path, zone in zip(zones_path, zones):
    base_name, zone_name = zone_path.split('/')
    for zgc, gc in PT.get_children_from_predicates(zone, gc_query, ancestors=True):
      opp_zone_path = sids.getZoneDonorPath(base_name, gc)
      if opp_zone_path in zone_to_id:
        if is_perio(gc):
          I.newUserDefinedData('__maia_jn_update__', value = zone_to_id[opp_zone_path], parent=gc)
        else:
          I.newDescriptor('__maia_merge__', parent=gc)
          gc_path = f"{zone_path}/{I.getName(zgc)}/{I.getName(gc)}"
          gc_path_opp = MJT.get_jn_donor_path(tree, gc_path)
      if PT.get_child_from_name(gc, '__maia_merge__') is not None and gc_path < gc_path_opp:
        interface_dom.append((zone_to_id[zone_path], zone_to_id[opp_zone_path]))

        pl  = PT.get_child_from_name(gc, 'PointList')[1][0]
        pld = PT.get_child_from_name(gc, 'PointListDonor')[1][0]

        interface_dn_f.append(pl.size)
        interface_ids_f.append(np_utils.interweave_arrays([pl,pld]))

        # Find corresponding vertex
        gc_vtx = PT.get_child_from_name(zgc, f'{I.getName(gc)}#Vtx')
        pl_v  = PT.get_child_from_name(gc_vtx, 'PointList')[1][0]
        pld_v = PT.get_child_from_name(gc_vtx, 'PointListDonor')[1][0]
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
  blocks_distri_l = {entity : [] for entity in entities}
  selected_l      = {entity : [] for entity in entities}
  for zone in zones:
    for entity in entities:
      if entity == 'Face':
        distri = MT.getDistribution(sids.Zone.NGonNode(zone), 'Element')[1].astype(pdm_dtype)
      else:
        distri = MT.getDistribution(zone, entity)[1].astype(pdm_dtype)
      blocks_distri_l[entity].append(par_utils.partial_to_full_distribution(distri, comm))
      selected_l[entity].append(np.arange(distri[0], distri[1], dtype=pdm_dtype)+1)
  
  # Create merge protocols
  mbm_vtx  = PDM.MultiBlockMerge(n_zone, blocks_distri_l['Vertex'], selected_l['Vertex'], graph_dict_v, comm)
  mbm_face = PDM.MultiBlockMerge(n_zone, blocks_distri_l['Face'  ], selected_l['Face'  ], graph_dict_f, comm)
  mbm_cell = PDM.MultiBlockMerge(n_zone, blocks_distri_l['Cell'  ], selected_l['Cell'  ], graph_dict_c, comm)
  all_mbm = {'Vertex' : mbm_vtx, 'Face' : mbm_face, 'Cell' : mbm_cell}

  merged_distri_vtx  = mbm_vtx .get_merged_distri()
  merged_distri_face = mbm_face.get_merged_distri()
  merged_distri_cell = mbm_cell.get_merged_distri()
  
  zone_dims = np.array([[merged_distri_vtx[-1], merged_distri_cell[-1], 0]], order='F')
  merged_zone = I.newZone('MergedZone', zone_dims, ztype='Unstructured')

  # NGon
  I._addChild(merged_zone, _merge_ngon(all_mbm, tree, comm))

  # Generate NFace (TODO)
  pass


  loc_without_pl = lambda n, loc : sids.Subset.GridLocation(n) == loc and PT.get_node_from_name(n, 'PointList') is None
  # Merge all mesh data
  vtx_data_queries = [
                      ['GridCoordinates_t'],
                      [lambda n: I.getType(n) == 'FlowSolution_t' and loc_without_pl(n, 'Vertex')],
                      [lambda n: I.getType(n) == 'DiscreteData_t' and loc_without_pl(n, 'Vertex')],
                     ]
  cell_data_queries = [
                       [lambda n: I.getType(n) == 'FlowSolution_t' and loc_without_pl(n, 'CellCenter')],
                       [lambda n: I.getType(n) == 'DiscreteData_t' and loc_without_pl(n, 'CellCenter')],
                      ]
  _merge_allmesh_data(mbm_vtx,  zones, merged_zone, vtx_data_queries)
  _merge_allmesh_data(mbm_cell, zones, merged_zone, cell_data_queries)

  _merge_pls_data(all_mbm, zones, merged_zone, comm, subset_merge_strategy)

  MT.newDistribution({'Vertex' : par_utils.full_to_partial_distribution(merged_distri_vtx, comm),
                      'Cell'   : par_utils.full_to_partial_distribution(merged_distri_cell, comm)},
                     merged_zone)
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
        dic_path = I.getName(node) + '/' + I.getName(data)
        try:
          to_merge[dic_path].append(data[1])
        except KeyError:
          to_merge[dic_path] = [ data[1] ]
    
  merged = {key : mbm.merge_field(datas) for key, datas in to_merge.items()}


  additional_types = ['GridLocation_t', 'Descriptor_t', 'DataClass_t', 'DimensionalUnits_t']
  for query in data_queries:
    #Use zone 0 to get node type and value. Nodes must be know in every zone
    for node in PT.get_children_from_predicates(zones[0], query):
      m_node = I.createUniqueChild(merged_zone, I.getName(node), I.getType(node), I.getValue(node))
      for data in PT.iter_children_from_label(node, 'DataArray_t'):
        I.newDataArray(data[0], merged[I.getName(node) + '/' + I.getName(data)], parent=m_node)
      for type in additional_types:
        for sub_node in PT.iter_children_from_label(node, type):
          I._addChild(m_node, sub_node)

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
  has_pl = lambda n : PT.get_child_from_name(n, 'PointList') is not None
  jn_to_keep = lambda n : I.getType(n) == 'GridConnectivity_t' and sids.Subset.GridLocation(n) == 'FaceCenter'\
      and PT.get_child_from_name(n, '__maia_merge__') is None

  #Order : FlowSolution/DiscreteData/ZoneSubRegion, BC, BCDataSet, GridConnectivity_t, 
  all_subset_queries = [
      [lambda n : I.getType(n) in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t'] and has_pl(n)],
      ['ZoneBC_t', 'BC_t'],
      ['ZoneBC_t', 'BC_t', lambda n : I.getType(n) == 'BCDataSet_t' and has_pl(n)],
      ['ZoneGridConnectivity_t', jn_to_keep]
      ]

  all_data_queries = [
      ['DataArray_t'],
      [lambda n : I.getType(n) == 'BCDataSet_t' and not has_pl(n), 'BCData_t', 'DataArray_t'],
      ['BCData_t', 'DataArray_t'],
      ['PointListDonor'],
      ]

  # Trick to avoid spectific treatment of ZoneSubRegions (add PL)
  for zone in zones:
    for zsr in PT.iter_children_from_label(zone, 'ZoneSubRegion_t'):
      #Copy PL when related to bc/gc to avoid specific treatement
      if PT.get_child_from_name(zsr, 'BCRegionName') is not None or \
         PT.get_child_from_name(zsr, 'GridConnectivityRegionName') is not None:
        related = I.getNodeFromPath(zone, sids.getSubregionExtent(zsr, zone))
        I._addChild(zsr, PT.get_child_from_name(related, 'PointList'))

  for query, rules in zip(all_subset_queries, all_data_queries):
    if merge_strategy != 'name' or query[0] == 'ZoneGridConnectivity_t':
      _add_zone_suffix(zones, query)
    collected_paths = []
    #Collect
    for zone in zones:
      for nodes in PT.get_children_from_predicates(zone, query, ancestors=True):
        py_utils.append_unique(collected_paths, '/'.join([I.getName(node) for node in nodes]))
    #Merge and add to output
    for pl_path in collected_paths:
      #We have to retrieve a zone knowing this node to deduce the kind of parent nodes and gridLocation
      master_node = None
      for zone in zones:
        if I.getNodeFromPath(zone, pl_path) is not None:
          master_node = zone
          break
      assert master_node is not None
      parent = merged_zone
      location = sids.Subset.GridLocation(I.getNodeFromPath(master_node, pl_path))
      mbm = all_mbm[location.split('Center')[0]]
      merged_pl = _merge_pl_data(mbm, zones, pl_path, location, rules, comm)
      #Rebuild structure until last node
      for child_name in pl_path.split('/')[:-1]:
        master_node = PT.get_child_from_name(master_node, child_name)
        parent = I.createUniqueChild(parent, child_name, I.getType(master_node), I.getValue(master_node))

      I._addChild(parent, merged_pl)
    if merge_strategy != 'name'or query[0] == 'ZoneGridConnectivity_t':
      _rm_zone_suffix(zones, query)

  # Trick to avoid spectific treatment of ZoneSubRegions (remove PL on original zones)
  for zone in zones:
    for zsr in PT.iter_children_from_label(zone, 'ZoneSubRegion_t'):
      if PT.get_child_from_name(zsr, 'BCRegionName') is not None or \
         PT.get_child_from_name(zsr, 'GridConnectivityRegionName') is not None:
        PT.rm_children_from_name(zsr, 'PointList*')
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
  ideal_distri_full = par_utils.partial_to_full_distribution(ideal_distri, comm)

  BTB = PDM.BlockToBlock(distri_full, ideal_distri_full, comm)
  dist_data = dict()
  BTB.BlockToBlock_Exchange(data, dist_data)
  
  return ideal_distri, dist_data


def _merge_pl_data(mbm, zones, subset_path, loc, data_query, comm):
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
  for i, zone in enumerate(zones):
    node = I.getNodeFromPath(zone, subset_path)
    if loc == 'Vertex': 
      distri_ptb = MT.getDistribution(zone, 'Vertex')[1]
    elif loc == 'FaceCenter':
      distri_ptb = MT.getDistribution(sids.Zone.NGonNode(zone), 'Element')[1]
    elif loc == 'CellCenter':
      distri_ptb = MT.getDistribution(zone, 'Cell')[1]
    if node is not None:
      ref_node = node #Take any node as reference, to build name/type/value of merged node

      pl = PT.get_child_from_name(node, 'PointList')[1][0]
      part_data = {'PL' : [pl]}
      for nodes in PT.get_children_from_predicates(node, data_query, ancestors=True):
        path =  '/'.join([I.getName(node) for node in nodes])
        data_n = nodes[-1]
        data = data_n[1]
        if data_n[1].ndim > 1:
          assert data_n[1].ndim == 2 and data_n[1].shape[0] == 1
          data = data_n[1][0]
        try:
          part_data[path].append(data)
        except KeyError:
          part_data[path] = [data]
      #TODO maybe it is just a BtB -- nope because we want to reorder; but we could do one with all pl at once
      dist_data = MPTB.part_to_dist(distri_ptb, part_data, [pl.astype(pdm_dtype)], comm)

      stride = np.zeros(distri_ptb[1] - distri_ptb[0], np.int32)
      stride[dist_data['PL'] - distri_ptb[0] - 1] = 1

      has_data.append(True)
      strides.append(stride)
      for data_path, data in dist_data.items():
        try:
          all_datas[data_path].append(data)
        except KeyError:
          all_datas[data_path] = [data]

    else:
      has_data.append(False)
      strides.append(np.zeros(distri_ptb[1] - distri_ptb[0], np.int32))

  #Fill data for void zones
  for data_path, datas in all_datas.items():
    zero_data = np.empty(0, datas[0].dtype)
    data_it = iter(datas)
    updated_data = [next(data_it) if _has_data else zero_data for i,_has_data in enumerate(has_data)]
    all_datas[data_path] = updated_data
  
  pl_data = all_datas.pop('PL')
  _, merged_pl = mbm.merge_and_update(mbm, [pl.astype(pdm_dtype) for pl in pl_data], strides)
  merged_data = {'PointList' : merged_pl}

  # For periodic jns of zones to merge, PointListDonor must be transported and updated.
  # Otherwise, it must just be transported to new zone
  if PT.get_node_from_name(ref_node, '__maia_jn_update__') is not None:
    opp_dom = PT.get_node_from_name(ref_node, '__maia_jn_update__')[1][0]
    pld_data = all_datas.pop('PointListDonor')
    block_datas   = [pld.astype(pdm_dtype) for pld in pld_data]
    block_domains = [opp_dom*np.ones(pld.size, np.int32) for pld in pld_data]
    merged_data['PointListDonor'] = mbm.merge_and_update(mbm, block_datas, strides, block_domains)[1]

  merged_data.update({path : mbm.merge_field(datas, strides)[1] for path, datas in all_datas.items()})

  # Data is merged, but distributed using pl distri. We do a BtB to re equilibrate it
  merged_pl_distri, merged_data = _equilibrate_data(merged_data, comm)

  #Creation of node
  merged_node = I.createNode(I.getName(ref_node), I.getType(ref_node), I.getValue(ref_node))
  I.newIndexArray('PointList', merged_data['PointList'].reshape((1, -1), order='F'), parent=merged_node)

  for nodes in PT.get_children_from_predicates(ref_node, data_query, ancestors=True):
    path =  '/'.join([I.getName(node) for node in nodes])
    # #Rebuild structure if any
    sub_ref = ref_node
    merged_parent = merged_node
    for node in nodes[:-1]:
      sub_ref = PT.get_child_from_name(sub_ref, I.getName(node))
      merged_parent = I.createUniqueChild(merged_parent, I.getName(sub_ref), I.getType(sub_ref), I.getValue(sub_ref))
    if I.getType(nodes[-1]) == 'IndexArray_t':
      I.newIndexArray(I.getName(nodes[-1]), merged_data[path].reshape((1,-1), order='F'), merged_parent)
    else:
      I.newDataArray(I.getName(nodes[-1]), merged_data[path], merged_parent)

  additional_types = ['GridLocation_t', 'FamilyName_t', 'Descriptor_t',
                      'GridConnectivityType_t', 'GridConnectivityProperty_t']
  additional_names = []
  for type in additional_types:
    for sub_node in PT.iter_children_from_label(ref_node, type):
      I._addChild(merged_node, sub_node)
  for name in additional_names:
    for sub_node in PT.iter_children_from_name(ref_node, name):
      I._addChild(merged_node, sub_node)

  MT.newDistribution({'Index' : merged_pl_distri}, merged_node)

  return merged_node

def _merge_ngon(all_mbm, tree, comm):
  """
  Internal function used by _merge_zones to create the merged NGonNode
  """

  zones_path = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  zone_to_id = {path : i for i, path in enumerate(zones_path)}

  # Create working data
  for zone_path, dom_id in zone_to_id.items():
    ngon_node = sids.Zone.NGonNode(I.getNodeFromPath(tree, zone_path))
    pe_bck = PT.get_child_from_name(ngon_node, 'ParentElements')[1]
    pe = pe_bck.copy()
    # If NGon are first, then PE indexes cell, we must shift : PDM expect cell starting at 1
    if sids.Element.Range(ngon_node)[0] == 1:
      np_utils.shift_nonzeros(pe, -sids.Element.Size(ngon_node))
    I.newDataArray('UpdatedPE', pe, parent=ngon_node)
    I.newDataArray('PEDomain',  dom_id * np.ones_like(pe_bck), parent=ngon_node)

  # First, we need to update the PE node to include cells of opposite zone
  query = lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
                and PT.get_child_from_name(n, '__maia_merge__') is not None

  for zone_path_send in zones_path:
    base_n = zone_path_send.split('/')[0]
    dom_id_send = zone_to_id[zone_path_send]
    zone_send = I.getNodeFromPath(tree, zone_path_send)
    ngon_send = sids.Zone.NGonNode(zone_send)
    face_distri_send = MT.getDistribution(ngon_send, 'Element')[1].astype(pdm_dtype)
    pe_send          =  PT.get_child_from_name(ngon_send, 'UpdatedPE')[1]
    dist_data_send = {'PE' : pe_send[:,0]}

    gcs = PT.get_nodes_from_predicate(zone_send, query, depth=2)
    all_pls = [PT.get_child_from_name(gc, 'PointList')[1][0] for gc in gcs]
    part_data = MBTP.dist_to_part(face_distri_send, dist_data_send, all_pls, comm)
    for i, gc in enumerate(gcs):

      pld = PT.get_child_from_name(gc, 'PointListDonor')[1][0]

      #This is the left cell of the join face present in PL. Send it to opposite zone
      part_data_gc = {key : [data[i]] for key, data in part_data.items()}
      part_data_gc['FaceId'] = [pld]
    
      # Get send data on the opposite zone and update PE
      zone_path = sids.getZoneDonorPath(base_n, gc)
      zone = I.getNodeFromPath(tree, zone_path)
      ngon_node = sids.Zone.NGonNode(zone)
      face_distri = MT.getDistribution(ngon_node, 'Element')[1].astype(pdm_dtype)
      dist_data = MPTB.part_to_dist(face_distri, part_data_gc, [pld], comm)

      pe      = PT.get_child_from_name(ngon_node, 'UpdatedPE')[1]
      pe_dom  = PT.get_child_from_name(ngon_node, 'PEDomain')[1]
      local_faces = dist_data['FaceId'] - face_distri[0] - 1
      assert np.max(pe[local_faces, 1], initial=0) == 0 #Initial = trick to admit empty array
      pe[local_faces, 1] = dist_data['PE']
      pe_dom[local_faces, 1] = dom_id_send

  #PE are ready, collect data
  ec_l = []
  ec_stride_l = []
  pe_l = []
  pe_stride_l = []
  pe_dom_l = []
  for zone_path in zones_path:
    ngon_node = sids.Zone.NGonNode(I.getNodeFromPath(tree, zone_path))
    eso    = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
    pe     = PT.get_child_from_name(ngon_node, 'UpdatedPE')[1]
    pe_dom = PT.get_child_from_name(ngon_node, 'PEDomain')[1]

    ec_l.append(PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1])
    ec_stride_l.append(np.diff(eso).astype(np.int32))

    #We have to detect and remove bnd faces from PE to use PDM stride
    bnd_faces = np.where(pe == 0)[0]
    stride = 2*np.ones(pe.shape[0], dtype=np.int32) 
    stride[bnd_faces] = 1
    pe_stride_l.append(stride)
    #Also remove 0 from pe and pe_domain
    pe_l.append(np.delete(pe.reshape(-1), 2*bnd_faces+1))
    pe_dom_l.append(np.delete(pe_dom.reshape(-1), 2*bnd_faces+1).astype(np.int32))

  # Now merge and update
  merged_ec_stri, merged_ec = all_mbm['Face'].merge_and_update(all_mbm['Vertex'], ec_l, ec_stride_l)
  merged_pe_stri, merged_pe = all_mbm['Face'].merge_and_update(all_mbm['Cell'],   pe_l, pe_stride_l, pe_dom_l)
  merged_distri_face = all_mbm['Face'].get_merged_distri()

  # Reshift ESO to make it global
  eso_loc = np_utils.sizes_to_indices(merged_ec_stri, pdm_dtype)
  ec_distri = par_utils.gather_and_shift(eso_loc[-1], comm)
  eso = eso_loc + ec_distri[comm.Get_rank()]

  #Post treat PE : we need to reintroduce 0 on boundary faces (TODO : could avoid tmp array ?)
  bnd_faces = np.where(merged_pe_stri == 1)[0]
  merged_pe_idx  = np_utils.sizes_to_indices(merged_pe_stri)
  merged_pe_full = np.insert(merged_pe, merged_pe_idx[bnd_faces]+1, 0)
  assert (merged_pe_full.size == 2*merged_pe_stri.size)
  pe = np.empty((merged_pe_stri.size, 2), order='F', dtype=merged_pe.dtype)
  pe[:,0] = merged_pe_full[0::2]
  pe[:,1] = merged_pe_full[1::2]
  np_utils.shift_nonzeros(pe, merged_distri_face[-1])

  # Finally : create ngon node
  merged_ngon = I.newElements('NGonElements', 'NGON', erange=[1, merged_distri_face[-1]])
  I.newDataArray('ElementStartOffset',  eso,       parent=merged_ngon)
  I.newDataArray('ElementConnectivity', merged_ec, parent=merged_ngon)
  I.newDataArray('ParentElements',      pe,        parent=merged_ngon)
  MT.newDistribution({'Element' :             par_utils.full_to_partial_distribution(merged_distri_face, comm),
                      'ElementConnectivity' : par_utils.full_to_partial_distribution(ec_distri, comm)},
                      merged_ngon)
  return merged_ngon
  
