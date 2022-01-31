import numpy as np

import Converter.Internal as I

from Pypdm import Pypdm as PDM

from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.sids import Internal_ext as IE
from maia.sids import sids
from maia.utils import py_utils
from maia.utils.parallel import utils as par_utils
from maia.transform.dist_tree import add_joins_ordinal as AJO
from maia.generate import nodes as GN

from maia.sids import pytree as PT
from maia.connectivity import vertex_list as VL
from maia.tree_exchange.dist_to_part import data_exchange as MBTP
from maia.tree_exchange.part_to_dist import data_exchange as MPTB

def merge_connected_zones(tree, comm, **kwargs):
  """
  Shortcut for merge_zones to merge all the zones of the tree
  """
  grouped_zones_paths = IE.find_connected_zones(tree)

  for i, zones_path in enumerate(grouped_zones_paths):
    base = zones_path[0].split('/')[0]
    merge_zones(tree, zones_path, comm, output_path=f'{base}/mergedZone{i}', **kwargs)

def merge_zones(tree, zones_path, comm, output_path=None, subset_merge='name', concatenate_jns=True):
  """
  Highlevel function to merge some zones of a cgns dist tree.
  Zone to merge are specified by their path and must be unstructured.
  Output zone will be placed under first base with name "MergedZone", unless if output_path
  (base/zone) is specified
  Input zones will be removed from the tree

  Option concatenate_jns, if True, reduce the multiple 1to1 matching joins from or to merged_zone
  to a single one.
  """
  #Those one will be needed for jn recovering
  AJO.add_joins_ordinal(tree, comm)

  #Force full donor name, otherwise it is hard to reset jns
  IE.enforceDonorAsPath(tree)

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
  #Since merged zone have been removed, ordinal are unique
  merged_zone_path = I.getName(output_base) + '/' + I.getName(merged_zone)
  ordinal_to_pl = {}
  for gc in IE.getNodesByMatching(tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', 'GridConnectivity_t']):
    ordinal_to_pl[I.getNodeFromName1(gc, 'Ordinal')[1][0]] = \
        (I.getNodeFromName1(gc, 'PointList')[1], I.getNodeFromName1(gc, 'PointListDonor')[1], IE.getDistribution(gc))

  for base, zone in IE.getNodesWithParentsByMatching(tree, ['CGNSBase_t', 'Zone_t']):
    is_merged_zone = f"{I.getName(base)}/{I.getName(zone)}" == merged_zone_path
    for gc in IE.getNodesByMatching(zone, ['ZoneGridConnectivity_t', 'GridConnectivity_t']):
      #Update name and PL
      if I.getValue(gc) in zones_path:
        I.setValue(gc, merged_zone_path)
        jn_ord = I.getNodeFromName1(gc, 'Ordinal')[1][0]
        key    = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]
        # Copy and permute pl/pld only for all the zones != merged zone OR for one gc over two for
        # merged zone
        if not is_merged_zone or key < jn_ord:
          I.newIndexArray('PointList'     , ordinal_to_pl[key][1], parent=gc)
          I.newIndexArray('PointListDonor', ordinal_to_pl[key][0], parent=gc)
          I._rmNodesByName(gc, ":CGNS#Distribution")
          I._addChild(gc, ordinal_to_pl[key][2])

  if concatenate_jns:
    GN.concatenate_jns(tree, comm)

  #Finally, ordinals can be removed
  AJO.rm_joins_ordinal(tree)



def _add_zone_suffix(zones, query):
  """Util function prefixing all the nodes founds by a query by the number of the zone"""
  for izone, zone in enumerate(zones):
    for node in IE.getNodesByMatching(zone, query):
      I.setName(node, I.getName(node) + f".{izone}")
def _rm_zone_suffix(zones, query):
  """Util function removing the number of the zone in all the nodes founds by a query"""
  for zone in zones:
    for node in IE.getNodesByMatching(zone, query):
      I.setName(node, '.'.join(I.getName(node).split('.')[:-1]))

def _merge_zones(tree, comm, subset_merge_strategy='name'):
  """
  Tree must contain *only* the zones to merge. We use a tree instead of a list of zone because it's easier
  to retrieve opposites zones througt joins. Interface beetween zones shall be described by faces
  """

  zones_path = [f'{I.getName(base)}/{I.getName(zone)}' for base, zone in \
      IE.getNodesWithParentsByMatching(tree, ['CGNSBase_t', 'Zone_t'])]
  n_zone = len(zones_path)
  zones = I.getZones(tree)
  assert min([sids.Zone.Type(zone) == 'Unstructured' for zone in zones]) == True

  zone_to_id = {path : i for i, path in enumerate(zones_path)}

  is_perio = lambda n : I.getNodeFromType1(n, 'GridConnectivityProperty_t') is not None
  query = ['ZoneGridConnectivity_t', \
      lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] 
                and sids.GridLocation(n) == 'FaceCenter']

  # JNs to external zones must be excluded from vertex list computing
  tree_vl = I.copyRef(tree)
  for base, zone in IE.getNodesWithParentsByMatching(tree_vl, ['CGNSBase_t', 'Zone_t']):
    for zgc, gc in IE.getNodesWithParentsByMatching(zone, query):
      if IE.getZoneDonorPath(I.getName(base), gc) not in zone_to_id:
        I.rmNode(zgc, gc)
  VL.generate_jns_vertex_list(tree_vl, comm)
  #Reput in tree
  for zone_path in zones_path:
    zone    = I.getNodeFromPath(tree, zone_path)
    zone_vl = I.getNodeFromPath(tree_vl, zone_path)
    for zgc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
      zgc_v = I.getNodeFromName1(zone_vl, f"{I.getName(zgc)}#Vtx")
      I._addChild(zone, zgc_v)
  
  # Collect interface data
  interface_dn_f = []
  interface_ids_f = []
  interface_dom = []
  interface_dn_v = []
  interface_ids_v = []
  for zone_path, zone in zip(zones_path, zones):
    base_name, zone_name = zone_path.split('/')
    for zgc, gc in IE.getNodesWithParentsByMatching(zone, query):
      jn_ordinal     = I.getNodeFromName1(gc, 'Ordinal')[1][0]
      jn_ordinal_opp = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]
      opp_zone_path = IE.getZoneDonorPath(base_name, gc)
      if opp_zone_path in zone_to_id:
        if is_perio(gc):
          I.newUserDefinedData('__maia_jn_update__', value = zone_to_id[opp_zone_path], parent=gc)
        else:
          I.newDescriptor('__maia_merge__', parent=gc)
      if I.getNodeFromName1(gc, '__maia_merge__') is not None and jn_ordinal < jn_ordinal_opp:
        interface_dom.append((zone_to_id[zone_path], zone_to_id[opp_zone_path]))

        pl  = I.getNodeFromName1(gc, 'PointList')[1][0]
        pld = I.getNodeFromName1(gc, 'PointListDonor')[1][0]

        interface_dn_f.append(pl.size)
        interface_ids_f.append(py_utils.interweave_arrays([pl,pld]))

        # Find corresponding vertex
        gc_vtx = I.getNodeFromPath(zone, f'{I.getName(zgc)}#Vtx/{I.getName(gc)}#Vtx')
        pl_v  = I.getNodeFromName1(gc_vtx, 'PointList')[1][0]
        pld_v = I.getNodeFromName1(gc_vtx, 'PointListDonor')[1][0]
        interface_dn_v.append(pl_v.size)
        interface_ids_v.append(py_utils.interweave_arrays([pl_v,pld_v]))

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
        distri = IE.getDistribution(sids.Zone.NGonNode(zone), 'Element')[1].astype(pdm_dtype)
      else:
        distri = IE.getDistribution(zone, entity)[1].astype(pdm_dtype)
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


  loc_without_pl = lambda n, loc : sids.GridLocation(n) == loc and I.getNodeFromName(n, 'PointList') is None
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

  IE.newDistribution({'Vertex' : par_utils.full_to_partial_distribution(merged_distri_vtx, comm),
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
      for node, data in IE.getNodesWithParentsByMatching(zone, query + ['DataArray_t']):
        dic_path = I.getName(node) + '/' + I.getName(data)
        try:
          to_merge[dic_path].append(data[1])
        except KeyError:
          to_merge[dic_path] = [ data[1] ]
    
  merged = {key : mbm.merge_field(datas) for key, datas in to_merge.items()}


  additional_types = ['GridLocation_t', 'Descriptor_t', 'DataClass_t', 'DimensionalUnits_t']
  for query in data_queries:
    #Use zone 0 to get node type and value. Nodes must be know in every zone
    for node in IE.getNodesByMatching(zones[0], query):
      m_node = I.createUniqueChild(merged_zone, I.getName(node), I.getType(node), I.getValue(node))
      for data in I.getNodesFromType1(node, 'DataArray_t'):
        I.newDataArray(data[0], merged[I.getName(node) + '/' + I.getName(data)], parent=m_node)
      for type in additional_types:
        for sub_node in I.getNodesFromType1(node, type):
          I._addChild(m_node, sub_node)

def _merge_pls_data(all_mbm, zones, merged_zone, comm, merge_strategy='name'):
  """
  Wrapper to perform a merge off the following subset nodes (when having a PointList) :
    FlowSolution_t, DiscreteData_t, ZoneSubRegion_t, BC_t, GridConnectivity_t, BCDataSet_t
  from the input zones to the merged zone merged_zone.

  If merged_strategy=='name', the nodes subset set having the same name on different zones are merged
  into a single one.
  Otherwise, force to keep one subset_node per input zone to let the user manage its merge.
  Merging by name is not performed for GridConnectivity_t
  """
  #In each case, we need to collect all the nodes, since some can be absent of a given zone
  has_pl = lambda n : I.getNodeFromName1(n, 'PointList') is not None
  jn_to_keep = lambda n : I.getType(n) == 'GridConnectivity_t' and sids.GridLocation(n) == 'FaceCenter'\
      and I.getNodeFromName1(n, '__maia_merge__') is None

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
    for zsr in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
      #Copy PL when related to bc/gc to avoid specific treatement
      if I.getNodeFromName1(zsr, 'BCRegionName') is not None or \
         I.getNodeFromName1(zsr, 'GridConnectivityRegionName') is not None:
        related = I.getNodeFromPath(zone, IE.getSubregionExtent(zsr, zone))
        I._addChild(zsr, I.getNodeFromName1(related, 'PointList'))

  for query, rules in zip(all_subset_queries, all_data_queries):
    if merge_strategy != 'name' or query[0] == 'ZoneGridConnectivity_t':
      _add_zone_suffix(zones, query)
    collected_paths = []
    #Collect
    for zone in zones:
      for nodes in IE.getNodesWithParentsByMatching(zone, query):
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
      location = sids.GridLocation(I.getNodeFromPath(master_node, pl_path))
      mbm = all_mbm[location.split('Center')[0]]
      merged_pl = _merge_pl_data(mbm, zones, pl_path, location, rules, comm)
      #Rebuild structure until last node
      for child_name in pl_path.split('/')[:-1]:
        master_node = I.getNodeFromName1(master_node, child_name)
        parent = I.createUniqueChild(parent, child_name, I.getType(master_node), I.getValue(master_node))

      I._addChild(parent, merged_pl)
    if merge_strategy != 'name'or query[0] == 'ZoneGridConnectivity_t':
      _rm_zone_suffix(zones, query)

  # Trick to avoid spectific treatment of ZoneSubRegions (remove PL)
  for zone in zones + [merged_zone]:
    for zsr in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
      if I.getNodeFromName1(zsr, 'BCRegionName') is not None or \
         I.getNodeFromName1(zsr, 'GridConnectivityRegionName') is not None:
        I._rmNodesByName(zsr, 'PointList*')

def _equilibrate_data(data, comm, distri=None, distri_full=None):
  from maia.distribution import distribution_function as DF
  if distri_full is None:
    if distri is None:
      first = next(iter(data.values()))
      distri_full = par_utils.gather_and_shift(first.size, comm, pdm_dtype)
    else:
      distri_full = par_utils.partial_to_full_distribution(distri, comm)

  ideal_distri = DF.uniform_distribution(distri_full[-1], comm)
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
      distri_ptb = IE.getDistribution(zone, 'Vertex')[1]
    elif loc == 'FaceCenter':
      distri_ptb = IE.getDistribution(sids.Zone.NGonNode(zone), 'Element')[1]
    elif loc == 'CellCenter':
      distri_ptb = IE.getDistribution(zone, 'Cell')[1]
    if node is not None:
      ref_node = node #Take any node as reference, to build name/type/value of merged node

      pl = I.getNodeFromName1(node, 'PointList')[1][0]
      part_data = {'PL' : [pl]}
      for nodes in IE.getNodesWithParentsByMatching(node, data_query):
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
  if I.getNodeFromName(ref_node, '__maia_jn_update__') is not None:
    opp_dom = I.getNodeFromName(ref_node, '__maia_jn_update__')[1][0]
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

  for nodes in IE.getNodesWithParentsByMatching(ref_node, data_query):
    path =  '/'.join([I.getName(node) for node in nodes])
    # #Rebuild structure if any
    sub_ref = ref_node
    merged_parent = merged_node
    for node in nodes[:-1]:
      sub_ref = I.getNodeFromName1(sub_ref, I.getName(node))
      merged_parent = I.createUniqueChild(merged_parent, I.getName(sub_ref), I.getType(sub_ref), I.getValue(sub_ref))
    if I.getType(nodes[-1]) == 'IndexArray_t':
      I.newIndexArray(I.getName(nodes[-1]), merged_data[path].reshape((1,-1), order='F'), merged_parent)
    else:
      I.newDataArray(I.getName(nodes[-1]), merged_data[path], merged_parent)

  additional_types = ['GridLocation_t', 'FamilyName_t', 'Descriptor_t',
                      'GridConnectivityType_t', 'GridConnectivityProperty_t']
  additional_names = ['Ordinal', 'OrdinalOpp']
  for type in additional_types:
    for sub_node in I.getNodesFromType1(ref_node, type):
      I._addChild(merged_node, sub_node)
  for name in additional_names:
    for sub_node in I.getNodesFromName1(ref_node, name):
      I._addChild(merged_node, sub_node)

  IE.newDistribution({'Index' : merged_pl_distri}, merged_node)

  return merged_node

def _merge_ngon(all_mbm, tree, comm):
  """
  Internal function used by _merge_zones to create the merged NGonNode
  """

  zones_path = [f'{I.getName(base)}/{I.getName(zone)}' for base, zone in \
      IE.getNodesWithParentsByMatching(tree, ['CGNSBase_t', 'Zone_t'])]
  zone_to_id = {path : i for i, path in enumerate(zones_path)}

  # Create working data
  for zone_path, dom_id in zone_to_id.items():
    ngon_node = sids.Zone.NGonNode(I.getNodeFromPath(tree, zone_path))
    pe = I.getNodeFromPath(ngon_node, 'ParentElements')[1]
    I.newDataArray('UpdatedPE', pe.copy(), parent=ngon_node)
    I.newDataArray('PEDomain',  dom_id * np.ones_like(pe), parent=ngon_node)

  # First, we need to update the PE node to include cells of opposite zone
  query = lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
                and I.getNodeFromName1(n, '__maia_merge__') is not None

  for zone_path_send in zones_path:
    base_n = zone_path_send.split('/')[0]
    dom_id_send = zone_to_id[zone_path_send]
    zone_send = I.getNodeFromPath(tree, zone_path_send)
    ngon_send = sids.Zone.NGonNode(zone_send)
    face_distri_send = IE.getDistribution(ngon_send, 'Element')[1].astype(pdm_dtype)
    pe_send          =  I.getNodeFromPath(ngon_send, 'UpdatedPE')[1]
    dist_data_send = {'PE' : pe_send[:,0]}

    gcs = PT.get_nodes_from_predicate(zone_send, query, depth=2)
    all_pls = [I.getNodeFromName1(gc, 'PointList')[1][0] for gc in gcs]
    part_data = MBTP.dist_to_part(face_distri_send, dist_data_send, all_pls, comm)
    for i, gc in enumerate(gcs):

      pld = I.getNodeFromName1(gc, 'PointListDonor')[1][0]

      #This is the left cell of the join face present in PL. Send it to opposite zone
      part_data_gc = {key : [data[i]] for key, data in part_data.items()}
      part_data_gc['FaceId'] = [pld]
    
      # Get send data on the opposite zone and update PE
      zone_path = IE.getZoneDonorPath(base_n, gc)
      zone = I.getNodeFromPath(tree, zone_path)
      ngon_node = sids.Zone.NGonNode(zone)
      face_distri = IE.getDistribution(ngon_node, 'Element')[1].astype(pdm_dtype)
      dist_data = MPTB.part_to_dist(face_distri, part_data_gc, [pld], comm)

      pe      = I.getNodeFromPath(ngon_node, 'UpdatedPE')[1]
      pe_dom  = I.getNodeFromPath(ngon_node, 'PEDomain')[1]
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
    eso    = I.getNodeFromName(ngon_node, 'ElementStartOffset')[1]
    pe     = I.getNodeFromName(ngon_node, 'UpdatedPE')[1]
    pe_dom = I.getNodeFromName(ngon_node, 'PEDomain')[1]

    ec_l.append(I.getNodeFromName(ngon_node, 'ElementConnectivity')[1])
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
  eso_loc = py_utils.sizes_to_indices(merged_ec_stri, pdm_dtype)
  ec_distri = par_utils.gather_and_shift(eso_loc[-1], comm)
  eso = eso_loc + ec_distri[comm.Get_rank()]

  #Post treat PE : we need to reintroduce 0 on boundary faces (TODO : could avoir tmp array ?)
  bnd_faces = np.where(merged_pe_stri == 1)[0]
  merged_pe_idx  = py_utils.sizes_to_indices(merged_pe_stri)
  merged_pe_full = np.insert(merged_pe, merged_pe_idx[bnd_faces]+1, 0)
  assert (merged_pe_full.size == 2*merged_pe_stri.size)
  pe = np.empty((merged_pe_stri.size, 2), order='F', dtype=merged_pe.dtype)
  pe[:,0] = merged_pe_full[0::2]
  pe[:,1] = merged_pe_full[1::2]

  # Finally : create ngon node
  merged_ngon = I.newElements('NGonElements', 'NGON', erange=[1, merged_distri_face[-1]])
  I.newDataArray('ElementStartOffset',  eso,       parent=merged_ngon)
  I.newDataArray('ElementConnectivity', merged_ec, parent=merged_ngon)
  I.newDataArray('ParentElements',      pe,        parent=merged_ngon)
  IE.newDistribution({'Element' :             par_utils.full_to_partial_distribution(merged_distri_face, comm),
                      'ElementConnectivity' : par_utils.full_to_partial_distribution(ec_distri, comm)},
                      merged_ngon)
  return merged_ngon
  
