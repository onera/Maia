import numpy as np

import Converter.Internal as I

from Pypdm import Pypdm as PDM

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.distribution import distribution_function as DIF

from maia.sids import Internal_ext as IE
from maia.sids import sids
from maia.sids import elements_utils as EU
from maia.utils import py_utils
from maia.utils.parallel import utils as par_utils
from maia.transform.dist_tree import add_joins_ordinal as AJO
from maia.connectivity import merge_jn
from maia.generate import nodes as GN


def _find_connected_zones(tree):
  """
  Define groups of independant zones
  """
  connected_zones = []
  matching_gcs_u = lambda n : I.getType(n) == 'GridConnectivity_t' \
                          and I.getValue(I.getNodeFromType1(n, 'GridConnectivityType_t')) == 'Abutting1to1'
  matching_gcs_s = lambda n : I.getType(n) == 'GridConnectivity1to1_t'
  matching_gcs = lambda n : (matching_gcs_u(n) or matching_gcs_s(n)) \
                          and I.getNodeFromType1(n, 'GridConnectivityProperty_t') is None
  
  for base, zone in IE.getNodesWithParentsByMatching(tree, 'CGNSBase_t/Zone_t'):
    zone_path = I.getName(base) + '/' + I.getName(zone)
    group     = [zone_path]
    for gc in IE.getNodesByMatching(zone, ['ZoneGridConnectivity_t', matching_gcs]):
      opp_zone_path = IE.getZoneDonorPath(I.getName(base), gc)
      py_utils.append_unique(group, opp_zone_path)
    connected_zones.append(group)

  for base, zone in IE.getNodesWithParentsByMatching(tree, 'CGNSBase_t/Zone_t'):
    zone_path     = I.getName(base) + '/' + I.getName(zone)
    groups_to_merge = []
    for i, group in enumerate(connected_zones):
      if zone_path in group:
        groups_to_merge.append(i)
    if groups_to_merge != []:
      new_group = []
      for i in groups_to_merge[::-1]: #Reverse loop to pop without changing idx
        zones_paths = connected_zones.pop(i)
        for z_p in zones_paths:
          py_utils.append_unique(new_group, z_p)
      connected_zones.append(new_group)
  return connected_zones

def merge_connected_zones(tree, comm, **kwargs):
  """
  Shortcut for merge_zones to merge all the zones of the tree
  """
  # zones_path = []
  # for base, zone in IE.getNodesWithParentsByMatching(tree, ['CGNSBase_t', 'Zone_t']):
    # zones_path.append(I.getName(base) + '/' + I.getName(zone))
  grouped_zones_paths = _find_connected_zones(tree)

  for i, zones_path in enumerate(grouped_zones_paths):
    base = zones_path[0].split('/')[0]
    merge_zones(tree, zones_path, comm, output_path=f'{base}/mergedZone{i}', **kwargs)

def merge_zones(tree, zones_path, comm, output_path=None, remove_internal_jns=True, concatenate_jns=True):
  """
  Highlevel function to merge some zones of a cgns dist tree.
  Zone to merge are specified by their path and must be unstructured.
  Output zone will be placed under first base with name "MergedZone", unless if output_path
  (base/zone) is specified
  Input zones will be removed from the tree

  Option remove_internal_jns, if True, remove all the duplicated faces obtained after merging
  the zone (intra zone jns)
  Option concatenate_jns, if True, reduce the multiple 1to1 matching joins from or to merged_zone
  to a single one.
  """
  #Those one will be needed for jn recovering
  AJO.add_joins_ordinal(tree, comm)

  #Merge zones
  zones = [I.getNodeFromPath(tree, zone_path) for zone_path in zones_path]
  merged_zone = _merge_zones(zones, comm)

  # Remove input zones and add output in tree
  for zone_path in zones_path:
    I._rmNodeByPath(tree, zone_path)

  if output_path is None:
    output_base = I.getNodeFromPath(tree, zones_path[0].split('/')[0])
  else:
    output_base = I.getNodeFromPath(tree, output_path.split('/')[0])
    I.setName(merged_zone, output_path.split('/')[1])
  I._addChild(output_base, merged_zone)

  #First we have to retrieve PLDonor for external jn and update opposite zones
  #Since merged zone have been removed, ordinal are unique
  merged_zone_path = I.getName(output_base) + '/' + I.getName(merged_zone)
  ordinal_to_pl = {}
  for gc in IE.getNodesByMatching(tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', 'GridConnectivity_t']):
    ordinal_to_pl[I.getNodeFromName1(gc, 'Ordinal')[1][0]] = I.getNodeFromName1(gc, 'PointList')[1]

  for base in I.getBases(tree):
    for gc in IE.getNodesByMatching(base, ['Zone_t', 'ZoneGridConnectivity_t', 'GridConnectivity_t']):
      #Update name
      if IE.getZoneDonorPath(I.getName(base), gc) in zones_path:
        I.setValue(gc, merged_zone_path)
      #Update PL
      I.newIndexArray('PointListDonor', ordinal_to_pl[I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]], parent=gc)

  #Now we can merge internal jns (PLDonor are needed for that)
  internal_jns = lambda n: I.getType(n) == 'GridConnectivity_t' and I.getValue(n) == merged_zone_path \
                           and I.getNodeFromType(n, 'GridConnectivityProperty_t') is None\
                           and I.getValue(I.getNodeFromType(n, 'GridConnectivityType_t')) == 'Abutting1to1'
  #Find pairs of joins
  jns_pathes = dict()
  for zgc, jn in IE.getNodesWithParentsByMatching(merged_zone, ['ZoneGridConnectivity_t', internal_jns]):
    key = min(I.getNodeFromName(jn, 'Ordinal')[1][0], I.getNodeFromName(jn, 'OrdinalOpp')[1][0])
    jn_path = '/'.join([merged_zone_path, I.getName(zgc), I.getName(jn)])
    try:
      jns_pathes[key].append(jn_path)
    except KeyError:
      jns_pathes[key] = [jn_path]
  jns_pathes = [val for val in jns_pathes.values()] #Key are useless now

  # Merging all intrazone jns before merging is unsafe : vertex list generation will fail if three (or more) vertices
  # join at the same point. This could be predicted if we had a global numbering on vertices, but for now we just 
  # merge the jns two by two (wich is safe and divide the number of fusions by two)
  # In addition, we should check the location
  allow_two = False
  if remove_internal_jns and allow_two:
    jns_pathes_cat = []
    i = 0
    while jns_pathes:
      jn_pathes_1 = jns_pathes.pop()
      try:
        jn_pathes_2 = jns_pathes.pop()
        gcs_A = [I.getNodeFromPath(tree, jn_pathes_1[0]), I.getNodeFromPath(tree, jn_pathes_2[0])]
        gcs_B = [I.getNodeFromPath(tree, jn_pathes_1[1]), I.getNodeFromPath(tree, jn_pathes_2[1])]
        assert sids.GridLocation(gcs_A[0]) == sids.GridLocation(gcs_A[1]), "Multiple locations are not yet supported"
        mergedA = GN.concatenate_subset_nodes(gcs_A, comm, output_name=f'IntraZoneJnA_{i}', \
            additional_child_queries=['GridConnectivityType_t'])
        mergedB = GN.concatenate_subset_nodes(gcs_B, comm, output_name=f'IntraZoneJnB_{i}', \
            additional_child_queries=['GridConnectivityType_t'])
        for path in jn_pathes_1 + jn_pathes_2:
          I._rmNodeByPath(tree, path)

        I._addChild(zgc, mergedA)
        I._addChild(zgc, mergedB)
        jns_pathes_cat.append((merged_zone_path + '/' + I.getName(zgc) + f'/IntraZoneJnA_{i}', 
                               merged_zone_path + '/' + I.getName(zgc) + f'/IntraZoneJnB_{i}'))
      except IndexError: #When jns_pathes is even, we can not pop 2 times for last element
        jns_pathes_cat.append(jn_pathes_1)
      i += 1

    jns_pathes = jns_pathes_cat
    #If we concatenate, ordinal are outdated, we must udpate it
    AJO.rm_joins_ordinal(tree)
  AJO.add_joins_ordinal(tree, comm)

  if remove_internal_jns:
    for jn_pathes in jns_pathes:
      merge_jn.merge_intrazone_jn(tree, jn_pathes, comm)

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

def _merge_zones(zones, comm):
  """
  Merge the input zones and return the merged with updated distributions, connectivities, subset and data.
  All the releveant data should be updated, exepted GridConnectivity_t for which donor data is not recovered
  here (PointListDonor and DonorName)
  Only unstructured zones are supported
  """
  assert min([sids.Zone.Type(zone) == 'Unstructured' for zone in zones]) == True

  # Compute merge zone cell and vertex distributions
  n_cell_tot = sum([sids.Zone.n_cell(zone) for zone in zones])
  n_vtx_tot  = sum([sids.Zone.n_vtx(zone) for zone in zones])

  merged_zone = I.newZone('MergedZone', [[n_vtx_tot, n_cell_tot, 0]], ztype='Unstructured')
  DIF.create_distribution_node(n_vtx_tot,  comm, 'Vertex', merged_zone)
  DIF.create_distribution_node(n_cell_tot, comm, 'Cell',   merged_zone)

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
  _merge_allmesh_data(zones, merged_zone, 'Vertex', vtx_data_queries, comm)
  _merge_allmesh_data(zones, merged_zone, 'CellCenter', cell_data_queries, comm)

  # Connectivity
  _merge_ngon(zones, merged_zone, comm)
  _merge_nface(zones, merged_zone, comm)
  
  #Shift nface 
  if I.getNodeFromPath(merged_zone, 'NFaceElements') is not None:
    ngon_tot = sids.ElementSize(I.getNodeFromPath(merged_zone, 'NGonElements'))
    I.getNodeFromPath(merged_zone, 'NFaceElements/ElementRange')[1] += ngon_tot

  _merge_pls_data(zones, merged_zone, comm)

  return merged_zone

def _merge_allmesh_data(zones, merged_zone, location, data_queries, comm):
  """
  Merge the all DataArray supported by allCells or allVertex (depending on location),
  found under each of data_query (query must start from zone node), from input zones
  to merged_zone.
  """
  assert location in ['Vertex', 'CellCenter']
  distri_name = 'Cell' if location == 'CellCenter' else 'Vertex'

  offset = 0
  ln_to_gn = list()
  for zone in zones:
    distri = I.getVal(IE.getDistribution(zone, distri_name))
    ln_to_gn.append(np.arange(distri[0], distri[1], dtype=pdm_dtype) + offset + 1)
    offset += distri[2]

  partial_distri = I.getVal(IE.getDistribution(merged_zone, distri_name)).astype(pdm_dtype)
  pdm_distrib = par_utils.partial_to_full_distribution(partial_distri, comm)
  PTB = PDM.PartToBlock(comm, ln_to_gn, None, len(zones), 0, 0, 0, userDistribution=pdm_distrib)

  pField  = dict()
  
  for query in data_queries:
    for zone in zones:
      #For global data, we should have only one parent
      for node, data in IE.getNodesWithParentsByMatching(zone, query + ['DataArray_t']):
        dic_path = I.getName(node) + '/' + I.getName(data)
        try:
          pField[I.getName(node) + '/' + I.getName(data)].append(data[1])
        except KeyError:
          pField[I.getName(node) + '/' + I.getName(data)] = [ data[1] ]
    
  dField = dict()
  PTB.PartToBlock_Exchange(dField, pField)

  additional_types = ['GridLocation_t', 'Descriptor_t', 'DataClass_t', 'DimensionalUnits_t']
  for query in data_queries:
    #Use zone 0 to get node type and value. Nodes must be know in every zone
    for node in IE.getNodesByMatching(zones[0], query):
      m_node = I.createUniqueChild(merged_zone, I.getName(node), I.getType(node), I.getValue(node))
      for data in I.getNodesFromType1(node, 'DataArray_t'):
        I.newDataArray(data[0], dField[I.getName(node) + '/' + I.getName(data)], parent=m_node)
      for type in additional_types:
        for sub_node in I.getNodesFromType1(node, type):
          I._addChild(m_node, sub_node)

def _merge_pls_data(zones, merged_zone, comm, merge_strategy='name'):
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

  #Order : FlowSolution/DiscreteData/ZoneSubRegion, BC, BCDataSet, GridConnectivity_t, 
  all_subset_queries = [
      [lambda n : I.getType(n) in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t'] and has_pl(n)],
      ['ZoneBC_t', 'BC_t'],
      ['ZoneBC_t', 'BC_t', lambda n : I.getType(n) == 'BCDataSet_t' and has_pl(n)],
      ['ZoneGridConnectivity_t', 'GridConnectivity_t'],
      ]

  all_data_queries = [
      [ [] ],
      [ [lambda n : I.getType(n) == 'BCDataSet_t' and not has_pl(n), 'BCData_t'] ],
      [['BCData_t']],
      [ ],
      ]

  # Precompute offsets only once
  ngons   = [sids.Zone.NGonNode(zone) for zone in zones]
  nvtx_per_zone  = np.array([sids.Zone.n_vtx(zone)  for zone in zones])
  nface_per_zone = np.array([sids.ElementSize(ngon) for ngon in ngons])
  ncell_per_zone = np.array([sids.Zone.n_cell(zone) for zone in zones])
  vtx_offset  = py_utils.sizes_to_indices(nvtx_per_zone)
  #Face and Cell offset are for ngon only, standart elements not suppported
  face_offset = py_utils.sizes_to_indices(nface_per_zone)
  cell_offset = py_utils.sizes_to_indices(ncell_per_zone) + sum(nface_per_zone) 
  cell_offset[:-1 ] += -nface_per_zone #Last idx is not managed, but is ok
  offsets = {'Vertex' : vtx_offset, 'FaceCenter' : face_offset, 'CellCenter' : cell_offset}

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
      merged_pl = _merge_pl_data(zones, pl_path, offsets[location], rules, comm)
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


def _merge_pl_data(zones, subset_path, loc_offset, data_queries, comm):
  """
  Generic function to produced a merged node from the zone to merge and the path to a
  node having a PointList
  PointList elements of output pl are shifted depending on their type (vtx, face, cell)
  (supports only NGon/NFace based zones)
  Subset nodes comming from different zones with the same path will be merged
  Also merge all the DataArray nodes found under each query (starting from subset_node)
  requested in data_queries list

  Return the merged subset node 
  """

  pl_offset   = 0

  ln_to_gn = list()
  p_data   = {'PointList' : list()}

  ref_node = None

  for i, zone in enumerate(zones):
    node = I.getNodeFromPath(zone, subset_path)
    if node is not None:
      ref_node = node #Take any node as reference, to build name/type/value of merged node

      distri = I.getVal(IE.getDistribution(node, 'Index'))
      ln_to_gn.append(np.arange(distri[0], distri[1], dtype=pdm_dtype) + pl_offset + 1)

      #Pl values must be shifted since elements to match new numbering of zone
      pl = I.getNodeFromName1(node, 'PointList')[1][0]
      p_data['PointList'].append(pl + loc_offset[i])

      #Add DataArrays for FS or ZSR (not shift is needed)
      for query in data_queries:
        for nodes in IE.getNodesWithParentsByMatching(node, query + ['DataArray_t']):
          path =  '/'.join([I.getName(node) for node in nodes])
          data_n = nodes[-1]
          try:
            p_data[path].append(data_n[1])
          except KeyError:
            p_data[path] = [data_n[1]]

      pl_offset += distri[2]

  PTB = PDM.PartToBlock(comm, ln_to_gn, None, len(ln_to_gn), 0, 0, 0)
  pdm_distri = PTB.getDistributionCopy()
  merged_pl_distri = pdm_distri[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

  d_data  = dict()
  PTB.PartToBlock_Exchange(d_data, p_data)

  #Creation of node
  merged_node = I.createNode(I.getName(ref_node), I.getType(ref_node), I.getValue(ref_node))
  I.newIndexArray('PointList', d_data['PointList'].reshape(1, -1, order='F'), parent=merged_node)

  for query in data_queries:
    for nodes in IE.getNodesWithParentsByMatching(ref_node, query + ['DataArray_t']):
      path =  '/'.join([I.getName(node) for node in nodes])
      #Rebuild structure if any
      sub_ref = ref_node
      merged_parent = merged_node
      for node in nodes[:-1]:
        sub_ref = I.getNodeFromName1(sub_ref, I.getName(node))
        merged_parent = I.createUniqueChild(merged_parent, I.getName(sub_ref), I.getType(sub_ref), I.getValue(sub_ref))
      I.newDataArray(I.getName(nodes[-1]), d_data[path], merged_parent)

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

def _merge_ngon(zones, merged_zone, comm):
  """
  Create a new NGon Element_t node on the merged zone from the NGon nodes of zones
  to merge.
  Face->vtx connectivity and PE are updated to match the vertices of the merged zone
  """
  vtx_offset = 0
  ngon_offset = 0
  cell_offset = 0
  pLNToGN = []
  p_data_pe = {'PE' : list()}
  p_data_ec = {'EC' : list()}
  p_strid_ec = list()
  p_strid_pe = list()
  for zone in zones:
    ngon   = sids.Zone.NGonNode(zone)
    ngon_distri = I.getVal(IE.getDistribution(ngon, 'Element'))
    vtx_distri  = I.getVal(IE.getDistribution(zone, 'Vertex'))
    cell_distri = I.getVal(IE.getDistribution(zone, 'Cell'))

    pLNToGN.append(np.arange(ngon_distri[0], ngon_distri[1], dtype=pdm_dtype) + ngon_offset + 1)

    PE  = I.getNodeFromName1(ngon, 'ParentElements')[1]
    PE += cell_offset * (PE > 0)
    EC    = I.getNodeFromName1(ngon, 'ElementConnectivity')[1] + vtx_offset
    ECIdx = I.getNodeFromName1(ngon, 'ElementStartOffset')[1]

    p_data_pe['PE'].append(PE.ravel())
    p_data_ec['EC'].append(EC)
    p_strid_pe.append(2*np.ones(PE.shape[0], dtype=np.int32))
    p_strid_ec.append(np.diff(ECIdx).astype(np.int32))

    vtx_offset  += vtx_distri[2]
    cell_offset += cell_distri[2]
    ngon_offset += ngon_distri[2]

  PTB = PDM.PartToBlock(comm, pLNToGN, None, len(zones), 0, 0, 1)
  pdm_distri = PTB.getDistributionCopy()
  merged_ngon_distri = pdm_distri[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

  d_data_ec = dict()
  d_data_pe = dict()
  PTB.PartToBlock_Exchange(d_data_ec, p_data_ec, p_strid_ec)
  PTB.PartToBlock_Exchange(d_data_pe, p_data_pe, p_strid_pe)

  #Do it by hand to have good ordering (reshape gives C ordering)
  dist_pe = np.empty([d_data_pe['PE'].shape[0]//2, 2], order='F', dtype=np.int32)
  dist_pe[:,0] = d_data_pe['PE'][0::2]
  dist_pe[:,1] = d_data_pe['PE'][1::2]
  assert (d_data_pe['PE'].reshape(-1, 2, order='A') == dist_pe).all()

  #Recompute ESO and ElementConnectivity distribution
  eso_unshifted = py_utils.sizes_to_indices(d_data_ec['EC#Stride'])
  shift_for_eso = par_utils.gather_and_shift(eso_unshifted[-1], comm)
  eso = eso_unshifted + shift_for_eso[comm.Get_rank()]

  merged_eso_distri = np.array([eso[0], eso[-1], shift_for_eso[-1]], dtype=pdm_dtype)

  #Now build NGon node
  elt_node = I.newElements('NGonElements', 'NGON', parent=merged_zone)
  I.newPointRange('ElementRange',        [1, pdm_distri[-1]], parent=elt_node)
  I.newDataArray ('ElementStartOffset',  eso,                 parent=elt_node)
  I.newDataArray ('ElementConnectivity', d_data_ec['EC'],     parent=elt_node)
  I.newDataArray ('ParentElements',      dist_pe,             parent=elt_node)

  IE.newDistribution({'Element' : merged_ngon_distri, 'ElementConnectivity' : merged_eso_distri}, elt_node)
  
def _merge_nface(zones, merged_zone, comm):
  """
  Create a new NFace Element_t node on the merged zone from the NFace nodes of zones
  to merge.
  Cell->face connectivity is updated to match the face ids of the merged zone
  NFace output node is provided with ElementRange from 1 to nCell and should be shifted
  afterward if needed.
  """
  has_nface = True
  for zone in zones:
    nface = [elem for elem in I.getNodesFromType1(zone,   'Elements_t') if elem[1][0] == 23]
    has_nface = len(nface) > 0 and has_nface
  if not has_nface:
    return

  ngon_offset  = 0
  nface_offset = 0
  pLNToGN = []
  p_data_ec = {'EC' : list()}
  p_strid_ec = list()
  for zone in zones:
    ngon   = sids.Zone.NGonNode(zone)
    nface  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NFACE_n'][0]

    ngon_distri  = I.getVal(IE.getDistribution(ngon, 'Element'))
    nface_distri = I.getVal(IE.getDistribution(nface, 'Element'))

    pLNToGN.append(np.arange(nface_distri[0], nface_distri[1], dtype=pdm_dtype) + nface_offset + 1)

    EC    = I.getNodeFromName1(nface, 'ElementConnectivity')[1] + ngon_offset
    ECIdx = I.getNodeFromName1(nface, 'ElementStartOffset')[1]

    p_data_ec['EC'].append(EC)
    p_strid_ec.append(np.diff(ECIdx).astype(np.int32))

    nface_offset += nface_distri[2]
    ngon_offset  += ngon_distri[2]

  PTB = PDM.PartToBlock(comm, pLNToGN, None, len(zones), 0, 0, 1)
  pdm_distri = PTB.getDistributionCopy()
  merged_nface_distri = pdm_distri[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

  d_data_ec = dict()
  PTB.PartToBlock_Exchange(d_data_ec, p_data_ec, p_strid_ec)

  #Recompute ESO and ElementConnectivity distribution
  eso_unshifted = py_utils.sizes_to_indices(d_data_ec['EC#Stride'])
  shift_for_eso = par_utils.gather_and_shift(eso_unshifted[-1], comm)
  eso = eso_unshifted + shift_for_eso[comm.Get_rank()]

  merged_eso_distri = np.array([eso[0], eso[-1], shift_for_eso[-1]], dtype=pdm_dtype)

  #Now build NFace node
  elt_node = I.newElements('NFaceElements', 'NFACE', parent=merged_zone)
  I.newPointRange('ElementRange',        [1, pdm_distri[-1]], parent=elt_node)
  I.newDataArray ('ElementStartOffset',  eso,                 parent=elt_node)
  I.newDataArray ('ElementConnectivity', d_data_ec['EC'],     parent=elt_node)

  IE.newDistribution({'Element' : merged_nface_distri, 'ElementConnectivity' : merged_eso_distri}, elt_node)
