from mpi4py import MPI
import numpy as np
import itertools

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia       import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import par_utils

from maia.algo.dist           import remove_element    as RME
from maia.algo.dist           import matching_jns_tools as MJT
from maia.algo.dist.merge_ids import merge_distributed_ids
from maia.algo.dist           import vertex_list as VL

from maia.transfer import protocols as EP

HAS_POINTLIST = PT.pred.has_child_of_name('PointList')

def _update_ngon_exchange_PE(ngon, ref_faces, del_faces, comm):
  """
  Update ParentElements in ngon to combinate faces
  """
  face_distri = MT.distribution_value(ngon, 'Element')
  pe          = PT.get_node_from_path(ngon, 'ParentElements')[1]


  #TODO This method asserts that PE is CGNS compliant ie left_parent != 0 for bnd elements
  assert not np.any(pe[:,0] == 0)

  # 1. Get the left cell of the faces to delete
  part_data = EP.GlobalIndexer(face_distri, del_faces-1, comm).Take(pe[:,0])
  
  # 2. Put it in the right cell of the faces to keep
  GI = EP.GlobalIndexer(face_distri, ref_faces-1, comm)
  assert np.max(pe[GI.access_counts > 0, 1], initial=0) == 0 #Initial = trick to admit empty array
  GI.Put(part_data, pe[:,1])

def _update_ngon_remove_faces(ngon, del_faces, comm):
  """
  Remove faces from EC, PE and ESO and update distribution info in ngon
  """
  face_distri = MT.distribution_value(ngon, 'Element')
  
  GI = EP.GlobalIndexer(face_distri, del_faces-1, comm)
  local_faces = np.nonzero(GI.access_counts > 0)[0]
  RME.remove_ngons(ngon, local_faces, comm)
  
def _update_ngon_update_EC(ngon, vtx_distri_ini, old_to_new_vtx, comm):
  """
  Update ElementConnectivity using vertex old_to_new order in ngon
  """
  # C/ Update vertex ids in EC
  ngon_ec_n = PT.get_child_from_name(ngon, 'ElementConnectivity')
  part_data = EP.block_to_part(old_to_new_vtx, vtx_distri_ini, PT.get_value(ngon_ec_n)-1, comm)
  assert len(ngon_ec_n[1]) == len(part_data)
  PT.set_value(ngon_ec_n, part_data)

def _update_ngon(ngon, ref_faces, del_faces, vtx_distri_ini, old_to_new_vtx, comm):
  """
  Update ngon node after face and vertex merging, ie
   - update ParentElements to combinate faces
   - remove faces from EC, PE and ESO and update distribution info
   - update ElementConnectivity using vertex old_to_new order
  """

  # A/ Exchange parent cells before removing :
  _update_ngon_exchange_PE(ngon, ref_faces, del_faces, comm)

  # B/ Update EC, PE and ESO removing some faces
  _update_ngon_remove_faces(ngon, del_faces, comm)

  # C/ Update vertex ids in EC
  _update_ngon_update_EC(ngon, vtx_distri_ini, old_to_new_vtx, comm)

def _update_nface(nface, face_distri_ini, old_to_new_face, n_rmvd_face, comm):
  """
  Update nface node after face merging, ie
   - update ElementConnectivity using face old_to_new order
   - Shift ElementRange (to substract nb of removed faces) if NFace is after NGon
  If input array old_to_new_face is signed (ie is negative for face ids that will be removed),
  then the orientation of nface connectivity is preserved
  """

  #Update list of faces
  nface_ec_n = PT.get_child_from_name(nface, 'ElementConnectivity')
  part_data = EP.block_to_part(old_to_new_face, face_distri_ini, np.abs(nface_ec_n[1])-1, comm)
  assert len(nface_ec_n[1]) == len(part_data)
  #Get sign of nface_ec to preserve orientation
  PT.set_value(nface_ec_n, np.sign(nface_ec_n[1]) * part_data)

  #Update ElementRange
  er = PT.Element.Range(nface)
  if er[0] > 1:
    er -= n_rmvd_face

def _update_subset(node, pl_new, data_query, comm):
  """
  Update a PointList and all the data
  """
  part_data = {}
  dist_data = {}
  for data_nodes in PT.iter_children_from_predicates(node, data_query, ancestors=True):
    path = "/".join([PT.get_name(n) for n in data_nodes])
    data_n = data_nodes[-1]
    if data_n[1].ndim == 1:
      part_data[path] = data_n[1]
    else:
      assert data_n[1].ndim == 2 and data_n[1].shape[0] == 1
      part_data[path] = data_n[1][0]

  #Add PL, needed for next blocktoblock
  pl_identifier = r'@\PointList/@' # just a string that is unlikely to clash
  part_data[pl_identifier] = pl_new

  old_distri_f = par_utils.distribution_from_gnum(pl_new, comm, full=True)

  GI = EP.GlobalIndexer(old_distri_f, pl_new-1, comm)
  mask = (GI.access_counts > 0)
  dist_data = {field: GI.Put(pdata)[mask] for field, pdata in part_data.items()}
  
  new_distri_full = par_utils.gather_and_shift(mask.sum(), comm, pdm_dtype)
  #Result is badly distributed, we can do a BlockToBlock to have a uniform distribution
  ideal_distri      = par_utils.uniform_distribution(new_distri_full[-1], comm)
  dist_data_ideal = EP.block_to_block(dist_data, new_distri_full, ideal_distri, comm)

  #Update distribution and size
  MT.new_Distribution({'Index' : ideal_distri}, node)

  #Update PointList and data
  PT.update_child(node, 'PointList', 'IndexArray_t', dist_data_ideal.pop(pl_identifier).reshape(1,-1, order='F'))
  #Update data
  for data_nodes in PT.iter_children_from_predicates(node, data_query, ancestors=True):
    path = "/".join([PT.get_name(n) for n in data_nodes])
    if PT.get_label(data_nodes[-1]) == 'IndexArray_t':
      PT.set_value(data_nodes[-1], dist_data_ideal[path].reshape(1,-1, order='F'))
    elif PT.get_label(data_nodes[-1]) == 'DataArray_t':
      PT.set_value(data_nodes[-1], dist_data_ideal[path])

def _update_cgns_subsets(zone, location, entity_distri, old_to_new_face, base_name, comm):
  """
  Treated for now :
    BC, BCDataset (With or without PL), FlowSol, DiscreteData, ZoneSubRegion, JN

    Careful! PointList/PointListDonor arrays of joins present in the zone are updated, but opposite joins
    are not informed of this modification. This has to be done after the function.
  """

  # Prepare iterators
  matches_loc = PT.pred.has_location(location)
  is_bcds_with_pl    = PT.pred.label_is('BCDataSet_t') &  HAS_POINTLIST
  is_bcds_without_pl = PT.pred.label_is('BCDataSet_t') & ~HAS_POINTLIST

  is_sol  = PT.pred.label_in(['FlowSolution_t', 'DiscreteData_t']) & matches_loc
  is_bc   = PT.pred.label_is('BC_t') & matches_loc
  is_bcds = is_bcds_with_pl & matches_loc
  is_zsr  = PT.pred.label_is('ZoneSubRegion_t') & matches_loc
  is_jn   = PT.pred.label_is('GridConnectivity_t') & matches_loc

  sol_list  = PT.get_children_from_predicate(zone, is_sol)
  bc_list   = PT.get_children_from_predicates(zone, ['ZoneBC_t', is_bc])
  bcds_list = PT.get_children_from_predicates(zone, ['ZoneBC_t', 'BC_t', is_bcds])
  zsr_list  = PT.get_children_from_predicate(zone, is_zsr)
  jn_list   = PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', is_jn])

  i_jn_list = [jn for jn in jn_list if PT.GridConnectivity.ZoneDonorPath(jn, base_name) == base_name + '/'+ PT.get_name(zone)]
  i_jn_list = [jn for jn in jn_list if PT.GridConnectivity.ZoneDonorPath(jn, base_name) == base_name + '/'+ PT.get_name(zone)]

  #Loop in same order using to get apply pl using generic func
  all_nodes_and_queries = [
    ( sol_list , ['DataArray_t']                                 ),
    ( bc_list  , [is_bcds_without_pl, 'BCData_t', 'DataArray_t'] ),
    ( bcds_list, ['BCData_t', 'DataArray_t']                     ),
    ( zsr_list , ['DataArray_t']                                 ),
    ( jn_list  , ['PointListDonor']                              ),
  ]
  all_nodes = itertools.chain.from_iterable([elem[0] for elem in all_nodes_and_queries])

  #Trick to add a PL to each subregion to be able to use same algo
  for zsr in zsr_list:
    if PT.Subset.ZSRExtent(zsr, zone) != PT.get_name(zsr):
      pl_node = PT.get_node_from_path(zone, PT.Subset.ZSRExtent(zsr, zone) + '/PointList')
      PT.add_child(zsr, PT.deep_copy(pl_node))

  #Get new index for every PL at once
  all_pl_list = [PT.get_child_from_name(fs, 'PointList')[1][0]-1 for fs in all_nodes]
  part_data_pl = EP.block_to_part(old_to_new_face, entity_distri, all_pl_list, comm)

  part_offset = 0
  for node_list, data_query in all_nodes_and_queries:
    for node in node_list:
      _update_subset(node, part_data_pl[part_offset], data_query, comm)
      part_offset += 1

  #For internal jn only, we must update PointListDonor with new face id. Non internal jn reorder the array,
  # but do not apply old_to_new transformation.
  # Note that we will lost symmetry PL/PLD for internal jn, we need a rule to update it afterward
  all_pld = [PT.get_child_from_name(jn, 'PointListDonor') for jn in i_jn_list]
  updated_pld = EP.block_to_part(old_to_new_face, entity_distri, [pld[1][0]-1 for pld in all_pld], comm)
  for i, pld in enumerate(all_pld):
    PT.set_value(pld, updated_pld[i].reshape((1,-1), order='F'))

  #Cleanup after trick
  for zsr in zsr_list:
    if PT.Subset.ZSRExtent(zsr, zone) != PT.get_name(zsr):
      PT.rm_children_from_name(zsr, 'PointList')
      PT.rm_children_from_name(zsr, ':CGNS#Distribution')


# TODO move to sids module, doc, unit test
#(take the one of _shift_cgns_subsets, and for _shift_cgns_subsets, make a trivial test)
def all_nodes_with_point_list(zone, pl_location):
  has_pl = HAS_POINTLIST & PT.pred.has_location(pl_location)
  return itertools.chain(
      PT.get_children_from_predicate(zone, has_pl)                      , #FlowSolution_t, ZoneSubRegion_t, ...
      PT.get_children_from_predicates(zone, ['ZoneBC_t', has_pl])              , #BC_t
      #For this one we must exclude BC since predicate is also tested on root (and should not be ?)
      PT.get_children_from_predicates(zone, ['ZoneBC_t', 'BC_t', has_pl & ~PT.pred.label_is('BC_t')])      , #BCDataSet_t
      PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', has_pl]), #GridConnectivity_t
    )

def _shift_cgns_subsets(zone, location, shift_value):
  """
  Shift all the PointList of the requested location with the given value
  PointList are seached in every node below zone, + in BC_t, BCDataSet_t,
  GridConnectivity_t
  """
  for node in all_nodes_with_point_list(zone,location):
    PT.get_child_from_name(node, 'PointList')[1][0] += shift_value

def _update_vtx_data(zone, vtx_to_remove, comm):
  """
  Remove the vertices in data array supported by allVertex (currently
  managed : GridCoordinates, FlowSolution, DiscreteData)
  and update vertex distribution info
  """
  vtx_distri_ini  = MT.distribution_value(zone, 'Vertex')

  GI = EP.GlobalIndexer(vtx_distri_ini, vtx_to_remove-1, comm)
  mask = GI.access_counts == 0

  #Update all vertex entities
  for coord_n in PT.iter_children_from_predicates(zone, ['GridCoordinates_t', 'DataArray_t']):
    PT.set_value(coord_n, coord_n[1][mask])

  is_all_vtx_sol = PT.pred.label_in(['FlowSolution_t', 'DiscreteData_t']) \
                 & PT.pred.has_location('Vertex') & ~HAS_POINTLIST

  for node in PT.iter_children_from_predicate(zone, is_all_vtx_sol):
    for data_n in PT.iter_children_from_label(node, 'DataArray_t'):
      PT.set_value(data_n, data_n[1][mask])

  # Update vertex distribution
  i_rank, n_rank = comm.Get_rank(), comm.Get_size()
  n_rmvd   = mask.size - mask.sum()
  n_rmvd_offset  = par_utils.gather_and_shift(n_rmvd, comm, pdm_dtype)
  vtx_distri = vtx_distri_ini - [n_rmvd_offset[i_rank], n_rmvd_offset[i_rank+1],  n_rmvd_offset[n_rank]]
  MT.new_Distribution({'Vertex' : vtx_distri}, zone)
  zone[1][0][0] = vtx_distri[2]

def _update_pl_pld_in_jn(dist_tree, zone_path):
  # Since PointList/PointList donor of JN have changed, we must change opposite join as well
  # Carefull : for intra zone jn, we may have a clash of actualized pl/pld. We use pathes to break it
  base_n, zone_n = zone_path.split('/')
  zone = PT.get_node_from_path(dist_tree, zone_path)
  jn_to_opp = {}
  all_gcs_query = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', 'GridConnectivity_t']
  for zgc, gc in PT.iter_children_from_predicates(zone, all_gcs_query[2:], ancestors=True):
    jn_to_opp[base_n + '/' + zone_n + '/' + zgc[0] + '/' + gc[0]] = \
        (np.copy(PT.get_child_from_name(gc, 'PointList')[1]), np.copy(PT.get_child_from_name(gc, 'PointListDonor')[1]))
  for o_base, o_zone, o_zgc, o_gc in PT.iter_children_from_predicates(dist_tree, all_gcs_query, ancestors=True):
    gc_path = '/'.join([PT.get_name(node) for node in [o_base, o_zone, o_zgc, o_gc]])
    gc_path_opp = MJT.get_jn_donor_path(dist_tree, gc_path)
    try:
      pl_opp, pld_opp = jn_to_opp[gc_path_opp]
      # Skip one internal jn over two
      if PT.get_name(o_base) + '/' + PT.get_name(o_zone) == zone_path and gc_path_opp >= gc_path:
        pass
      else:
        PT.set_value(PT.get_child_from_name(o_gc, 'PointList'),     pld_opp)
        PT.set_value(PT.get_child_from_name(o_gc, 'PointListDonor'), pl_opp)
      #Since we modify the PointList of this join, we must check that no data is related to it
      assert PT.get_child_from_label(o_gc, 'DataArray_t') is None, \
          "Can not reorder a GridConnectivity PointList to which data is related"
      for zsr in PT.iter_children_from_label(o_zone, 'ZoneSubRegion_t'):
        assert PT.Subset.ZSRExtent(zsr, o_zone) != PT.get_name(o_zgc) + '/' + PT.get_name(o_gc), \
            "Can not reorder a GridConnectivity PointList to which data is related"
    except KeyError:
      pass



def merge_intrazone_jn(dist_tree, jn_pathes, comm):
  """
  """
  base_n, zone_n, zgc_n, gc_n = jn_pathes[0].split('/')
  gc = PT.get_node_from_path(dist_tree, jn_pathes[0])
  assert PT.Subset.GridLocation(gc) == 'FaceCenter'
  zone = PT.get_node_from_path(dist_tree, base_n + '/' + zone_n)
  ngon  = [elem for elem in PT.iter_children_from_label(zone, 'Elements_t') if elem[1][0] == 22][0]
  nface_l = [elem for elem in PT.iter_children_from_label(zone, 'Elements_t') if elem[1][0] == 23]
  nface = nface_l[0] if len(nface_l) == 1 else None

  MJT.add_joins_donor_name(dist_tree, comm)

  ref_faces      = PT.get_node_from_path(dist_tree, jn_pathes[0]+'/PointList')[1][0]
  face_to_remove = PT.get_node_from_path(dist_tree, jn_pathes[0]+'/PointListDonor')[1][0]
  #Create pl and pl_d from vertex, needed to know which vertex will be deleted
  ref_vtx, vtx_to_remove, _ = VL.generate_jn_vertex_list(dist_tree, jn_pathes[0], comm)
  # In some cases, we can have some vertices shared between PlVtx and PldVtx (eg. when a vertex
  # belongs to more than 2 gcs, and some of them has been deleted.
  # We just ignore thoses vertices by removing them from the arrays
  vtx_is_shared = ref_vtx == vtx_to_remove
  ref_vtx       = ref_vtx[~vtx_is_shared]
  vtx_to_remove = vtx_to_remove[~vtx_is_shared]
  assert np.intersect1d(ref_faces, face_to_remove).size == 0
  assert np.intersect1d(ref_vtx, vtx_to_remove).size == 0

  #Get initial distributions
  face_distri_ini = MT.distribution_value(ngon, 'Element').copy()
  vtx_distri_ini  = MT.distribution_value(zone, 'Vertex')

  old_to_new_face = merge_distributed_ids(face_distri_ini, face_to_remove, ref_faces, comm, True)
  old_to_new_vtx  = merge_distributed_ids(vtx_distri_ini, vtx_to_remove, ref_vtx, comm)
  old_to_new_face_unsg = np.abs(old_to_new_face)

  n_rmvd_face    = comm.allreduce(len(face_to_remove), op=MPI.SUM)

  _update_ngon(ngon, ref_faces, face_to_remove, vtx_distri_ini, old_to_new_vtx, comm)
  if nface:
    _update_nface(nface, face_distri_ini, old_to_new_face, n_rmvd_face, comm)

  _update_vtx_data(zone, vtx_to_remove, comm)

  _update_cgns_subsets(zone, 'FaceCenter', face_distri_ini, old_to_new_face_unsg, base_n, comm)
  _update_cgns_subsets(zone, 'Vertex', vtx_distri_ini, old_to_new_vtx, base_n, comm)
  #Shift all CellCenter PL by the number of removed faces
  if PT.Element.Range(ngon)[0] == 1:
    _shift_cgns_subsets(zone, 'CellCenter', -n_rmvd_face)

  current_zone_path = base_n + '/' + zone_n
  _update_pl_pld_in_jn(dist_tree, current_zone_path)

  #Cleanup
  PT.rm_node_from_path(dist_tree, jn_pathes[0])
  PT.rm_node_from_path(dist_tree, jn_pathes[1])
