from mpi4py import MPI
import numpy as np
import itertools

import Converter.Internal as I
from Pypdm import Pypdm as PDM

from maia.sids import Internal_ext as IE
from maia.sids import sids
from maia.sids import pytree as PT

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.tree_exchange.dist_to_part import data_exchange as MBTP
from maia.tree_exchange.part_to_dist import data_exchange as MPTB

from maia.utils.parallel import utils as par_utils
from maia.distribution import distribution_function as DIF

import maia.connectivity.remove_element as RME
from maia.transform.dist_tree import add_joins_ordinal as AJO
from maia.transform.dist_tree.merge_ids import merge_distributed_ids
from maia.connectivity import vertex_list as VL

def _update_ngon(ngon, ref_faces, del_faces, vtx_distri_ini, old_to_new_vtx, comm):
  """
  Update ngon node after face and vertex merging, ie
   - update ParentElements to combinate faces
   - remove faces from EC, PE and ESO and update distribution info
   - update ElementConnectivity using vertex old_to_new order
  """
  face_distri = I.getVal(IE.getDistribution(ngon, 'Element')).astype(pdm_dtype)
  pe          =  I.getNodeFromPath(ngon, 'ParentElements')[1]

  # A/ Exchange parent cells before removing :
  # 1. Get the left cell of the faces to delete
  dist_data = {'PE' : pe[:,0]}
  part_data = MBTP.dist_to_part(face_distri, dist_data, [del_faces], comm)
  # 2. Put it in the right cell of the faces to keep
  #TODO : exchange of ref_faces could be avoided using get gnum copy
  part_data['FaceId'] = [ref_faces]
  dist_data = MPTB.part_to_dist(face_distri, part_data, [ref_faces], comm)

  local_faces = dist_data['FaceId'] - face_distri[0] - 1
  assert np.max(pe[local_faces, 1], initial=0) == 0 #Initial = trick to admit empty array
  pe[local_faces, 1] = dist_data['PE']

  # B/ Update EC, PE and ESO removing some faces
  part_data = {'FaceIdDonor' : [del_faces]}
  dist_data = MPTB.part_to_dist(face_distri, part_data, [del_faces], comm)
  local_faces = dist_data['FaceIdDonor'] - face_distri[0] - 1
  RME.remove_ngons(ngon, local_faces, comm)

  # C/ Update vertex ids in EC
  ngon_ec_n = I.getNodeFromName1(ngon, 'ElementConnectivity')
  dist_data = {'OldToNew' : old_to_new_vtx}
  part_data = MBTP.dist_to_part(vtx_distri_ini, dist_data, [ngon_ec_n[1].astype(pdm_dtype)], comm)
  assert len(ngon_ec_n[1]) == len(part_data['OldToNew'][0])
  I.setValue(ngon_ec_n, part_data['OldToNew'][0])

def _update_nface(nface, face_distri_ini, old_to_new_face, n_rmvd_face, comm):
  """
  Update nface node after face merging, ie
   - update ElementConnectivity using face old_to_new order
   - Shift ElementRange (to substract nb of removed faces) if NFace is after NGon
  If input array old_to_new_face is signed (ie is negative for face ids that will be removed),
  then the orientation of nface connectivity is preserved
  """

  #Update list of faces
  dist_data = {'OldToNew' : old_to_new_face}
  nface_ec_n = I.getNodeFromName1(nface, 'ElementConnectivity')
  part_data = MBTP.dist_to_part(face_distri_ini, dist_data, [np.abs(nface_ec_n[1].astype(pdm_dtype))], comm)
  assert len(nface_ec_n[1]) == len(part_data['OldToNew'][0])
  #Get sign of nface_ec to preserve orientation
  I.setValue(nface_ec_n, np.sign(nface_ec_n[1]) * part_data['OldToNew'][0])

  #Update ElementRange
  er = sids.ElementRange(nface)
  if er[0] > 1:
    er -= n_rmvd_face

def _update_subset(node, pl_new, data_query, comm):
  """
  Update a PointList and all the data
  """
  part_data = {}
  dist_data = {}
  for data_nodes in IE.getNodesWithParentsByMatching(node, data_query):
    path = "/".join([I.getName(n) for n in data_nodes])
    part_data[path] = [data_nodes[-1][1][0]]
  #Add PL, needed for next blocktoblock
  pl_identifier = r'@\PointList/@' # just a string that is unlikely to clash
  part_data[pl_identifier] = [pl_new.astype(pdm_dtype)]

  #Don't use maia interface since we need a new distribution
  PTB = PDM.PartToBlock(comm, [pl_new.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=1, t_stride=0)
  PTB.PartToBlock_Exchange(dist_data, part_data)

  d_pl_new = PTB.getBlockGnumCopy()

  new_distri_full = par_utils.gather_and_shift(len(d_pl_new), comm, pdm_dtype)
  #Result is badly distributed, we can do a BlockToBlock to have a uniform distribution
  ideal_distri      = DIF.uniform_distribution(new_distri_full[-1], comm)
  ideal_distri_full = par_utils.partial_to_full_distribution(ideal_distri, comm)
  dist_data_ideal = dict()
  BTB = PDM.BlockToBlock(new_distri_full, ideal_distri_full, comm)
  BTB.BlockToBlock_Exchange(dist_data, dist_data_ideal)

  #Update distribution and size
  IE.newDistribution({'Index' : ideal_distri}, node)
  if I.getNodeFromPath(node, 'PointList#Size') is not None:
    I.getNodeFromPath(node, 'PointList#Size')[1][1] = ideal_distri[-1]

  #Update PointList and data
  I.createUniqueChild(node, 'PointList', 'IndexArray_t', dist_data_ideal.pop(pl_identifier).reshape(1,-1, order='F'))
  #Update data
  for data_nodes in IE.getNodesWithParentsByMatching(node, data_query):
    path = "/".join([I.getName(n) for n in data_nodes])
    I.setValue(data_nodes[-1], dist_data_ideal[path].reshape(1,-1, order='F'))

def _update_cgns_subsets(zone, location, entity_distri, old_to_new_face, base_name, comm):
  """
  Treated for now :
    BC, BCDataset (With or without PL), FlowSol, DiscreteData, ZoneSubRegion, JN

    Careful! PointList/PointListDonor arrays of joins present in the zone are updated, but opposite joins
    are not informed of this modification. This has to be done after the function.
  """

  # Prepare iterators
  matches_loc = lambda n : sids.GridLocation(n) == location
  is_bcds_with_pl    = lambda n: I.getType(n) == 'BCDataSet_t'and I.getNodeFromName1(n, 'PointList') is not None
  is_bcds_without_pl = lambda n: I.getType(n) == 'BCDataSet_t'and I.getNodeFromName1(n, 'PointList') is None

  is_sol  = lambda n: matches_loc(n) and I.getType(n) in ['FlowSolution_t', 'DiscreteData_t']
  is_bc   = lambda n: matches_loc(n) and I.getType(n) == 'BC_t'
  is_bcds = lambda n: matches_loc(n) and is_bcds_with_pl(n)
  is_zsr  = lambda n: matches_loc(n) and I.getType(n) == 'ZoneSubRegion_t'
  is_jn   = lambda n: matches_loc(n) and I.getType(n) == 'GridConnectivity_t'

  sol_list  = PT.getChildrenFromPredicate(zone, is_sol)
  bc_list   = PT.getChildrenFromPredicates(zone, ['ZoneBC_t', is_bc])
  bcds_list = PT.getChildrenFromPredicates(zone, ['ZoneBC_t', 'BC_t', is_bcds])
  zsr_list  = PT.getChildrenFromPredicate(zone, is_zsr)
  jn_list   = PT.getChildrenFromPredicates(zone, ['ZoneGridConnectivity_t', is_jn])
  i_jn_list = [jn for jn in jn_list if IE.getZoneDonorPath(base_name, jn) == base_name + '/'+ I.getName(zone)]

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
    if IE.getSubregionExtent(zsr, zone) != I.getName(zsr):
      I._addChild(zsr, I.getNodeFromPath(zone, IE.getSubregionExtent(zsr, zone) + '/PointList'))
      I._addChild(zsr, I.getNodeFromPath(zone, IE.getSubregionExtent(zsr, zone) + '/PointList#Size'))

  #Get new index for every PL at once
  all_pl_list = [I.getNodeFromName1(fs, 'PointList')[1][0].astype(pdm_dtype) for fs in all_nodes]
  dist_data_pl = {'OldToNew' : old_to_new_face}
  part_data_pl = MBTP.dist_to_part(entity_distri, dist_data_pl, all_pl_list, comm)

  part_offset = 0
  for node_list, data_query in all_nodes_and_queries:
    for node in node_list:
      _update_subset(node, part_data_pl['OldToNew'][part_offset], data_query, comm)
      part_offset += 1

  #For internal jn only, we must update PointListDonor with new face id. Non internal jn reorder the array,
  # but do not apply old_to_new transformation.
  # Note that we will lost symmetry PL/PLD for internal jn, we need a rule to update it afterward
  all_pld = [I.getNodeFromName1(jn, 'PointListDonor') for jn in i_jn_list]
  updated_pld = MBTP.dist_to_part(entity_distri, dist_data_pl, [pld[1][0].astype(pdm_dtype) for pld in all_pld], comm)
  for i, pld in enumerate(all_pld):
    I.setValue(pld, updated_pld['OldToNew'][i].reshape((1,-1), order='F'))

  #Cleanup after trick
  for zsr in zsr_list:
    if IE.getSubregionExtent(zsr, zone) != I.getName(zsr):
      I._rmNodesByName(zsr, 'PointList')


# TODO move to sids module, doc, unit test
#(take the one of _shift_cgns_subsets, and for _shift_cgns_subsets, make a trivial test)
def all_nodes_with_point_list(zone, pl_location):
  has_pl = lambda n: I.getNodeFromName1(n, 'PointList') is not None \
                     and sids.GridLocation(n) == pl_location
  return itertools.chain(
      PT.getChildrenFromPredicate(zone, has_pl)                      , #FlowSolution_t, ZoneSubRegion_t, ...
      PT.getChildrenFromPredicates(zone, ['ZoneBC_t', has_pl])              , #BC_t
      #For this one we must exclude BC since predicate is also tested on root (and should not be ?)
      PT.getChildrenFromPredicates(zone, ['ZoneBC_t', 'BC_t', lambda n : has_pl(n) and I.getType(n) != 'BC_t'])      , #BCDataSet_t
      PT.getChildrenFromPredicates(zone, ['ZoneGridConnectivity_t', has_pl]), #GridConnectivity_t
    )

def _shift_cgns_subsets(zone, location, shift_value):
  """
  Shift all the PointList of the requested location with the given value
  PointList are seached in every node below zone, + in BC_t, BCDataSet_t,
  GridConnectivity_t
  """
  for node in all_nodes_with_point_list(zone,location):
    I.getNodeFromName1(node, 'PointList')[1][0] += shift_value

def _update_vtx_data(zone, vtx_to_remove, comm):
  """
  Remove the vertices in data array supported by allVertex (currently
  managed : GridCoordinates, FlowSolution, DiscreteData)
  and update vertex distribution info
  """
  vtx_distri_ini  = I.getVal(IE.getDistribution(zone, 'Vertex')).astype(pdm_dtype)
  pdm_distrib     = par_utils.partial_to_full_distribution(vtx_distri_ini, comm)

  PTB = PDM.PartToBlock(comm, [vtx_to_remove.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=1, t_stride=0, userDistribution=pdm_distrib)
  local_vtx_to_rmv = PTB.getBlockGnumCopy() - vtx_distri_ini[0] - 1

  #Update all vertex entities
  for coord_n in IE.getNodesByMatching(zone, ['GridCoordinates_t', 'DataArray_t']):
    I.setValue(coord_n, np.delete(coord_n[1], local_vtx_to_rmv))

  is_all_vtx_sol = lambda n: I.getType(n) in ['FlowSolution_t', 'DiscreteData_t'] \
      and sids.GridLocation(n) == 'Vertex' and I.getNodeFromPath(n, 'PointList') is None

  for node in PT.iterChildrenFromPredicate(zone, is_all_vtx_sol):
    for data_n in I.getNodesFromType1(node, 'DataArray_t'):
      I.setValue(data_n, np.delete(data_n[1], local_vtx_to_rmv))

  # Update vertex distribution
  i_rank, n_rank = comm.Get_rank(), comm.Get_size()
  n_rmvd   = len(local_vtx_to_rmv)
  n_rmvd_offset  = par_utils.gather_and_shift(n_rmvd, comm)
  vtx_distri = vtx_distri_ini - [n_rmvd_offset[i_rank], n_rmvd_offset[i_rank+1],  n_rmvd_offset[n_rank]]
  IE.newDistribution({'Vertex' : vtx_distri}, zone)
  zone[1][0][0] = vtx_distri[2]



def merge_intrazone_jn(dist_tree, jn_pathes, comm):
  """
  """
  base_n, zone_n, zgc_n, gc_n = jn_pathes[0].split('/')
  gc = I.getNodeFromPath(dist_tree, jn_pathes[0])
  assert sids.GridLocation(gc) == 'FaceCenter'
  zone = I.getNodeFromPath(dist_tree, base_n + '/' + zone_n)
  ngon  = [elem for elem in I.getNodesFromType1(zone, 'Elements_t') if elem[1][0] == 22][0]
  nface_l = [elem for elem in I.getNodesFromType1(zone, 'Elements_t') if elem[1][0] == 23]
  nface = nface_l[0] if len(nface_l) == 1 else None

  if I.getNodeFromName1(gc, 'Ordinal') is None:
    AJO.add_joins_ordinal(dist_tree, comm)

  ref_faces      = I.getNodeFromPath(dist_tree, jn_pathes[0]+'/PointList')[1][0]
  face_to_remove = I.getNodeFromPath(dist_tree, jn_pathes[0]+'/PointListDonor')[1][0]
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
  face_distri_ini = I.getVal(IE.getDistribution(ngon, 'Element')).astype(pdm_dtype)
  vtx_distri_ini  = I.getVal(IE.getDistribution(zone, 'Vertex')).astype(pdm_dtype)

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
  if sids.ElementRange(ngon)[0] == 1:
    _shift_cgns_subsets(zone, 'CellCenter', -n_rmvd_face)

  # Since PointList/PointList donor of JN have changed, we must change opposite join as well
  # Carefull : for intra zone jn, we may have a clash of actualized pl/pld. We use ordinal to break it
  ordinal_to_pl = {}
  current_zone_path = base_n + '/' + zone_n
  all_gcs_query = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', 'GridConnectivity_t']
  for gc in IE.getNodesByMatching(zone, all_gcs_query[2:]):
    ordinal_to_pl[I.getNodeFromName1(gc, 'Ordinal')[1][0]] = \
        (np.copy(I.getNodeFromName1(gc, 'PointList')[1]), np.copy(I.getNodeFromName1(gc, 'PointListDonor')[1]))
  for o_base, o_zone, o_zgc, o_gc in IE.getNodesWithParentsByMatching(dist_tree, all_gcs_query):
    ord, ord_opp = I.getNodeFromName1(o_gc, 'Ordinal')[1][0], I.getNodeFromName1(o_gc, 'OrdinalOpp')[1][0]
    try:
      pl_opp, pld_opp = ordinal_to_pl[ord_opp]
      # Skip one internal jn over two
      if I.getName(o_base) + '/' + I.getName(o_zone) == current_zone_path and ord_opp >= ord:
        pass
      else:
        I.setValue(I.getNodeFromName1(o_gc, 'PointList'),     pld_opp)
        I.setValue(I.getNodeFromName1(o_gc, 'PointListDonor'), pl_opp)
      #Since we modify the PointList of this join, we must check that no data is related to it
      assert I.getNodeFromType1(o_gc, 'DataArray_t') is None, \
          "Can not reorder a GridConnectivity PointList to which data is related"
      for zsr in I.getNodesFromType1(o_zone, 'ZoneSubRegion_t'):
        assert sids.getSubregionExtent(zsr, o_zone) != I.getName(o_zgc) + '/' + I.getName(o_gc), \
            "Can not reorder a GridConnectivity PointList to which data is related"
    except KeyError:
      pass

  #Cleanup
  I._rmNodeByPath(dist_tree, jn_pathes[0])
  I._rmNodeByPath(dist_tree, jn_pathes[1])
