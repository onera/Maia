from mpi4py import MPI
import numpy as np

import Converter.Internal as I
from Pypdm import Pypdm as PDM

from maia.sids import Internal_ext as IE
from maia.sids import sids

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.tree_exchange.dist_to_part import data_exchange as MBTP
from maia.tree_exchange.part_to_dist import data_exchange as MPTB

from maia.utils.parallel import utils as par_utils

import maia.connectivity.remove_element as RME
from maia.transform.dist_tree import add_joins_ordinal as AJO
from maia.transform.dist_tree.merge_ids import merge_distributed_ids
from maia.connectivity import vertex_list as VL

def _update_ngon(ngon, ref_faces, del_faces, vtx_distri_ini, old_to_new_vtx, comm):
  """
  """
  face_distri = IE.getDistribution(ngon, 'Element').astype(pdm_dtype)
  pe          =  I.getNodeFromPath(ngon, 'ParentElements')[1]

  # A/ Exchange parent cells before removing : 
  # 1. Get the left cell of the faces to delete
  dist_data = {'PE' : pe[:,0]}
  part_data = MBTP.dist_to_part(face_distri, dist_data, [del_faces], comm)
  # 2. Put it in the right cell of the faces to keep
  #Todo : exchange of ref_faces could be avoided using get gnum copy
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

def _update_subset(node, pl_new, data_query, comm):
  """
  Update a PointList and all the data
  """
  part_data = {}
  dist_data = {}
  for data_nodes in IE.getNodesWithParentsByMatching(node, data_query):
    path = "/".join([I.getName(n) for n in data_nodes])
    part_data[path] = [data_nodes[-1][1][0]]

  #Dont use maia interface since we need a new distribution
  PTB = PDM.PartToBlock(comm, [pl_new.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=1, t_stride=0)
  PTB.PartToBlock_Exchange(dist_data, part_data)

  d_pl_new = PTB.getBlockGnumCopy()

  new_distri_full = par_utils.gather_and_shift(len(d_pl_new), comm)
  new_distri = new_distri_full[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]
  #Update distribution and size
  IE.newDistribution({'Index' : new_distri}, node)
  I.getNodeFromPath(node, 'PointList#Size')[1][1] = new_distri[-1]
    
  #Update PointList and data
  I.createUniqueChild(node, 'PointList', 'IndexArray_t', d_pl_new.reshape(1,-1, order='F'))
  for data_nodes in IE.getNodesWithParentsByMatching(node, data_query):
    path = "/".join([I.getName(n) for n in data_nodes])
    I.setValue(data_nodes[-1], dist_data[path].reshape(1,-1, order='F'))

def _update_cgns_subsets(zone, location, entity_distri, old_to_new_face, comm):
  """
  Treated for now :
    BC, BCDataset (With or without PL), FlowSol, DiscreteData, ZoneSubRegion, JN

    Carefull ! PointList arrays of joins present in the zone are updated, but opposite joins
    are not informed of this modification. This has to be done after the function if PointListDonor
    are needed
  """

  # Prepare iterators
  matches_loc = lambda n : sids.GridLocation(n) == location

  is_face_sol  = lambda n: matches_loc(n) and I.getType(n) in ['FlowSolution_t', 'DiscreteData_t']
  is_face_bc   = lambda n: matches_loc(n) and I.getType(n) == 'BC_t'
  is_face_bcds = lambda n: matches_loc(n) and I.getType(n) == 'BCDataSet_t'\
                                             and I.getNodeFromPath(n, 'PointList') is not None
  is_face_zsr  = lambda n: matches_loc(n) and I.getType(n) == 'ZoneSubRegion_t'
  is_face_jn   = lambda n: matches_loc(n) and I.getType(n) == 'GridConnectivity_t'

  sol_iterator  = IE.getChildrenFromPredicate(zone, is_face_sol)
  bc_iterator   = list(IE.getNodesByMatching(zone, ['ZoneBC_t', is_face_bc])) #To reuse generator
  bcds_iterator = list(IE.getNodesByMatching(zone, ['ZoneBC_t', 'BC_t', is_face_bcds]))
  zsr_iterator  = IE.getChildrenFromPredicate(zone, is_face_zsr)
  jn_iterator   = list(IE.getNodesByMatching(zone, ['ZoneGridConnectivity_t', is_face_jn]))

  #Trick to add a PL to each subregion to be able to use same algo
  for zsr in zsr_iterator:
    if IE.getSubregionExtent(zsr, zone) != I.getName(zsr):
      I._addChild(zsr, I.getNodeFromPath(zone, IE.getSubregionExtent(zsr, zone) + '/PointList'))
      I._addChild(zsr, I.getNodeFromPath(zone, IE.getSubregionExtent(zsr, zone) + '/PointList#Size'))

  #Get new index for every PL at once
  all_pl_list = [I.getNodeFromName1(fs, 'PointList')[1][0].astype(pdm_dtype) for fs in \
     sol_iterator + bc_iterator + bcds_iterator + zsr_iterator + jn_iterator]
  dist_data_pl = {'OldToNew' : old_to_new_face}
  part_data_pl = MBTP.dist_to_part(entity_distri, dist_data_pl, all_pl_list, comm)

  part_offset = 0

  #Loop in same order using to get apply pl using generic func
  data_query = ['DataArray_t']
  for fs in sol_iterator:
    _update_subset(fs, part_data_pl['OldToNew'][part_offset], data_query, comm)
    part_offset +=1

  related_bcds = lambda n: I.getType(n) == 'BCDataSet_t' and I.getNodeFromName(n, 'PointList') is None
  data_query = [related_bcds, 'BCData_t', 'DataArray_t']
  for bc in bc_iterator:
    _update_subset(bc, part_data_pl['OldToNew'][part_offset], data_query, comm)
    part_offset +=1

  data_query = ['BCData_t', 'DataArray_t']
  for bcds in bcds_iterator:
    _update_subset(bcds, part_data_pl['OldToNew'][part_offset], data_query, comm)
    part_offset += 1

  data_query = ['DataArray_t']
  for zsr in zsr_iterator:
    _update_subset(zsr, part_data_pl['OldToNew'][part_offset], data_query, comm)
    #Cleanup after trick
    part_offset += 1
    if IE.getSubregionExtent(zsr, zone) != I.getName(zsr):
      I._rmNodesByName(zsr, 'PointList')

  data_query = []
  for jn in jn_iterator:
    _update_subset(jn, part_data_pl['OldToNew'][part_offset], data_query, comm)
    part_offset += 1

def _shift_cgns_subsets(zone, location, shift_value):
  """
  Shift all the PointList of the requested location with the given value
  PointList are seached in every node below zone, + in BC_t, BCDataSet_t,
  GridConnectivity_t
  """
  has_pl = lambda n: I.getNodeFromName1(n, 'PointList') is not None \
                     and sids.GridLocation(n) == location
  for node in IE.getChildrenFromPredicate(zone, has_pl):
    I.getNodeFromName1(node, 'PointList')[1][0] += shift_value
  for node in IE.getNodesByMatching(zone, ['ZoneBC_t', has_pl, 'PointList']): #BC
    I.getNodeFromName1(node, 'PointList')[1][0] += shift_value
  for node in IE.getNodesByMatching(zone, ['ZoneBC_t', 'BC_t', has_pl]): #BCDataSet
    I.getNodeFromName1(node, 'PointList')[1][0] += shift_value
  for node in IE.getNodesByMatching(zone, ['ZoneGridConnectivity_t', has_pl]): #GC
    I.getNodeFromName1(node, 'PointList')[1][0] += shift_value

def _update_vtx_data(zone, vtx_to_remove, comm):
  """
  Remove the vertices in data array supported by allVertex (currently 
  managed : GridCoordinates, FlowSolution, DiscreteData)
  and update vertex distribution info
  """
  vtx_distri_ini  = IE.getDistribution(zone, 'Vertex')

  PTB = PDM.PartToBlock(comm, [vtx_to_remove.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=1, t_stride=0)
  local_vtx_to_rmv = PTB.getBlockGnumCopy() - vtx_distri_ini[0] - 1

  #Update all vertex entities
  for coord_n in IE.getNodesByMatching(zone, ['GridCoordinates_t', 'DataArray_t']):
    I.setValue(coord_n, np.delete(coord_n[1], local_vtx_to_rmv))
  
  is_all_vtx_sol = lambda n: I.getType(n) in ['FlowSolution_t', 'DiscreteData_t'] \
      and sids.GridLocation(n) == 'Vertex' and I.getNodeFromPath(n, 'PointList') is None

  for data_n in IE.getChildrenFromPredicate(zone, is_all_vtx_sol):
    I.setValue(data_n, np.delete(data_n[1], local_vtx_to_rmv))

  # Update vertex distribution
  i_rank, n_rank = comm.Get_rank(), comm.Get_size()
  n_rmvd   = len(local_vtx_to_rmv)
  n_rmvd_offset  = par_utils.gather_and_shift(n_rmvd, comm)
  vtx_distri = vtx_distri_ini - [n_rmvd_offset[i_rank], n_rmvd_offset[i_rank] + n_rmvd,  n_rmvd_offset[n_rank]]
  IE.newDistribution({'Vertex' : vtx_distri}, zone)
  zone[1][0][0] = vtx_distri[2]



def merge_intrazone_jn(dist_tree, jn_pathes, comm):
  """
  """
  base_n, zone_n, zgc_n, gc_n = jn_pathes[0].split('/')
  zone = I.getNodeFromPath(dist_tree, base_n + '/' + zone_n)
  ngon = [elem for elem in I.getNodesFromType1(zone, 'Elements_t') if elem[1][0] == 22][0]

  if I.getNodeFromName1(gc_n, 'Ordinal') is None:
    AJO.add_joins_ordinal(dist_tree, comm)

  ref_faces      = I.getNodeFromPath(dist_tree, jn_pathes[0]+'/PointList')[1][0]
  face_to_remove = I.getNodeFromPath(dist_tree, jn_pathes[0]+'/PointListDonor')[1][0]
  #Create pl and pl_d from vertex, needed to know which vertex will be deleted
  ref_vtx, vtx_to_remove, _ = VL.generate_jn_vertex_list(dist_tree, jn_pathes[0], comm)

  #Get initial distributions
  face_distri_ini = IE.getDistribution(ngon, 'Element').astype(pdm_dtype)
  vtx_distri_ini  = IE.getDistribution(zone, 'Vertex').astype(pdm_dtype)

  old_to_new_face = merge_distributed_ids(face_distri_ini, face_to_remove, ref_faces, comm)
  old_to_new_vtx  = merge_distributed_ids(vtx_distri_ini, vtx_to_remove, ref_vtx, comm)

  _update_ngon(ngon, ref_faces, face_to_remove, vtx_distri_ini, old_to_new_vtx, comm)
  _update_cgns_subsets(zone, 'FaceCenter', face_distri_ini, old_to_new_face, comm)
  _update_cgns_subsets(zone, 'Vertex', vtx_distri_ini, old_to_new_vtx, comm)

  _update_vtx_data(zone, vtx_to_remove, comm)

  #Shift all CellCenter PL by the number of removed faces
  if sids.ElementRange(ngon)[0] == 1:
    n_rmvd_face = comm.allreduce(len(face_to_remove), op=MPI.SUM)
    _shift_cgns_subsets(zone, 'CellCenter', -n_rmvd_face)

  # Update JN/PointListDonor if any : since PointList of JN have changed, we must change opposite PL as well
  ordinal_to_pl = {}
  all_gcs_query = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', 'GridConnectivity_t']
  for gc in IE.getNodesByMatching(zone, all_gcs_query[2:]):
    ordinal_to_pl[I.getNodeFromName1(gc, 'Ordinal')[1][0]] = I.getNodeFromName1(gc, 'PointList')[1]
  for gc in IE.getNodesByMatching(dist_tree, all_gcs_query):
    try:
      pl_opp = ordinal_to_pl[I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]]
      I.setValue(I.getNodeFromName1(gc, 'PointListDonor'), pl_opp)
    except KeyError:
      pass


  #Cleanup
  I._rmNodeByPath(dist_tree, jn_pathes[0])
  I._rmNodeByPath(dist_tree, jn_pathes[1])
  

