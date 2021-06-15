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
from maia.transform.dist_tree.merge_ids import merge_distributed_ids
from maia.connectivity import vertex_list as VL

def _update_ngon(ngon, ref_faces, del_faces, comm):
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

def _update_face_subsets(zone, face_distri, old_to_new_face, comm):
  """
  Treated for now :
    BC, BCDataset (With or without PL), FlowSol, DiscreteData, ZoneSubRegion, JN

    Carefull ! PointList arrays of joins present in the zone are updated, but opposite joins
    are not informed of this modification. This has to be done after the function if PointListDonor
    are needed
  """

  # Prepare iterators
  is_face_center = lambda n : sids.GridLocation(n) == 'FaceCenter'

  is_face_sol  = lambda n: is_face_center(n) and I.getType(n) in ['FlowSolution_t', 'DiscreteData_t']
  is_face_bc   = lambda n: is_face_center(n) and I.getType(n) == 'BC_t'
  is_face_bcds = lambda n: is_face_center(n) and I.getType(n) == 'BCDataSet_t'\
                                             and I.getNodeFromPath(n, 'PointList') is not None
  is_face_zsr  = lambda n: is_face_center(n) and I.getType(n) == 'ZoneSubRegion_t'
  is_face_jn   = lambda n: is_face_center(n) and I.getType(n) == 'GridConnectivity_t'

  face_sol_iterator  = IE.getChildrenFromPredicate(zone, is_face_sol)
  face_bc_iterator   = list(IE.getNodesByMatching(zone, ['ZoneBC_t', is_face_bc])) #To reuse generator
  face_bcds_iterator = list(IE.getNodesByMatching(zone, ['ZoneBC_t', 'BC_t', is_face_bcds]))
  face_zsr_iterator  = IE.getChildrenFromPredicate(zone, is_face_zsr)
  face_jn_iterator   = list(IE.getNodesByMatching(zone, ['GridConnectivity_t', is_face_jn]))

  #Trick to add a PL to each subregion to be able to use same algo
  for zsr in face_zsr_iterator:
    if IE.getSubregionExtent(zsr, zone) != I.getName(zsr):
      I._addChild(zsr, I.getNodeFromPath(zone, IE.getSubregionExtent(zsr, zone) + '/PointList'))
      I._addChild(zsr, I.getNodeFromPath(zone, IE.getSubregionExtent(zsr, zone) + '/PointList#Size'))

  #Get new index for every PL at once
  all_pl_list = [I.getNodeFromName1(fs, 'PointList')[1][0].astype(pdm_dtype) for fs in \
     face_sol_iterator + face_bc_iterator + face_bcds_iterator + face_zsr_iterator + face_jn_iterator]
  dist_data_pl = {'OldToNew' : old_to_new_face}
  part_data_pl = MBTP.dist_to_part(face_distri, dist_data_pl, all_pl_list, comm)

  part_offset = 0

  #Loop in same order using to get apply pl using generic func
  data_query = ['DataArray_t']
  for fs in face_sol_iterator:
    _update_subset(fs, part_data_pl['OldToNew'][part_offset], data_query, comm)
    part_offset +=1

  related_bcds = lambda n: I.getType(n) == 'BCDataSet_t' and I.getNodeFromName(n, 'PointList') is None
  data_query = [related_bcds, 'BCData_t', 'DataArray_t']
  for bc in face_bc_iterator:
    _update_subset(bc, part_data_pl['OldToNew'][part_offset], data_query, comm)
    part_offset +=1

  data_query = ['BCData_t', 'DataArray_t']
  for bcds in face_bcds_iterator:
    _update_subset(bcds, part_data_pl['OldToNew'][part_offset], data_query, comm)
    part_offset += 1

  data_query = ['DataArray_t']
  for zsr in face_zsr_iterator:
    _update_subset(zsr, part_data_pl['OldToNew'][part_offset], data_query, comm)
    #Cleanup after trick
    part_offset += 1
    if IE.getSubregionExtent(zsr, zone) != I.getName(zsr):
      I._rmNodesByName(zsr, 'PointList')

  data_query = []
  for jn in face_jn_iterator:
    _update_subset(jn, part_data_pl['OldToNew'][part_offset], data_query, comm)
    part_offset += 1


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

def _update_vtx_subset(zone, ngon, vtx_distri_ini, old_to_new_vtx, comm):
  #MAJ connectivité NGON
  # On va chercher le old2new qui correspond aux noeuds des faces du NGON
  ngon_ec_n = I.getNodeFromName1(ngon, 'ElementConnectivity')
  dist_data = {'OldToNew' : old_to_new_vtx}
  part_data = MBTP.dist_to_part(vtx_distri_ini, dist_data, [ngon_ec_n[1].astype(pdm_dtype)], comm)
  assert len(ngon_ec_n[1]) == len(part_data['OldToNew'][0])
  I.setValue(ngon_ec_n, part_data['OldToNew'][0])


def merge_intrazone_jn(dist_tree, jn_pathes, comm):
  """
  """
  base_n, zone_n, zgc_n, gc_n = jn_pathes[0].split('/')
  zone = I.getNodeFromPath(dist_tree, base_n + '/' + zone_n)
  ngon = [elem for elem in I.getNodesFromType1(zone, 'Elements_t') if elem[1][0] == 22][0]

  ref_faces      = I.getNodeFromPath(dist_tree, jn_pathes[0]+'/PointList')[1][0]
  face_to_remove = I.getNodeFromPath(dist_tree, jn_pathes[0]+'/PointListDonor')[1][0]
  #Create pl and pl_d from vertex, needed to know which vertex will be deleted
  ref_vtx, vtx_to_remove, _ = VL.generate_jn_vertex_list(dist_tree, jn_pathes[0], comm)

  #Get initial distributions
  face_distri_ini = IE.getDistribution(ngon, 'Element').astype(pdm_dtype)
  vtx_distri_ini  = IE.getDistribution(zone, 'Vertex').astype(pdm_dtype)

  old_to_new_face = merge_distributed_ids(face_distri_ini, face_to_remove, ref_faces, comm)
  old_to_new_vtx  = merge_distributed_ids(vtx_distri_ini, vtx_to_remove, ref_vtx, comm)

  _update_ngon(ngon, ref_faces, face_to_remove, comm)
  _update_face_subsets(zone, face_distri_ini, old_to_new_face, comm)

  _update_vtx_data(zone, vtx_to_remove, comm)
  _update_vtx_subset(zone, ngon, vtx_distri_ini, old_to_new_vtx, comm)

  #Todo update JN/PointListDonor if any

  #Cleanup
  I._rmNodeByPath(dist_tree, jn_pathes[0])
  I._rmNodeByPath(dist_tree, jn_pathes[1])
  

  """
  for fs in IE.getChildrenFromPredicate(zone, is_face_sol):
    pl_new = part_data_pl['OldToNew'][part_offset]
    part_data = {}
    dist_data = {}
    for data in IE.getNodesWithParentsByMatching(fs, ['DataArray_t']):
      path = "/".join([I.getName(n) for n in data])
      leave = data[-1]
      part_data[path] = [leave[1][0]]

    #Dont use maia interface since we need a new distribution
    PTB = PDM.PartToBlock(comm, [pl_new.astype(pdm_dtype)], pWeight=None, partN=1,
                          t_distrib=0, t_post=1, t_stride=0)
    PTB.PartToBlock_Exchange(dist_data, part_data)

    d_pl_new = PTB.getBlockGnumCopy()

    new_distri_full = par_utils.gather_and_shift(len(d_pl_new), comm)
    new_distri = new_distri_full[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

    #Update distribution and size
    IE.newDistribution({'Index' : new_distri}, fs)
    I.getNodeFromPath(fs, 'PointList#Size')[1][1] = new_distri[-1]
      
    #Update PointList and data
    I.createUniqueChild(fs, 'PointList', 'IndexArray_t', d_pl_new.reshape(1,-1, order='F'))
    for data in I.getNodesFromType1(fs, 'DataArray_t'):
      I.setValue(data, dist_data[I.getName(data)])

    part_offset += 1

  #BC & BCDataSet
  for bc in IE.getNodesByMatching(zone, ['ZoneBC_t', is_face_bc]):
    pl_new = part_data_pl['OldToNew'][part_offset]

    # Here we manage BCData which has same PL
    part_data = dict()
    dist_data = dict()
    related_bcds = lambda n: I.getType(n) == 'BCDataSet_t' and I.getNodeFromName(n, 'PointList') is None
    for bcds, bcdata, data in IE.getNodesWithParentsByMatching(bc, [related_bcds, 'BCData_t', 'DataArray_t']):
      path = "/".join([I.getName(n) for n in [bcds, bcdata, data]])
      part_data[path] = [data[1][0]]

    #Dont use maia interface since we need a new distribution
    PTB = PDM.PartToBlock(comm, [pl_new.astype(pdm_dtype)], pWeight=None, partN=1,
                          t_distrib=0, t_post=1, t_stride=0)
    PTB.PartToBlock_Exchange(dist_data, part_data)

    d_pl_new = PTB.getBlockGnumCopy()

    new_distri_full = par_utils.gather_and_shift(len(d_pl_new), comm)
    new_distri = new_distri_full[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

    #Update distribution and size
    IE.newDistribution({'Index' : new_distri}, bc)
    I.getNodeFromPath(bc, 'PointList#Size')[1][1] = new_distri[-1]
      
    #Update PointList and data
    I.createUniqueChild(bc, 'PointList', 'IndexArray_t', d_pl_new.reshape(1,-1, order='F'))
    for bcds, bcdata, data in IE.getNodesWithParentsByMatching(bc, [related_bcds, 'BCData_t', 'DataArray_t']):
      path = "/".join([I.getName(n) for n in [bcds, bcdata, data]])
      I.setValue(data, dist_data[path].reshape(1,-1, order='F'))

    part_offset += 1


  # Solo BCDataSet
  for bcds in IE.getNodesByMatching(zone, ['ZoneBC_t', 'BC_t', is_face_bcds]):
    pl_new = part_data_pl['OldToNew'][part_offset]

    # Here we manage BCData which has same PL
    part_data = dict()
    dist_data = dict()
    for bcdata, data in IE.getNodesWithParentsByMatching(bcds, ['BCData_t', 'DataArray_t']):
      path = "/".join([I.getName(n) for n in [bcdata, data]])
      part_data[path] = [data[1][0]]

    #Dont use maia interface since we need a new distribution
    PTB = PDM.PartToBlock(comm, [pl_new.astype(pdm_dtype)], pWeight=None, partN=1,
                          t_distrib=0, t_post=1, t_stride=0)
    PTB.PartToBlock_Exchange(dist_data, part_data)

    d_pl_new = PTB.getBlockGnumCopy()

    new_distri_full = par_utils.gather_and_shift(len(d_pl_new), comm)
    new_distri = new_distri_full[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

    #Update distribution and size
    IE.newDistribution({'Index' : new_distri}, bcds)
    I.getNodeFromPath(bcds, 'PointList#Size')[1][1] = new_distri[-1]
      
    #Update PointList and data
    I.createUniqueChild(bcds, 'PointList', 'IndexArray_t', d_pl_new.reshape(1,-1, order='F'))
    for bcdata, data in IE.getNodesWithParentsByMatching(bcds, ['BCData_t', 'DataArray_t']):
      path = "/".join([I.getName(n) for n in [bcdata, data]])
      I.setValue(data, dist_data[path].reshape(1,-1, order='F'))

    part_offset += 1
  """
