import mpi4py.MPI as MPI

import numpy as np
import copy
import os

import maia
import maia.pytree as PT

from maia.algo.dist.merge_ids import merge_distributed_ids
from maia                     import npy_pdm_gnum_dtype    as pdm_gnum_dtype

import Pypdm.Pypdm as PDM
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
from maia.transfer  import protocols      as EP
from maia.algo.dist import remove_element as RME

import maia.pytree.maia                 as MT
import maia.algo.part.point_cloud_utils as PCU

from maia       import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import par_utils

import itertools





def distribute_unique_vtx_ids_from_face_ids(pl_faces, ngon_n, comm):
  # Get the nodes ids of all faces in pl_faces
  __, nodes_pl = maia.algo.dist.vertex_list.face_ids_to_vtx_ids(pl_faces, ngon_n, comm)
  # Make unique
  PTB = EP.PartToBlock(None, [nodes_pl], comm, weight=True, keep_multiple=False)
  nodes_pl = PTB.getBlockGnumCopy()
  # Because result could be badly distributed, redistribute it
  distrib_nodes_pl_init   = par_utils.gather_and_shift(len(nodes_pl), comm, pdm_gnum_dtype)
  distrib_nodes_pl_wanted = par_utils.uniform_distribution(distrib_nodes_pl_init[-1], comm)
  return EP.block_to_block(nodes_pl, distrib_nodes_pl_init, distrib_nodes_pl_wanted, comm)



# ------------------------------------------------------------------------------------------
# Inspirer de _new_update_ngon(ngon_n, ref_faces, face_to_remove, vtx_distri_ini, old_to_new_vtx, comm)
# Gerer EC et ESO differemment car suppression de noeuds pas uniquement remplacement !
def _new_update_ngon(ngon, del_faces, vtx_distri_ini, old_to_new_vtx, comm):
  """
  Update ngon node after face and vertex merging, ie
   - remove faces from EC, PE and ESO and update distribution info
   - update ElementConnectivity using vertex old_to_new order
  """
  
  face_distri = PT.get_value(PT.maia.getDistribution(ngon, 'Element'))
  pe          = PT.get_node_from_path(ngon, 'ParentElements')[1]


  #TODO This method asserts that PE is CGNS compliant ie left_parent != 0 for bnd elements
  assert not np.any(pe[:,0] == 0)

  # A/ Update EC, PE and ESO removing some faces
  # PT.print_tree(ngon)
  ngon_ec_n = PT.get_child_from_name(ngon, 'ElementConnectivity')
  # print([comm.rank, f"BEFORE len(ngon_ec_n[1]) = {len(ngon_ec_n[1])}"])
  part_data = [del_faces]
  dist_data = EP.part_to_block(part_data, face_distri, [del_faces], comm)
  local_faces = dist_data - face_distri[0] - 1
  RME.remove_ngons(ngon, local_faces, comm)
  # print("Step A done !")
  # PT.print_tree(ngon)
  # exit()

  # B/ Update vertex ids in EC
  ngon_ec_n = PT.get_child_from_name(ngon, 'ElementConnectivity')
  # print([comm.rank, f"len(old_to_new_vtx) = {len(old_to_new_vtx)}"])
  # print([comm.rank, f"vtx_distri_ini = {vtx_distri_ini}"])
  # print([comm.rank, f"AFTER len(ngon_ec_n[1]) = {len(ngon_ec_n[1])}"])
  # exit()
  part_data = EP.block_to_part(old_to_new_vtx, vtx_distri_ini, [PT.get_value(ngon_ec_n)], comm)
  assert len(ngon_ec_n[1]) == len(part_data[0])
  PT.set_value(ngon_ec_n, part_data[0])
  # print(len(np.unique(part_data[0])))
  # print([comm.rank, f"len(part_data[0]) = {len(part_data[0])}", (4*4*5*3-16)*4])
  # print(len(part_data[0]), 224*4-5)
  # exit()
  # print("Step B done !")
  
  # C/ Delete vertex ids duplicates in EC and update ESO
  ngon_eso_n = PT.get_child_from_name(ngon, 'ElementStartOffset')
  ngon_eso   = PT.get_value(ngon_eso_n)
  ngon_ec    = PT.get_value(ngon_ec_n)
  # print([comm.rank, f"AFTER2 len(ngon_ec_n[1]) = {len(ngon_ec_n[1])}"])
  # print([comm.rank, f"ngon_ec_n[1] = {ngon_ec_n[1]}"])
  # print([comm.rank, f"ngon_eso_n[1] = {ngon_eso_n[1]}"])
  # print(len(ngon_eso)-1)
  new_ngon_ec = []
  new_ngon_eso = np.zeros(len(ngon_eso), dtype=np.int32)
  for n in range(len(ngon_eso)-1):
    # if comm.rank==1: print(ngon_eso[n]-ngon_eso[0], ngon_eso[n+1]-ngon_eso[0])
    ngon_ec_tmp = ngon_ec[ngon_eso[n]-ngon_eso[0]:ngon_eso[n+1]-ngon_eso[0]]
    # if comm.rank==1: print(ngon_ec_tmp)
    # exit()
    _, idx = np.unique(ngon_ec_tmp, return_index=True)
    ngon_ec_tmp = ngon_ec_tmp[np.sort(idx)]
    new_ngon_ec.append(ngon_ec_tmp)
    new_ngon_eso[n+1] = new_ngon_eso[n] + len(ngon_ec_tmp)
    # print(ngon_ec_tmp)
  if new_ngon_ec: new_ngon_ec = np.concatenate(new_ngon_ec)
  # print([comm.rank, f"AFTER3 len(new_ngon_ec) = {len(new_ngon_ec)}"])
  # print([comm.rank, f"new_ngon_ec = {new_ngon_ec}"])
  # print([comm.rank, f"new_ngon_eso = {new_ngon_eso}"])
  # print([comm.rank, f"size_new_ngon_ec = {new_ngon_eso[-1]}"])
  size_new_ngon_ec = new_ngon_eso[-1]
  size_new_ngon_ec_per_proc = comm.allgather(size_new_ngon_ec)
  # print([comm.rank, f"size_new_ngon_ec_per_proc = {size_new_ngon_ec_per_proc}"])
  shift_new_ngon_eso = int(np.sum(size_new_ngon_ec_per_proc[:comm.rank]))
  size_new_ngon_eso_total = int(np.sum(size_new_ngon_ec_per_proc[:comm.size])) 
  new_ngon_eso += shift_new_ngon_eso
  # print([comm.rank, f"shift_new_ngon_eso = {shift_new_ngon_eso}"])
  # new_ngon_eso += 
  PT.set_value(ngon_eso_n, new_ngon_eso)
  PT.set_value(ngon_ec_n, new_ngon_ec)
  # exit()
  # print(new_ngon_eso)
  #TO DO : exchange len(new_ngon_ec) to all rank to update new_ngon_eso and distribution
  distrib_face_vtx_n = PT.maia.getDistribution(ngon, 'ElementConnectivity')
  PT.set_value(distrib_face_vtx_n, [new_ngon_eso[0], new_ngon_eso[-1], size_new_ngon_eso_total])
  # print([0, new_ngon_eso[-1], new_ngon_eso[-1]])
  # print([comm.rank, f"distrib_face_vtx_n = {[new_ngon_eso[0], new_ngon_eso[-1], size_new_ngon_eso_total]}"])
  # print(ngon_n)
  # print("Step C done !")

# ------------------------------------------------------------------------------------------
def _new_update_nface(nface, face_distri_ini, old_to_new_face, n_rmvd_face, comm):
  """
  Update nface node after face merging, ie
   - update ElementConnectivity using face old_to_new order
   - Shift ElementRange (to substract nb of removed faces) if NFace is after NGon
  If input array old_to_new_face is signed (ie is negative for face ids that will be removed),
  then the orientation of nface connectivity is preserved
  """

  #Update list of faces
  nface_ec_n = PT.get_child_from_name(nface, 'ElementConnectivity')
  part_data = EP.block_to_part(old_to_new_face, face_distri_ini, [np.abs(nface_ec_n[1])], comm)
  assert len(nface_ec_n[1]) == len(part_data[0])
  #Get sign of nface_ec to preserve orientation
  PT.set_value(nface_ec_n, np.sign(nface_ec_n[1]) * part_data[0])

  #Update ElementRange
  er = PT.Element.Range(nface)
  if er[0] > 1:
    er -= n_rmvd_face
    
  # Delete fake face numbered "nb_faces + 1"
  nface_eso_n = PT.get_child_from_name(nface, 'ElementStartOffset')
  nface_eso   = PT.get_value(nface_eso_n)
  nface_ec    = PT.get_value(nface_ec_n)
  # print(nface_ec)
  # print(len(nface_eso)-1)
  new_nface_ec = []
  new_nface_eso = np.zeros(len(nface_eso), dtype=np.int32)
  fake_face_num = face_distri_ini[2] - n_rmvd_face
  # print([comm.rank, f"n_rmvd_face = {n_rmvd_face}"])
  # print([comm.rank, f"fake_face_num = {fake_face_num}"])
  # print(fake_face_num)
  # exit()
  for n in range(len(nface_eso)-1):
    nface_ec_tmp = nface_ec[nface_eso[n]-nface_eso[0]:nface_eso[n+1]-nface_eso[0]]
    # print(nface_ec_tmp)
    # exit()
    # print(nface_ec_tmp)
    # if fake_face_num in nface_ec_tmp:
      # print(f"SB : fake_face_num {fake_face_num} in nface_ec_tmp")
    index_to_remove = np.where(nface_ec_tmp == fake_face_num)
    nface_ec_tmp   = np.delete(nface_ec_tmp, index_to_remove)
    # print(nface_ec_tmp)
    # print("---")
    new_nface_ec.append(nface_ec_tmp)
    new_nface_eso[n+1] = new_nface_eso[n] + len(nface_ec_tmp)
    # print(nface_ec_tmp)
  if new_nface_ec: new_nface_ec = np.concatenate(new_nface_ec)
  # print(new_nface_ec)
  size_new_nface_ec = new_nface_eso[-1]
  size_new_nface_ec_per_proc = comm.allgather(size_new_nface_ec)
  shift_new_nface_eso = int(np.sum(size_new_nface_ec_per_proc[:comm.rank]))
  size_new_nface_eso_total = int(np.sum(size_new_nface_ec_per_proc[:comm.size])) 
  new_nface_eso += shift_new_nface_eso
  # print([comm.rank, f"shift_new_nface_eso = {shift_new_nface_eso}"])
  PT.set_value(nface_eso_n, new_nface_eso)
  PT.set_value(nface_ec_n, new_nface_ec)
  # exit()
  # print(new_nface_eso)
  #TO DO : exchange len(new_nface_ec) to all rank to update new_nface_eso and distribution
  distrib_cell_face_n = PT.maia.getDistribution(nface, 'ElementConnectivity')
  # PT.set_value(distrib_cell_face_n, [0, new_nface_eso[-1], new_nface_eso[-1]])
  PT.set_value(distrib_cell_face_n, [new_nface_eso[0], new_nface_eso[-1], size_new_nface_eso_total])
  # print(nface_n)
  # print("Step C done !")

# ------------------------------------------------------------------------------------------
def _new_update_vtx_data(zone, vtx_to_remove, comm):
  """
  Remove the vertices in data array supported by allVertex (currently
  managed : GridCoordinates, FlowSolution, DiscreteData)
  and update vertex distribution info
  """
  vtx_distri_ini  = PT.get_value(MT.getDistribution(zone, 'Vertex'))
  pdm_distrib     = par_utils.partial_to_full_distribution(vtx_distri_ini, comm)

  PTB = EP.PartToBlock(vtx_distri_ini, [vtx_to_remove], comm)
  local_vtx_to_rmv = PTB.getBlockGnumCopy() - vtx_distri_ini[0] - 1

  #Update all vertex entities
  for coord_n in PT.iter_children_from_predicates(zone, ['GridCoordinates_t', 'DataArray_t']):
    PT.set_value(coord_n, np.delete(coord_n[1], local_vtx_to_rmv))

  is_all_vtx_sol = lambda n: PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t'] \
      and PT.Subset.GridLocation(n) == 'Vertex' and PT.get_node_from_path(n, 'PointList') is None

  for node in PT.iter_children_from_predicate(zone, is_all_vtx_sol):
    for data_n in PT.iter_children_from_label(node, 'DataArray_t'):
      PT.set_value(data_n, np.delete(data_n[1], local_vtx_to_rmv))

  # Update vertex distribution
  i_rank, n_rank = comm.Get_rank(), comm.Get_size()
  n_rmvd   = len(local_vtx_to_rmv)
  n_rmvd_offset  = par_utils.gather_and_shift(n_rmvd, comm, pdm_dtype)
  vtx_distri = vtx_distri_ini - [n_rmvd_offset[i_rank], n_rmvd_offset[i_rank+1],  n_rmvd_offset[n_rank]]
  MT.newDistribution({'Vertex' : vtx_distri}, zone)
  zone[1][0][0] = vtx_distri[2]

# ------------------------------------------------------------------------------------------
def _new_update_subset(node, pl_new, data_query, comm):
  """
  Update a PointList and all the data
  """
  part_data = {}
  dist_data = {}
  for data_nodes in PT.iter_children_from_predicates(node, data_query, ancestors=True):
    path = "/".join([PT.get_name(n) for n in data_nodes])
    data_n = data_nodes[-1]
    if data_n[1].ndim == 1:
      part_data[path] = [data_n[1]]
    else:
      assert data_n[1].ndim == 2 and data_n[1].shape[0] == 1
      part_data[path] = [data_n[1][0]]

  #Add PL, needed for next blocktoblock
  pl_identifier = r'@\PointList/@' # just a string that is unlikely to clash
  part_data[pl_identifier] = [pl_new]

  PTB = EP.PartToBlock(None, [pl_new], comm)
  PTB.PartToBlock_Exchange(dist_data, part_data)

  d_pl_new = PTB.getBlockGnumCopy()

  new_distri_full = par_utils.gather_and_shift(len(d_pl_new), comm, pdm_dtype)
  #Result is badly distributed, we can do a BlockToBlock to have a uniform distribution
  ideal_distri      = par_utils.uniform_distribution(new_distri_full[-1], comm)
  dist_data_ideal = EP.block_to_block(dist_data, new_distri_full, ideal_distri, comm)

  #Update distribution and size
  MT.newDistribution({'Index' : ideal_distri}, node)

  #Update PointList and data
  PT.update_child(node, 'PointList', 'IndexArray_t', dist_data_ideal.pop(pl_identifier).reshape(1,-1, order='F'))
  #Update data
  for data_nodes in PT.iter_children_from_predicates(node, data_query, ancestors=True):
    path = "/".join([PT.get_name(n) for n in data_nodes])
    if PT.get_label(data_nodes[-1]) == 'IndexArray_t':
      PT.set_value(data_nodes[-1], dist_data_ideal[path].reshape(1,-1, order='F'))
    elif PT.get_label(data_nodes[-1]) == 'DataArray_t':
      PT.set_value(data_nodes[-1], dist_data_ideal[path])

# ------------------------------------------------------------------------------------------
def _new_update_cgns_subsets(zone, location, entity_distri, old_to_new_face, base_name, comm):
  """
  Treated for now :
    BC, BCDataset (With or without PL), FlowSol, DiscreteData, ZoneSubRegion, JN

    Careful! PointList/PointListDonor arrays of joins present in the zone are updated, but opposite joins
    are not informed of this modification. This has to be done after the function.
  """

  # Prepare iterators
  matches_loc = lambda n : PT.Subset.GridLocation(n) == location
  is_bcds_with_pl    = lambda n: PT.get_label(n) == 'BCDataSet_t'and PT.get_child_from_name(n, 'PointList') is not None
  is_bcds_without_pl = lambda n: PT.get_label(n) == 'BCDataSet_t'and PT.get_child_from_name(n, 'PointList') is None

  is_sol  = lambda n: PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t'] and matches_loc(n) 
  is_bc   = lambda n: PT.get_label(n) == 'BC_t' and matches_loc(n) 
  is_bcds = lambda n: is_bcds_with_pl(n) and matches_loc(n) 
  is_zsr  = lambda n: PT.get_label(n) == 'ZoneSubRegion_t' and matches_loc(n) 
  is_jn   = lambda n: PT.get_label(n) == 'GridConnectivity_t' and matches_loc(n) 

  sol_list  = PT.getChildrenFromPredicate(zone, is_sol)
  bc_list   = PT.getChildrenFromPredicates(zone, ['ZoneBC_t', is_bc])
  bcds_list = PT.getChildrenFromPredicates(zone, ['ZoneBC_t', 'BC_t', is_bcds])
  zsr_list  = PT.getChildrenFromPredicate(zone, is_zsr)
  jn_list   = PT.getChildrenFromPredicates(zone, ['ZoneGridConnectivity_t', is_jn])
  i_jn_list = [jn for jn in jn_list if PT.getZoneDonorPath(base_name, jn) == base_name + '/'+ PT.get_name(zone)]

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
      PT.add_child(zsr, PT.get_node_from_path(zone, PT.Subset.ZSRExtent(zsr, zone) + '/PointList'))

  #Get new index for every PL at once
  all_pl_list = [PT.get_child_from_name(fs, 'PointList')[1][0] for fs in all_nodes]
  part_data_pl = EP.block_to_part(old_to_new_face, entity_distri, all_pl_list, comm)

  part_offset = 0
  for node_list, data_query in all_nodes_and_queries:
    for node in node_list:
      _new_update_subset(node, part_data_pl[part_offset], data_query, comm)
      part_offset += 1

  #For internal jn only, we must update PointListDonor with new face id. Non internal jn reorder the array,
  # but do not apply old_to_new transformation.
  # Note that we will lost symmetry PL/PLD for internal jn, we need a rule to update it afterward
  all_pld = [PT.get_child_from_name(jn, 'PointListDonor') for jn in i_jn_list]
  updated_pld = EP.block_to_part(old_to_new_face, entity_distri, [pld[1][0] for pld in all_pld], comm)
  for i, pld in enumerate(all_pld):
    PT.set_value(pld, updated_pld[i].reshape((1,-1), order='F'))

  #Cleanup after trick
  for zsr in zsr_list:
    if PT.Subset.ZSRExtent(zsr, zone) != PT.get_name(zsr):
      PT.rm_children_from_name(zsr, 'PointList')


# ------------------------------------------------------------------------------------------
# TODO move to sids module, doc, unit test
#(take the one of _shift_cgns_subsets, and for _shift_cgns_subsets, make a trivial test)
def new_all_nodes_with_point_list(zone, pl_location):
  has_pl = lambda n: PT.get_child_from_name(n, 'PointList') is not None \
                     and PT.Subset.GridLocation(n) == pl_location
  return itertools.chain(
      PT.getChildrenFromPredicate(zone, has_pl)                      , #FlowSolution_t, ZoneSubRegion_t, ...
      PT.getChildrenFromPredicates(zone, ['ZoneBC_t', has_pl])              , #BC_t
      #For this one we must exclude BC since predicate is also tested on root (and should not be ?)
      PT.getChildrenFromPredicates(zone, ['ZoneBC_t', 'BC_t', lambda n : has_pl(n) and PT.get_label(n) != 'BC_t'])      , #BCDataSet_t
      PT.getChildrenFromPredicates(zone, ['ZoneGridConnectivity_t', has_pl]), #GridConnectivity_t
    )

# ------------------------------------------------------------------------------------------
def _new_shift_cgns_subsets(zone, location, shift_value):
  """
  Shift all the PointList of the requested location with the given value
  PointList are seached in every node below zone, + in BC_t, BCDataSet_t,
  GridConnectivity_t
  """
  for node in new_all_nodes_with_point_list(zone,location):
    PT.get_child_from_name(node, 'PointList')[1][0] += shift_value

# ------------------------------------------------------------------------------------------
def delete_degen_faces_for_one_zone(dist_tree, zone_name, new_dist_tree, pl_degen_faces, pl_degen_nodes_kept, degen_bc_name, comm):
  """
  In a zone, this function delete all degenerated faces store in ZoneSubRegion
  and update all nodes of the zone accept PointListDonor
  For now, this function is not inplace !!!
  """
  
  zone_n = PT.get_node_from_name_and_label(dist_tree, zone_name, 'Zone_t')
  ngon_n = PT.Zone.NGonNode(zone_n)
  
  # Defined private ZoneSubRegions to store mandatory information
  zsr0_n = PT.new_ZoneSubRegion(name='__maia_degen_faces', point_list=[pl_degen_faces], loc='FaceCenter', parent=zone_n)
  full_distri0 = par_utils.gather_and_shift(len(pl_degen_faces), comm)
  partial_distri0 = full_distri0[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]
  PT.maia.newDistribution({"Index" : partial_distri0}, zsr0_n)
  
  zsr1_n = PT.new_ZoneSubRegion(name='__maia_degen_nodes_kept', point_list=[pl_degen_nodes_kept], loc='Vertex', parent=zone_n)
  full_distri1 = par_utils.gather_and_shift(len(pl_degen_nodes_kept), comm)
  partial_distri1 = full_distri1[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]
  PT.maia.newDistribution({"Index" : partial_distri1}, zsr1_n)
    
  nodes_from_degen_faces = distribute_unique_vtx_ids_from_face_ids(pl_degen_faces, ngon_n, comm)
  zsr2_n = PT.new_ZoneSubRegion(name='__maia_degen_nodes', point_list=[nodes_from_degen_faces], loc='Vertex', parent=zone_n)
  full_distri2 = par_utils.gather_and_shift(len(nodes_from_degen_faces), comm)
  partial_distri2 = full_distri2[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]
  PT.maia.newDistribution({"Index" : partial_distri2}, zsr2_n)
  
  # Work only on a copy of the considered zone !
  new_base_n = PT.get_node_from_label(new_dist_tree, 'CGNSBase_t')
  shallow_dist_tree = PT.shallow_copy(dist_tree)
  PT.rm_nodes_from_predicate(
    shallow_dist_tree,
    lambda n: PT.get_label(n)=='Zone_t' and PT.get_name(zone_n) not in PT.get_name(n))
  
  # Partition volumic mesh
  part_tree_ngon = maia.factory.partition_dist_tree(shallow_dist_tree, comm)
  
  # Extract kept nodes of degenerated faces from volumic mesh
  pzone_n = PT.get_node_from_label(part_tree_ngon, 'Zone_t')  
  extractor_degen_line = maia.algo.part.extract_part.create_extractor_from_zsr(part_tree_ngon, '__maia_degen_nodes_kept', comm)
  part_tree_degen_line = extractor_degen_line.get_extract_part_tree()
  pzone_degen_line_n = PT.get_node_from_label(part_tree_degen_line, 'Zone_t')
  dom_name = PT.get_name(PT.get_node_from_label(shallow_dist_tree, 'Zone_t'))
  dom_path = f'{PT.get_name(PT.get_all_CGNSBase_t(part_tree_ngon)[0])}/{dom_name}'
  parent_vtx = extractor_degen_line.exch_tool_box[dom_path]['parent_elt']['Vertex']
  
  # Find closest kept node for each nodes of volumic mesh
  maia.algo.part.find_closest_points(part_tree_degen_line, part_tree_ngon, 'Vertex', comm)
  # Verification que les noeuds donnes sont a une distance inferieure a une tolerance ?
  
  # Find old to new global numbering for nodes of degenerated faces
  #> get link between each nodes of volume mesh and kept nodes (in the extract numbering)
  try:
    src_id = PT.get_value(PT.get_node_from_name(part_tree_ngon, 'SrcId'))
  except TypeError:
    src_id = np.empty(0, dtype=pdm_gnum_dtype)
  #> get nodes of degenerated faces in partitionned zone
  if pzone_n is None:
    p_nodes_from_degen_faces = np.zeros((0), dtype=np.int32)
  else:
    pzsr2_n = PT.get_node_from_name(pzone_n, '__maia_degen_nodes')
    if pzsr2_n is None:
      p_nodes_from_degen_faces = np.zeros((0), dtype=np.int32)
    else:
      p_nodes_from_degen_faces = PT.get_value(PT.get_node_from_name(pzsr2_n, 'PointList'))[0]
  #> get global numbering of each nodes of volume mesh
  if pzone_n is None:
    part1_gnum_vtx = np.empty(0, dtype=pdm_gnum_dtype)
  else:
    part1_gnum_vtx = PT.get_value(MT.getGlobalNumbering(pzone_n, 'Vertex'))
  #> get global numbering of each kept nodes
  if pzone_degen_line_n is None:
    part2_gnum_vtx = np.empty(0, dtype=pdm_gnum_dtype)
  else:
    part2_gnum_vtx = PT.get_value(MT.getGlobalNumbering(pzone_degen_line_n, 'Vertex'))
  #> compute old to new global numbering for each nodes of volumic mesh
  ptp = PDM.PartToPart(comm, [part1_gnum_vtx], [part2_gnum_vtx], [np.arange(len(src_id)+1,dtype=src_id.dtype)], [src_id])
  request1 = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P, PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2, [parent_vtx])
  _, part_data = ptp.reverse_wait(request1)
  #> keep only old to new global numbering for nodes of degenerated faces
  old_to_new_degen_bc_nodes = part_data[0][p_nodes_from_degen_faces-1]
  #> store this information in ZSR to transfer it from part to dist
  pzsr_n = PT.new_ZoneSubRegion(name='__maia_degen_nodes_new', point_list=[p_nodes_from_degen_faces], loc='Vertex', parent=pzone_n)
  PT.new_DataArray("OldToNew", old_to_new_degen_bc_nodes, parent=pzsr_n)
  if pzone_n is None:
    local_gnum = np.empty(0, dtype=pdm_gnum_dtype)
  else:
    local_gnum = PT.get_value(MT.getGlobalNumbering(pzone_n, 'Vertex'))[p_nodes_from_degen_faces-1]
  zsr_gnum = PCU.create_sub_numbering([local_gnum], comm)[0]
  PT.maia.newGlobalNumbering({'Index' : zsr_gnum}, parent=pzsr_n)
  #> transfer
  maia.transfer.part_tree_to_dist_tree_only_labels(shallow_dist_tree, part_tree_ngon, ['ZoneSubRegion_t'], comm)
  
  # Get information in dist
  #> from zone and ngon nodes
  shallow_zone_n  = PT.get_node_from_label(shallow_dist_tree, 'Zone_t')
  shallow_ngon_n  = PT.Zone.NGonNode(shallow_zone_n)
  #> from degenerated faces
  degen_faces_n = PT.get_node_from_name(shallow_zone_n, '__maia_degen_faces')
  pl_degen_faces = PT.get_value(PT.get_node_from_name(degen_faces_n, 'PointList'))[0]
  #> from nodes of degenerated faces
  zsr_n = PT.get_node_from_name(shallow_zone_n, '__maia_degen_nodes_new')
  old_to_new_all_vtx_from_degen_faces = PT.get_value(PT.get_node_from_name(zsr_n, 'OldToNew'))
  distrib_old_to_new_all_vtx_from_degen_faces = PT.get_value(PT.maia.getDistribution(zsr_n, 'Index')).copy()
  
  # Identify nodes to remove
  ref_vtx         = np.unique(old_to_new_all_vtx_from_degen_faces)
  in_or_not = maia.utils.parallel.algo.gnum_isin(nodes_from_degen_faces,ref_vtx, comm)
  index_to_remove = np.where(in_or_not == True)
  vtx_to_remove   = np.delete(nodes_from_degen_faces, index_to_remove)
  
  # Identify faces to remove
  face_to_remove = pl_degen_faces
  n_rmvd_face    = comm.allreduce(len(face_to_remove), op=MPI.SUM)
  
  # Need dto copy face distribution before it changes in ngon update !!!
  face_distri_ini = PT.get_value(PT.maia.getDistribution(shallow_ngon_n, 'Element')).copy()
  
  # Update ngon node
  #> define old to new global numbering for nodes of degenerated faces to remove
  vtx_distri_ini  = PT.get_value(PT.maia.getDistribution(shallow_zone_n, 'Vertex'))
  old_to_new_vtx_to_rm_from_degen_faces = np.delete(old_to_new_all_vtx_from_degen_faces, index_to_remove)
  old_to_new_vtx  = merge_distributed_ids(vtx_distri_ini, vtx_to_remove, old_to_new_vtx_to_rm_from_degen_faces, comm)
  #> update ngon node
  _new_update_ngon(shallow_ngon_n, face_to_remove, vtx_distri_ini, old_to_new_vtx, comm)
  
  # Update nface node
  # Trick : because some faces are removed, in the distribution, we create a new
  # face with number nb_faces + 1
  #> change face distribution
  nb_faces = face_distri_ini[2]
  face_distri_ext = copy.deepcopy(face_distri_ini)
  if face_distri_ext[1] == nb_faces:
    face_distri_ext[1] += 1
  face_distri_ext[2] += 1
  #> define old to new global numbering for degenerated faces to remove
  old_to_new_face_to_remove = (nb_faces+1)*np.ones(len(face_to_remove), dtype=np.int32)
  old_to_new_face = merge_distributed_ids(face_distri_ext, face_to_remove, old_to_new_face_to_remove, comm)
  #> update nface node
  nface_n = PT.Zone.NFaceNode(shallow_zone_n)
  if nface_n:
    _new_update_nface(nface_n, face_distri_ext, old_to_new_face, n_rmvd_face, comm)
  
  # Update all data stored at 'Vertex' except in subsets
  _new_update_vtx_data(shallow_zone_n, vtx_to_remove, comm)
  
  base_name = PT.get_name(PT.get_node_from_label(shallow_dist_tree, "CGNSBase_t"))
  
  # Update all data stored at 'Vertex' in subsets
  _new_update_cgns_subsets(shallow_zone_n, 'Vertex', vtx_distri_ini, old_to_new_vtx, base_name, comm)
  
  # Shift all CellCenter PL by the number of removed faces
  if PT.Element.Range(shallow_ngon_n)[0] == 1:
    _new_shift_cgns_subsets(shallow_zone_n, 'CellCenter', -n_rmvd_face)
  
  # Update all data stored at 'FaceCenter' in subsets
  old_to_new_face_unsg = np.abs(old_to_new_face)
  _new_update_cgns_subsets(shallow_zone_n, 'FaceCenter', face_distri_ext, old_to_new_face_unsg, base_name, comm)
  
  # TO DO: delete in all FaceCenter* PL all "nb_faces+1" numbered face
  # for now: del degen_bc
  # PT.print_tree(PT.get_node_from_path(shallow_zone_n, f'ZoneBC/{degen_bc_name}'))
  # exit()
  PT.rm_node_from_path(shallow_zone_n, f'ZoneBC/{degen_bc_name}')
  
  # Suppression des artefacts d'algo
  PT.rm_nodes_from_name(shallow_zone_n,'__maia_degen_*')
  
  # Add zone to base
  PT.add_child(new_base_n, shallow_zone_n)

