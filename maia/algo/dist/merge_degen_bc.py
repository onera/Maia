import copy
import itertools
import mpi4py.MPI as MPI
import numpy      as np
import os

import maia
import maia.algo.dist.merge_jn          as MJN
import maia.algo.part.point_cloud_utils as PCU
import maia.pytree                      as PT
import maia.pytree.maia                 as MT

from maia                     import npy_pdm_gnum_dtype    as pdm_gnum_dtype
from maia.algo.dist           import remove_element        as RME
from maia.algo.dist.merge_ids import merge_distributed_ids
from maia.transfer            import protocols             as EP
from maia.utils               import par_utils, np_utils

import Pypdm.Pypdm as PDM




# TO DO: define where to store
def distribute_unique_vtx_ids_from_face_ids(pl_faces, ngon_n, comm):
  """
  Get only unique nodes of faces in list and distribute it over all procs uniformly
  """
  # Get the nodes ids of all faces in pl_faces
  __, nodes_pl = maia.algo.dist.vertex_list.face_ids_to_vtx_ids(pl_faces, ngon_n, comm)
  # Make unique
  PTB = EP.PartToBlock(None, [nodes_pl], comm, weight=True, keep_multiple=False)
  nodes_pl = PTB.getBlockGnumCopy()
  # Because result could be badly distributed, redistribute it
  distrib_nodes_pl_init   = par_utils.gather_and_shift(len(nodes_pl), comm, pdm_gnum_dtype)
  distrib_nodes_pl_wanted = par_utils.uniform_distribution(distrib_nodes_pl_init[-1], comm)
  return EP.block_to_block(nodes_pl, distrib_nodes_pl_init, distrib_nodes_pl_wanted, comm)

def _remove_ids_in_ESO(poly, func, comm):
  """
  Remove entity ids defined by function in EC and update ESO
  Works for ngon or nface nodes
  """
  poly_eso_n = PT.get_child_from_name(poly, 'ElementStartOffset')
  poly_ec_n  = PT.get_child_from_name(poly, 'ElementConnectivity')
  poly_eso   = PT.get_value(poly_eso_n)
  poly_ec    = PT.get_value(poly_ec_n)
  new_poly_eso = np.zeros(len(poly_eso), dtype=pdm_gnum_dtype)
  new_poly_ec  = []
  _poly_eso = poly_eso - poly_eso[0]
  for n in range(len(poly_eso)-1):
    poly_ec_tmp = func(poly_ec[_poly_eso[n]:_poly_eso[n+1]])
    new_poly_ec.append(poly_ec_tmp)
    new_poly_eso[n+1] = new_poly_eso[n] + len(poly_ec_tmp)
  if new_poly_ec: new_poly_ec = np.concatenate(new_poly_ec)
  size_new_poly_ec = new_poly_eso[-1]
  size_new_poly_ec_per_proc = comm.allgather(size_new_poly_ec)
  shift_new_poly_eso = int(np.sum(size_new_poly_ec_per_proc[:comm.rank]))
  size_new_poly_eso_total = int(np.sum(size_new_poly_ec_per_proc[:comm.size])) 
  new_poly_eso += shift_new_poly_eso
  PT.set_value(poly_eso_n, new_poly_eso)
  PT.set_value(poly_ec_n, new_poly_ec)
  distrib_face_vtx_n = PT.maia.getDistribution(poly, 'ElementConnectivity')
  PT.set_value(distrib_face_vtx_n, [new_poly_eso[0], new_poly_eso[-1], size_new_poly_eso_total])

def _remove_dup_ids_in_ESO(poly, comm):
  """
  Remove duplicated entity ids in EC and update ESO
  Works for ngon or nface nodes
  """
  def unique_ids(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]
  
  _remove_ids_in_ESO(poly, unique_ids, comm)

def _remove_id_in_ESO(poly, index, comm) :
  """
  Remove id in EC and update ESO
  Works for ngon or nface nodes
  """
  def remove_index(array):
    index_to_remove = np.where(array == index)
    return np.delete(array, index_to_remove)
  
  _remove_ids_in_ESO(poly, remove_index, comm)

# ------------------------------------------------------------------------------------------
def _update_ngon(ngon, del_faces, vtx_distri_ini, old_to_new_vtx, comm):
  """
  Update ngon node after face and vertex merging, ie
   - remove faces from EC, PE and ESO and update distribution info
   - update ElementConnectivity using vertex old_to_new order
   - remove duplicated vertex ids in EC and update ESO
  """

  # A/ Update EC, PE and ESO removing some faces
  MJN._update_ngon_remove_faces(ngon, del_faces, comm)

  # B/ Update vertex ids in EC
  MJN._update_ngon_update_EC(ngon, vtx_distri_ini, old_to_new_vtx, comm)
  
  # C/ Delete vertex ids duplicates in EC and update ESO
  _remove_dup_ids_in_ESO(ngon, comm)

# ------------------------------------------------------------------------------------------
def _update_nface(nface, face_distri_ini, old_to_new_face, n_rmvd_face, comm):
  """
  Update nface node after face merging, ie
   - update ElementConnectivity using face old_to_new order
   - Shift ElementRange (to substract nb of removed faces) if NFace is after NGon
   - remove duplicated face ids in EC and update ESO
  If input array old_to_new_face is signed, then the orientation of nface connectivity is preserved
  """
  # A/ Update nface node after face merging
  MJN._update_nface(nface, face_distri_ini, old_to_new_face, n_rmvd_face, comm)
  
  # B/ # Delete fake face numbered "nb_faces + 1" in EC and update ESO
  fake_face_num = face_distri_ini[2] - n_rmvd_face
  _remove_id_in_ESO(nface, fake_face_num, comm) 

# ------------------------------------------------------------------------------------------
def delete_degen_faces_for_one_zone(dist_tree, zone_path, pl_degen_faces, pl_degen_nodes_kept, degen_subset_names, comm):
  """
  In a zone, this function delete all degenerated faces store in a ZoneSubRegion
  and update all nodes of the zone accept PointListDonor
  """
  base_name, zone_name = zone_path.split('/')
  base_n = PT.get_child_from_name(dist_tree, base_name)
  zone_n = PT.get_child_from_name(base_n, zone_name)
  ngon_n = PT.Zone.NGonNode(zone_n)
  
  # Define distribution 
  #> for nodes kept in degenerated faces
  full_distri_nodes_kept = par_utils.gather_and_shift(len(pl_degen_nodes_kept), comm)
  partial_distri_nodes_kept = full_distri_nodes_kept[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]
  #> for all unique nodes in degenerated faces
  nodes_from_degen_faces = distribute_unique_vtx_ids_from_face_ids(pl_degen_faces, ngon_n, comm)
  full_distri_all_nodes_degen_faces = par_utils.gather_and_shift(len(nodes_from_degen_faces), comm)
  partial_distri_all_nodes_degen_faces = full_distri_all_nodes_degen_faces[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]
  
  # Find closest kept node for each nodes of degenerated faces
  #> mimic fake partition to use _closest_points on distributed nodes clouds
  src_lngn = np.arange(partial_distri_nodes_kept[0], partial_distri_nodes_kept[1], dtype=pdm_gnum_dtype) + 1
  tgt_lngn = np.arange(partial_distri_all_nodes_degen_faces[0], partial_distri_all_nodes_degen_faces[1], dtype=pdm_gnum_dtype) + 1
  cx, cy, cz = PT.Zone.coordinates(zone_n)
  distri_vtx = PT.get_value(MT.getDistribution(zone_n, 'Vertex'))
  part_data_coords = EP.block_to_part({'cx':cx, 'cy': cy, 'cz': cz}, distri_vtx, [pl_degen_nodes_kept, nodes_from_degen_faces], comm)
  src_coords = np_utils.interweave_arrays([part_data_coords[c][0] for c in ['cx', 'cy', 'cz']])
  tgt_coords = np_utils.interweave_arrays([part_data_coords[c][1] for c in ['cx', 'cy', 'cz']])
  closest_src_gnum = maia.algo.part.closest_points._closest_points([(src_coords, src_lngn)], [(tgt_coords, tgt_lngn)], comm)[0]['closest_src_gnum']
  
  # Find old to new global numbering for nodes of degenerated faces
  #> compute old to new global numbering for each nodes of degenerated faces
  ptp = PDM.PartToPart(comm, [tgt_lngn], [src_lngn], [np.arange(len(closest_src_gnum)+1,dtype=closest_src_gnum.dtype)], [closest_src_gnum])
  request1 = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P, PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2, [pl_degen_nodes_kept])
  _, part_data = ptp.reverse_wait(request1)
  old_to_new_degen_faces_nodes = part_data[0]
  
  # Work only on a copy of the considered zone !
  shallow_zone_n  = copy.deepcopy(PT.get_child_from_name(base_n, zone_name))
  PT.rm_nodes_from_name(base_n, zone_name)
  shallow_ngon_n  = PT.Zone.NGonNode(shallow_zone_n)
  
  # Identify nodes to remove
  ref_vtx         = np.unique(old_to_new_degen_faces_nodes)
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
  old_to_new_vtx_to_rm_from_degen_faces = np.delete(old_to_new_degen_faces_nodes, index_to_remove)
  old_to_new_vtx  = merge_distributed_ids(vtx_distri_ini, vtx_to_remove, old_to_new_vtx_to_rm_from_degen_faces, comm)
  #> update ngon node
  _update_ngon(shallow_ngon_n, face_to_remove, vtx_distri_ini, old_to_new_vtx, comm)
  
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
  old_to_new_face_to_remove = (nb_faces+1)*np.ones(len(face_to_remove), dtype=pdm_gnum_dtype)
  old_to_new_face = merge_distributed_ids(face_distri_ext, face_to_remove, old_to_new_face_to_remove, comm)
  #> update nface node
  nface_n = PT.Zone.NFaceNode(shallow_zone_n)
  if nface_n:
    _update_nface(nface_n, face_distri_ext, old_to_new_face, n_rmvd_face, comm)
  
  # Update all data stored at 'Vertex' except in subsets
  MJN._update_vtx_data(shallow_zone_n, vtx_to_remove, comm)
  
  # Update all data stored at 'Vertex' in subsets
  MJN._update_cgns_subsets(shallow_zone_n, 'Vertex', vtx_distri_ini, old_to_new_vtx, base_name, comm)
  
  # Shift all CellCenter PL by the number of removed faces
  if PT.Element.Range(shallow_ngon_n)[0] == 1:
    MJN._shift_cgns_subsets(shallow_zone_n, 'CellCenter', -n_rmvd_face)
  
  # Update all data stored at 'FaceCenter' in subsets
  old_to_new_face_unsg = np.abs(old_to_new_face)
  MJN._update_cgns_subsets(shallow_zone_n, 'FaceCenter', face_distri_ext, old_to_new_face_unsg, base_name, comm)
  
  # TO DO: delete in all FaceCenter* PL all "nb_faces+1" numbered face
  # for now: del degen_bc
  # PT.print_tree(PT.get_node_from_path(shallow_zone_n, f'ZoneBC/{degen_bc_name}'))
  # exit()
  for degen_subset_name in degen_subset_names:
    PT.rm_node_from_path(shallow_zone_n, f'ZoneBC/{degen_subset_name}')
    PT.rm_node_from_path(shallow_zone_n, f'{degen_subset_name}')
  
  # Add zone to base
  PT.add_child(base_n, shallow_zone_n)
  
  # Update PointList/PointListDonor of other zones
  MJN._update_pl_pld_in_jn(dist_tree, zone_path)

# ------------------------------------------------------------------------------------------
def delete_degen_faces_from_family(dist_tree, fam_to_remove, fam_for_intersection, comm):
  """
  Delete all faces of family named fam_to_removed and keep only nodes that are shared with
  family named fam_for_intersection
  """
  
  base_n = PT.get_child_from_label(dist_tree, 'CGNSBase_t')
  
  for zone_n in PT.get_nodes_from_label(dist_tree, 'Zone_t'):
    
    pl_degen_faces           = np.empty(0, dtype=pdm_gnum_dtype)
    pl_intersect_degen_faces = np.empty(0, dtype=pdm_gnum_dtype)
    degen_subset_names       = []
    intersect_subset_names   = []
    for bc_n in PT.get_nodes_from_label(zone_n, 'BC_t'):
      fam_n = PT.get_node_from_label(bc_n, 'FamilyName_t')
      if fam_n is not None:
        fam = PT.get_value(fam_n)
        if fam == fam_to_remove:
          pl_degen_faces = np.append(pl_degen_faces, PT.get_value(PT.Subset.getPatch(bc_n))[0])
          degen_subset_names.append(PT.get_name(bc_n))
        elif fam == fam_for_intersection:
          pl_intersect_degen_faces = np.append(pl_intersect_degen_faces, PT.get_value(PT.Subset.getPatch(bc_n))[0])
          intersect_subset_names.append(PT.get_name(bc_n))
    for zsr_n in PT.get_nodes_from_label(zone_n, 'ZoneSubRegion_t'):
      fam_n = PT.get_node_from_label(zsr_n, 'FamilyName_t')
      if fam_n is not None:
        fam = PT.get_value(fam_n)
        if fam == fam_to_remove:
          if PT.get_node_from_name(zsr_n, 'PointList') is None:
            zsr_extent_path = PT.Subset.ZSRExtent(zsr_n, zone_n)
            zsr_extent_n = PT.get_node_from_path(zone_n, zsr_extent_path)
            pl_degen_faces = np.append(pl_degen_faces, PT.get_value(PT.Subset.getPatch(zsr_extent_n))[0])
          else:
            pl_degen_faces = np.append(pl_degen_faces, PT.get_value(PT.Subset.getPatch(zsr_n))[0])
          degen_subset_names.append(PT.get_name(zsr_n))
        elif fam == fam_for_intersection:
          if PT.Subset.getPatch(zsr_n) is None:
            zsr_extent_path = PT.Subset.ZSRExtent(zsr_n, zone_n)
            zsr_extent_n = PT.get_node_from_path(zone_n, zsr_extent_path)
            pl_intersect_degen_faces = np.append(pl_intersect_degen_faces, PT.get_value(PT.Subset.getPatch(zsr_extent_n))[0])
          else:
            pl_intersect_degen_faces = np.append(pl_intersect_degen_faces, PT.get_value(PT.Subset.getPatch(zsr_n))[0])
          intersect_subset_names.append(PT.get_name(zsr_n))
    if len(degen_subset_names) == 0:
      continue
    if len(intersect_subset_names) == 0:
      print("Error : 'intersect_degen_bc' is not defined !")
      exit()
    
    # List with unique nodes
    ngon_n = PT.Zone.NGonNode(zone_n)
    nodes_degen_faces           = distribute_unique_vtx_ids_from_face_ids(pl_degen_faces,           ngon_n, comm)
    nodes_intersect_degen_faces = distribute_unique_vtx_ids_from_face_ids(pl_intersect_degen_faces, ngon_n, comm)
    
    nodes_degen_faces_in = maia.utils.parallel.algo.gnum_isin(nodes_degen_faces,nodes_intersect_degen_faces, comm)
    degen_nodes_kept = nodes_degen_faces[nodes_degen_faces_in]
    
    zone_path = PT.get_name(base_n)+'/'+PT.get_name(zone_n)
    
    delete_degen_faces_for_one_zone(dist_tree, zone_path, pl_degen_faces, degen_nodes_kept, degen_subset_names, comm)
