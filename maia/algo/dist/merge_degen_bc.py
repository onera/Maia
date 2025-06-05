import numpy      as np
import mpi4py.MPI as MPI

import maia
from maia.typing import *
import maia.pytree           as PT
import maia.pytree.maia      as MT
import Pypdm.Pypdm           as PDM
from maia                    import npy_pdm_gnum_dtype    as pdm_gnum_dtype
from maia.transfer           import protocols             as EP
from maia.utils              import par_utils, np_utils, vstride
from maia.utils.parallel     import algo as par_algo

from .merge_ids      import merge_distributed_ids
from .vertex_list    import face_ids_to_vtx_ids
from .geometry.utils import get_local_coordinates

from maia.algo.dist  import merge_jn       as MJN


def distribute_unique_vtx_ids_from_face_ids(vtx_distri, pl_faces, ngon_n, comm):
  """
  Get only unique nodes of faces in list and distribute it over all procs uniformly
  """
  # Get the nodes ids of all faces in pl_faces
  nodes_pl = face_ids_to_vtx_ids(pl_faces, ngon_n, comm).values
  # Make unique
  GI = EP.GlobalIndexer(vtx_distri, nodes_pl, comm, gnum_offset=1)
  nodes_pl = np.flatnonzero(GI.access_counts > 0).astype(nodes_pl.dtype, copy=False) + vtx_distri[0] + 1

  # Because result could be badly distributed, redistribute it
  distrib_nodes_pl_init   = par_utils.gather_and_shift(nodes_pl.size, comm, pdm_gnum_dtype)
  distrib_nodes_pl_wanted = par_utils.uniform_distribution(distrib_nodes_pl_init[-1], comm)
  return EP.block_to_block(nodes_pl, distrib_nodes_pl_init, distrib_nodes_pl_wanted, comm)

def _remove_dup_ids_in_ESO(poly, comm):
  """
  Remove entity ids defined by function in EC and update ESO
  Works for ngon or nface nodes
  """
  poly_eso_n = PT.get_child_from_name(poly, 'ElementStartOffset')
  poly_ec_n  = PT.get_child_from_name(poly, 'ElementConnectivity')
  
  new_poly = vstride.unique(MT.Element.connectivity(poly), vstride.INNER_AXIS)
  
  ec_distri = par_utils.dn_to_distribution(new_poly.dsize, comm)

  # Update arrays and distribution
  PT.set_value(poly_eso_n, new_poly.displs + ec_distri[0])
  PT.set_value(poly_ec_n,  new_poly.values)
  MT.new_Distribution({'ElementConnectivity': ec_distri}, poly)

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
def _remove_subset_fictive_faces(zone, comm):
  """
  """

  def update_node(node):
    # Because of previous func, pl is sorted
    # so fake face to remove (if any) if on the last rank knowing data
    pl_n = PT.get_child_from_name(node, 'PointList')
    pl = PT.get_value(pl_n)
    has_last = pl[0,-1] == n_face+1 if pl.size > 0 else False
    is_empty = False
    if comm.allreduce(has_last, MPI.LOR):
      # Update distribution, on all ranks
      distri_entity = MT.distribution_value(node, 'Index')
      mask = (distri_entity == distri_entity[2])
      distri_entity[mask] -= 1

      if distri_entity[2] == 0: # Global PointList is empty, no need to update DataArray because node will be delete
        is_empty = True
      elif has_last: # Update PointList and DataArray for the rank having it (remove last elt)
        PT.set_value(pl_n, pl[:,:-1])
        da_labels = ['BCData_t', 'DataArray_t'] if PT.get_label(node) == 'BCDataSet_t' else ['DataArray_t']
        for data_n in PT.get_children_from_labels(node, da_labels):
          PT.set_value(data_n, PT.get_value(data_n)[:-1])
    return is_empty

  #Trick to add a PL to each subregion to be able to use same algo
  for zsr in PT.get_children_from_label(zone, 'ZoneSubRegion_t'):
    zsr_extent = PT.Subset.ZSRExtent(zsr, zone)
    zsr_extent_n = PT.find_node_from_path(zone, zsr_extent)
    if  zsr_extent != PT.get_name(zsr):
      PT.add_child(zsr, PT.deep_copy(PT.find_child_from_name(zsr_extent_n, 'PointList')))
      PT.add_child(zsr, PT.deep_copy(MT.find_Distribution(zsr_extent_n)))
  #Trick to add a PL to DataSet (for same reason)
  for _,bc,bcds in PT.get_children_from_labels(zone, ['ZoneBC_t', 'BC_t', 'BCDataSet_t'], ancestors=True):
    if PT.get_child_from_name(bcds, 'PointList') is None:
      PT.new_GridLocation(PT.Subset.GridLocation(bc), bcds)
      PT.add_child(bcds, PT.deep_copy(PT.find_child_from_name(bc, 'PointList')))
      PT.add_child(bcds, PT.deep_copy(MT.find_Distribution(bc)))
      MT.new_Distribution({'FakeDistri' : None}, bcds)

  n_face = PT.Zone.n_face(zone)

  fs_like = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t']
  is_facecenter = lambda n: PT.Subset.GridLocation(n) == 'FaceCenter'
  is_fs = lambda n: PT.get_label(n) in fs_like              and is_facecenter(n)
  is_bc = lambda n: PT.get_label(n) == 'BC_t'               and is_facecenter(n)
  is_gc = lambda n: PT.get_label(n) == 'GridConnectivity_t' and is_facecenter(n)
  is_ds = lambda n: PT.get_label(n) == 'BCDataSet_t'        and is_facecenter(n)
  for node in PT.get_children_from_predicate(zone, is_fs):
    if update_node(node): # Update node is executed, and return a flag if node must be removed
      PT.rm_child(zone, node)
  for zone_gc in PT.get_children_from_label(zone, 'ZoneGridConnectivity_t'):
    for gc in PT.get_children_from_predicate(zone_gc, is_gc):
      if update_node(gc): # Update node is executed, and return a flag if node must be removed
        PT.rm_child(zone_gc, gc)
  for zone_bc in PT.get_children_from_label(zone, 'ZoneBC_t'):
    for bc in PT.get_children_from_predicate(zone_bc, is_bc):
      if update_node(bc): # Update node is executed, and return a flag if node must be removed
        PT.rm_child(zone_bc, bc)
      else:
        for bcds in PT.get_children_from_predicate(bc, is_ds):
          if update_node(bcds):
            PT.rm_child(bc, bcds)

  #Cleanup after trick
  for zsr in PT.get_children_from_label(zone, 'ZoneSubRegion_t'):
    if PT.Subset.ZSRExtent(zsr, zone) != PT.get_name(zsr):
      PT.rm_children_from_name(zsr, 'PointList')
      PT.rm_children_from_name(zsr, ':CGNS#Distribution')
  name_to_remove = ['GridLocation', 'PointList', ':CGNS#Distribution']
  for bcds in PT.get_children_from_labels(zone, ['ZoneBC_t', 'BC_t', 'BCDataSet_t']):
    if MT.get_Distribution(bcds, 'FakeDistri') is not None:
      PT.rm_children_from_predicate(bcds, lambda n : PT.get_name(n) in name_to_remove)


# ------------------------------------------------------------------------------------------
def remove_degen_faces_for_one_zone(dist_tree, zone_path, pl_degen_faces, pl_degen_vtx, comm):
  """
  In a zone, this function delete all degenerated faces store in a ZoneSubRegion
  and update all nodes of the zone accept PointListDonor
  """
  base_name = zone_path.split('/')[0]
  zone_n = PT.get_node_from_path(dist_tree, zone_path)
  ngon_n = PT.Zone.NGonNode(zone_n)
  
  part_data_coords = get_local_coordinates(zone_n, pl_degen_vtx, comm)
  tgt_coords = np_utils.interweave_arrays(part_data_coords)

  # Find vertices have same coords (using gnum, they will have same id in output)
  pdm_gnum = PDM.GlobalNumbering(3, 1, True, 1E-10, comm)
  pdm_gnum.set_from_coords(0, tgt_coords, np.ones(pl_degen_vtx.size))
  pdm_gnum.compute()
  merged_id = pdm_gnum.get(0)
  # In each group, choose any and map others to it
  distri   = par_utils.uniform_distribution(comm.allreduce(merged_id.max(initial=0), MPI.MAX), comm)
  GI = EP.GlobalIndexer(distri, merged_id, comm, gnum_offset=1)
  selected_vtx_id = GI.Put(pl_degen_vtx, reduce=EP.ReduceOp.MAX)
  old_to_new_degen_faces_nodes = GI.Take(selected_vtx_id)
  
  # Identify nodes to remove (nodes of degen face not beloging to old_to_new)
  remove_mask = par_algo.gnum_isin(pl_degen_vtx, np.unique(old_to_new_degen_faces_nodes), comm, invert=True)
  vtx_to_remove   = pl_degen_vtx[remove_mask] # Nodes not belonging to intersection
  
  # Identify faces to remove
  face_to_remove = pl_degen_faces
  n_rmvd_face    = comm.allreduce(len(face_to_remove), op=MPI.SUM)
  
  # Need dto copy face distribution before it changes in ngon update !!!
  vtx_distri_ini  = MT.distribution_value(zone_n, 'Vertex').copy()
  face_distri_ini = MT.distribution_value(ngon_n, 'Element').copy()
  
  # Update ngon node
  #> define old to new global numbering for nodes of degenerated faces to remove
  old_to_new_vtx_to_rm_from_degen_faces = old_to_new_degen_faces_nodes[remove_mask]
  old_to_new_vtx  = merge_distributed_ids(vtx_distri_ini, vtx_to_remove, old_to_new_vtx_to_rm_from_degen_faces, comm)
  #> update ngon node
  _update_ngon(ngon_n, face_to_remove, vtx_distri_ini, old_to_new_vtx, comm)
  
  # Trick : Create a old_to_new_face which map all removed faces to a fictive face
  # numbered nb_face+1, so we can identifiate it afterward and remove the
  # associated data
  nb_faces_ini = face_distri_ini[2]
  mask = face_distri_ini == nb_faces_ini
  face_distri_ini[mask] += 1
  old_to_new_face_to_remove = (nb_faces_ini+1)*np.ones_like(face_to_remove)
  old_to_new_face = merge_distributed_ids(face_distri_ini, face_to_remove, old_to_new_face_to_remove, comm)
  
  #> update nface node if existes
  if PT.Zone.has_nface_elements(zone_n):
    nface_n = PT.Zone.NFaceNode(zone_n)
    PT.rm_child(zone_n, nface_n)
    maia.algo.pe_to_nface(zone_n, comm)
  
  # Update all data stored at 'Vertex' except in subsets
  MJN._update_vtx_data(zone_n, vtx_to_remove, comm)
  
  # Update all data stored at 'Vertex' in subsets
  MJN._update_cgns_subsets(zone_n, 'Vertex', vtx_distri_ini, old_to_new_vtx, base_name, comm)
  
  # Shift all CellCenter PL by the number of removed faces
  if PT.Element.Range(ngon_n)[0] == 1:
    MJN._shift_cgns_subsets(zone_n, 'CellCenter', -n_rmvd_face)
  
  # Update all data stored at 'FaceCenter' in subsets
  MJN._update_cgns_subsets(zone_n, 'FaceCenter', face_distri_ini, old_to_new_face, base_name, comm)
  _remove_subset_fictive_faces(zone_n, comm)
  
  # Update PointList/PointListDonor of other zones
  MJN._update_pl_pld_in_jn(dist_tree, zone_path)

# ------------------------------------------------------------------------------------------
def remove_degen_faces_from_family(dist_tree: CGNSDistTree, 
                                   degen_family: str, 
                                   comm: MPIComm) -> None:
  """
  Remove the specified degenerated faces in the input tree.

  Degenerated faces are faces whose vertices have distinct ids, but are in fact geometrically
  reduced to a line or to a single point.
  This function removes these faces and update the mesh to renumber the other entities.
  Input tree is modified inplace.

  Important:
    - Faces refered by ``degen_family`` **must** be degenerated faces, and will be removed anyway.
    - Only U-NGon meshes are managed in this function.

  Args:
    dist_tree  (CGNSDistTree) : Input distributed tree, with U-NGon connectivies
    degen_family (str): Name of the family refering to the degenerated faces
    comm       (`MPIComm`)    : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #remove_degen_faces_from_family@start
        :end-before: #remove_degen_faces_from_family@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  for zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    zone_n = PT.find_node_from_path(dist_tree, zone_path)
    vtx_distri = MT.distribution_value(zone_n, 'Vertex')
    
    pl_degen_faces_list = []
    for bc_n in PT.get_children_from_labels(zone_n, ['ZoneBC_t', 'BC_t']):
      if PT.pred.belongs_to_family(degen_family)(bc_n):
        pl_degen_faces_list.append(PT.get_np_value(PT.Subset.getPatch(bc_n)))
    for zsr_n in PT.get_children_from_label(zone_n, 'ZoneSubRegion_t'):
      zsr_extent_path = PT.Subset.ZSRExtent(zsr_n, zone_n)
      zsr_extent_n = PT.find_node_from_path(zone_n, zsr_extent_path)
      if PT.pred.belongs_to_family(degen_family)(zsr_n):
        pl_degen_faces_list.append(PT.get_np_value(PT.Subset.getPatch(zsr_extent_n)))
    if len(pl_degen_faces_list) == 0:
      continue
    
    _, pl_degen_faces = np_utils.concatenate_point_list(pl_degen_faces_list, pdm_gnum_dtype)

    # List with unique nodes
    nodes_degen_faces = distribute_unique_vtx_ids_from_face_ids(vtx_distri, pl_degen_faces, PT.Zone.NGonNode(zone_n), comm)
    
    remove_degen_faces_for_one_zone(dist_tree, zone_path, pl_degen_faces, nodes_degen_faces, comm)
