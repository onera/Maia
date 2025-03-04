import numpy as np

import maia

import maia.pytree      as PT
import maia.pytree.maia as MT
from maia.pytree.sids import elements_utils as EU

from maia.utils     import np_utils, par_utils, vstride
from maia.transfer  import protocols as EP
from maia.algo.dist import matching_jns_tools as MJT


from cmaia.algo import combine_to_tetra, combine_to_pyra, \
                       combine_to_penta, combine_to_hexa

is_poly_3d = lambda z: PT.Zone.CellDimension(z) == 3 and PT.Zone.has_ngon_elements(z)
is_poly_2d = lambda z: PT.Zone.CellDimension(z) == 2 and \
                        PT.Zone.Type(z) == 'Unstructured' and \
                        all(PT.Element.CGNSName(e) in ['BAR_2', 'NGON_n'] for e in PT.get_children_from_label(z, 'Elements_t'))

is_cell_full_container = lambda n : PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t'] and \
                                    PT.get_child_from_name(n, 'PointList') is None and \
                                    PT.get_child_from_name(n, 'PointRange') is None and \
                                    PT.Subset.GridLocation(n) == 'CellCenter'

def _collected_shifted_pl(zone, loc, shift):
  all_pl = []
  for subset in PT.iter_all_subsets(zone, loc):
    if (pl := PT.get_child_from_name(subset, 'PointList')) is not None:
      _pl = pl[1][0]
    elif (pr := PT.get_child_from_name(subset, 'PointRange')) is not None:
      distri = MT.getDistribution(subset, 'Index')[1]
      _pl = np_utils.single_dim_pr_to_pl(pr[1], distri)[0]
    all_pl.append(_pl + shift)
  return all_pl

def _update_pl(zone, loc, new_pl):
  for subset, _pl in zip(PT.iter_all_subsets(zone, loc), new_pl):
    PT.rm_children_from_name(subset, 'PointList')
    PT.rm_children_from_name(subset, 'PointRange')
    PT.new_IndexArray(value=_pl.reshape((1,-1), order='F'), parent=subset)
    # NB : PointListDonor of GCs will be copied afterward (under usual assumption that PL are symmetric) 
  
def _ngon_to_elements_zone_2d(zone, comm):
  """ Implementation of conversion for 2d zones. We assume that input zones
      are poly2d with BAR (+PE) and NGON node """

  # Start by constructing boundary edges
  edge_n = MT.Zone.EdgeNode(zone)
  
  edge_vtx     = PT.get_child_from_name(edge_n, 'ElementConnectivity')[1]
  pe           = PT.get_child_from_name(edge_n, 'ParentElements')[1]
  edge_distri  = MT.getDistribution(edge_n, 'Element')[1]

  edge_distri_f = par_utils.partial_to_full_distribution(edge_distri, comm)

  old_edge_pl = _collected_shifted_pl(zone, 'EdgeCenter', -PT.Element.Range(edge_n)[0])
  GI = EP.GlobalMultiIndexer(edge_distri_f, old_edge_pl, comm)
  is_subset_edge = (GI.access_counts > 0)

  is_bnd_edge = (pe[:,1] == 0)  | (is_subset_edge)
  bar_vtx = np.empty(2*is_bnd_edge.sum(), edge_vtx.dtype)
  bar_vtx[0::2] = edge_vtx[0::2][is_bnd_edge]
  bar_vtx[1::2] = edge_vtx[1::2][is_bnd_edge]


  bar_distri  = par_utils.dn_to_distribution(bar_vtx.size // 2, comm)

  bar_range  = np.array([1, bar_distri[-1]], dtype=zone[1].dtype)
  if bar_distri[-1] > 0:
    bar_n = PT.new_Elements('BAR_2', 'BAR_2', erange=bar_range, econn=bar_vtx, parent=zone)
    MT.newDistribution({'Element' : bar_distri}, bar_n)
  
  # Renumber PointList indexing Edges
  new_edge_id = -1*np.ones(edge_vtx.size // 2, zone[1].dtype)
  new_edge_id[is_bnd_edge] = np.arange(bar_distri[0]+1, bar_distri[1]+1)

  new_pl = GI.Take(new_edge_id)
  _update_pl(zone, 'EdgeCenter', new_pl)

  # Now take care of the faces
  ngon_n = PT.Zone.NGonNode(zone)

  face_distri = MT.getDistribution(ngon_n, 'Element')[1]
  face_vtx    = MT.Element.connectivity(ngon_n)
  n_face_loc  = len(face_vtx)

  n_treated = 0
  elt_shift = bar_range[1]
  mask = np.empty(n_face_loc, bool)
  new_face_id = np.empty(n_face_loc, zone[1].dtype) # For subset renumbering
  for elt_kind, target_size in zip(['TRI_3', 'QUAD_4'], [3,4]):
    # Find corresponding faces
    np.equal(face_vtx.counts, target_size, out=mask)
    face_ids = np.flatnonzero(mask)
    elt_conn = vstride.take(face_vtx, face_ids).values

    # Prepare elt node (ElementConnectivity will be computed later)
    n_elt_loc = face_ids.size
    distri = par_utils.dn_to_distribution(n_elt_loc, comm)
    if distri[-1] > 0:
      elt = PT.new_Elements(elt_kind, elt_kind,
                            erange=np.array([elt_shift+1, elt_shift+distri[-1]], zone[1].dtype),
                            econn=elt_conn,
                            parent=zone)
      MT.new_distribution({'Element' : distri}, elt)

    # Prepare renum table for faces
    new_face_id[mask] = np.arange(distri[0]+elt_shift+1, distri[1]+elt_shift+1)

    n_treated += n_elt_loc
    elt_shift += distri[-1]

  remaining_faces = comm.allreduce(n_face_loc - n_treated)
  if remaining_faces != 0:
    msg = f"Input 2d polyedric mesh can not be converted to standard elements, because some faces differs from TRI_3 or QUAD_4" \
          f" standard elements ({remaining_faces} faces detected on zone {PT.get_name(zone)})"
    raise RuntimeError(msg)


  # Renumber PointList indexing faces (CellCenter)
  all_pl = _collected_shifted_pl(zone, 'CellCenter', -PT.Element.Range(ngon_n)[0])
  # This last one is for fields supported by allCells (eg. FlowSolution)
  _pl = np_utils.single_dim_pr_to_pl(np.array([[0, PT.Element.Size(ngon_n)-1]]), face_distri)[0]
  all_pl.append(_pl)

  new_pl = EP.block_to_part(new_face_id, face_distri, all_pl, comm)

  # Update CellCentered PointList
  _update_pl(zone, 'CellCenter', new_pl[:-1])

  # For allCells containers, we need an additional exchange to reorder data in cell_distri order
  face_distri_f = par_utils.partial_to_full_distribution(face_distri, comm)
  GI = EP.GlobalIndexer(face_distri_f, new_pl[-1]-bar_range[1]-1, comm)

  for path in PT.predicates_to_paths(zone, [is_cell_full_container, 'DataArray_t']):
    data = PT.get_node_from_path(zone, path)[1]
    GI.Put(data, data) # Inplace update of node data

  # Remove NGON/Edge elements
  PT.rm_child(zone, edge_n)
  PT.rm_child(zone, ngon_n)



def _ngon_to_elements_zone_3d(zone, comm):
  """ Implementation of conversion for 3d zones. We assume that input zones
      are poly3d with NGON (+PE) and NFACE node """

  # Start by constructing boundary faces
  ngon_n = PT.Zone.NGonNode(zone)
  
  face_vtx     = MT.Element.connectivity(ngon_n)
  pe           = PT.get_child_from_name(ngon_n, 'ParentElements')[1]
  face_distri  = MT.getDistribution(ngon_n, 'Element')[1]
  dn_face   = len(face_vtx)

  face_distri_f = par_utils.partial_to_full_distribution(face_distri, comm)
  face_n = face_vtx.counts

  old_face_pl = _collected_shifted_pl(zone, 'FaceCenter', -PT.Element.Range(ngon_n)[0])
  
  # This is to detect faces that are indexed by some PL, in addition to boundary faces
  GI = EP.GlobalMultiIndexer(face_distri_f, old_face_pl, comm)
  is_subset_face = (GI.access_counts > 0)
  
  is_bnd_face = (pe[:,1] == 0) | (is_subset_face)

  is_bnd_tri  = (is_bnd_face) & (face_n == 3)
  is_bnd_quad = (is_bnd_face) & (face_n == 4)
  tri_vtx = vstride.take(face_vtx, np.where(is_bnd_tri)[0]).values
  quad_vtx = vstride.take(face_vtx, np.where(is_bnd_quad)[0]).values


  tri_distri  = par_utils.dn_to_distribution(tri_vtx.size  // 3, comm)
  quad_distri = par_utils.dn_to_distribution(quad_vtx.size // 4, comm)

  tri_range  = np.array([1, tri_distri[-1]], dtype=zone[1].dtype)
  quad_range = np.array([1, quad_distri[-1]], dtype=zone[1].dtype) + tri_range[-1]
  if tri_distri[-1] > 0:
    tri_n = PT.new_Elements('TRI_3', 'TRI_3', erange=tri_range, econn=tri_vtx, parent=zone)
    MT.newDistribution({'Element' : tri_distri}, tri_n)
  if quad_distri[-1] > 0:
    quad_n = PT.new_Elements('QUAD_4', 'QUAD_4', erange=quad_range, econn=quad_vtx, parent=zone)
    MT.newDistribution({'Element' : quad_distri}, quad_n)
  
  # Renumber PointList indexing Faces
  new_face_id = -1*np.ones(dn_face, zone[1].dtype)
  new_face_id[is_bnd_tri] = np.arange(tri_distri[0]+1, tri_distri[1]+1)
  new_face_id[is_bnd_quad] = np.arange(quad_distri[0]+tri_distri[-1]+1, quad_distri[1]+tri_distri[-1]+1)

  new_pl = GI.Take(new_face_id)
  _update_pl(zone, 'FaceCenter', new_pl)

  # Now take care of the cells 
  nface_n = PT.Zone.NFaceNode(zone)
  cell_face     = MT.Element.connectivity(nface_n)
  cell_distri   = MT.getDistribution(nface_n, 'Element')[1]
  dn_cell = len(cell_face)

  # Design choice : get the number of vertices (with reps) for **all** cells,
  # thus we can check if elements seems to be standard. Otherwise, we could
  # do it only for cells having 5 faces to resolve prism / pyra ambiguity
  cell_nvtx_per_face = EP.block_to_part(face_n, face_distri, np.abs(cell_face.values)-1, comm)
  cell_nvtx_tot = np.add.reduceat(cell_nvtx_per_face, cell_face.displs[:-1])

  n_treated = 0
  elt_shift = quad_range[1]
  cell_face_section = []
  mask = np.empty(dn_cell, bool)
  new_cell_id = np.empty(dn_cell, zone[1].dtype) # For subset renumbering
  # Gather (locally) the element per kind. In addition we prepare the renumbering table for cells
  for elt_kind, target_size in zip(['TETRA_4', 'PYRA_5', 'PENTA_6', 'HEXA_8'], [12, 16, 18, 24]):
    # Find corresponding cells
    np.equal(cell_nvtx_tot, target_size, out=mask)
    # Extract cell_face for this section
    cell_ids = np.nonzero(mask)[0]
    cell_face_section.append(vstride.take(cell_face, cell_ids).values)
    
    # Prepare elt node (ElementConnectivity will be computed later)
    n_elt_loc = cell_ids.size
    distri = par_utils.dn_to_distribution(n_elt_loc, comm)
    if distri[-1] > 0:
      n_vtx_per_elt = EU.element_number_of_nodes(EU.cgns_name_to_id(elt_kind))
      elt = PT.new_Elements(elt_kind, elt_kind,
                            erange=np.array([elt_shift+1, elt_shift+distri[-1]], zone[1].dtype),
                            econn=np.empty(n_vtx_per_elt*n_elt_loc, zone[1].dtype),
                            parent=zone)
      MT.new_distribution({'Element' : distri}, elt)
    
    # Prepare renum table for cells
    new_cell_id[mask] = np.arange(distri[0]+elt_shift+1, distri[1]+elt_shift+1)

    n_treated += n_elt_loc
    elt_shift += distri[-1]

  remaining_cells = comm.allreduce(dn_cell - n_treated)
  if remaining_cells != 0:
    msg = f"Input polyedric mesh can not be converted to standard elements, because some cells differs from standard elements" \
          f" TETRA_4, PYRA_5, PENTA_6 or HEXA_8 ({remaining_cells} cells detected on zone {PT.get_name(zone)})"
    raise RuntimeError(msg)

  # Now get for each cell section the corresponding vertices, which will be
  # gathered to make nodal connectivity
  sections_face_vtx = EP.block_to_part(face_vtx, face_distri, [np.abs(p)-1 for p in cell_face_section], comm)

  combine_funcs = [combine_to_tetra, combine_to_pyra, combine_to_penta, combine_to_hexa]

  for i, elt_kind in enumerate(['TETRA_4', 'PYRA_5', 'PENTA_6', 'HEXA_8']):
    elt = PT.get_child_from_name_and_label(zone, elt_kind, 'Elements_t')
    if elt is not None:
      _section_face_vtx = sections_face_vtx[i]
      ec = PT.get_child_from_name(elt, 'ElementConnectivity')[1]
      combine_funcs[i](_section_face_vtx.counts, _section_face_vtx.values, cell_face_section[i], ec) 

  # Renumber PointList indexing cells
  all_pl = _collected_shifted_pl(zone, 'CellCenter', -PT.Element.Range(nface_n)[0])
  # This last one is for fields supported by allCells (eg. FlowSolution)
  _pl = np.arange(cell_distri[0], cell_distri[1])
  all_pl.append(_pl)

  new_pl = EP.block_to_part(new_cell_id, cell_distri, all_pl, comm)
  
  # Update CellCentered PointList
  _update_pl(zone, 'CellCenter', new_pl[:-1])

  # For allCells containers, we need an additional exchange to reorder data in cell_distri order
  cell_distri_f = par_utils.partial_to_full_distribution(cell_distri, comm)
  GI = EP.GlobalIndexer(cell_distri_f, new_pl[-1]-quad_range[1]-1, comm)

  for path in PT.predicates_to_paths(zone, [is_cell_full_container, 'DataArray_t']):
    data = PT.get_node_from_path(zone, path)[1]
    GI.Put(data, data) # Inplace update of node data


  # Remove NGON/NFACE elements
  PT.rm_child(zone, ngon_n)
  PT.rm_child(zone, nface_n)
 

def convert_ngon_to_elements(dist_tree, comm):
  """
  Transform a polyedric (NGon based) connectivity into a standard nodal
  connectivity.
  
  Tree is modified in place : polyedric element, which are supposed to describe
  only standard elements (tris, quads, tets, pyras, prisms and hexa)
  are removed and relevant data (such as PointList) are updated.

  Args:
    dist_tree  (CGNSTree): distributed tree with polyedric connectivity
    comm       (`MPIComm`) : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #convert_ngon_to_elements@start
        :end-before: #convert_ngon_to_elements@end
        :dedent: 2
  """
  # Needed to update the joins afterward
  MJT.add_joins_donor_name(dist_tree, comm)

  for zone in PT.get_all_Zone_t(dist_tree):
    if is_poly_3d(zone):
      # Function require NFACE + NGON with PE
      if not PT.Zone.has_nface_elements(zone):
        maia.algo.pe_to_nface(zone, comm)
      ng = PT.Zone.NGonNode(zone)
      if PT.get_child_from_name(ng, 'ParentElements') is None:
        maia.algo.nface_to_pe(zone, comm)

      _ngon_to_elements_zone_3d(zone, comm)

    elif is_poly_2d(zone):
      # Function require NGON + Edge with PE
      if not PT.Zone.has_ngon_elements(zone):
        maia.algo.edge_pe_to_ngon(zone, comm)
      edge = MT.Zone.EdgeNode(zone)
      if PT.get_child_from_name(edge, 'ParentElements') is None:
        maia.algo.ngon_to_edge_pe(zone, comm)

      _ngon_to_elements_zone_2d(zone, comm)

  MJT.copy_donor_subset(dist_tree)

