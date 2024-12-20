import warnings
import numpy as np

import maia

import maia.pytree      as PT
import maia.pytree.maia as MT
from maia.pytree.sids import elements_utils as EU

from maia.utils     import np_utils, par_utils
from maia.transfer  import protocols as EP
from maia.algo.dist import matching_jns_tools as MJT


from cmaia.algo import combine_to_tetra, combine_to_pyra, \
                       combine_to_penta, combine_to_hexa

def _ngon_to_elements_zone(zone, comm):

  # Start by constructing boundary faces
  ngon_n = PT.Zone.NGonNode(zone)
  
  face_vtx_idx = PT.get_child_from_name(ngon_n, 'ElementStartOffset')[1]
  face_vtx     = PT.get_child_from_name(ngon_n, 'ElementConnectivity')[1]
  pe           = PT.get_child_from_name(ngon_n, 'ParentElements')[1]
  face_distri  = MT.getDistribution(ngon_n, 'Element')[1]
  _face_vtx_idx = (face_vtx_idx - face_vtx_idx[0]).astype(np.int64, copy=False)
  face_n = np.diff(_face_vtx_idx).astype(np.int32, copy=False)

  is_bnd_face = (pe[:,1] == 0)
  is_bnd_tri  = (is_bnd_face) & (face_n == 3)
  is_bnd_quad = (is_bnd_face) & (face_n == 4)
  tri_vtx  = np_utils.take_strided(_face_vtx_idx, face_vtx, np.where(is_bnd_tri)[0])
  quad_vtx = np_utils.take_strided(_face_vtx_idx, face_vtx, np.where(is_bnd_quad)[0])


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
  new_face_id = -1*np.ones(face_n.size, zone[1].dtype)
  new_face_id[is_bnd_tri] = np.arange(tri_distri[0]+1, tri_distri[1]+1)
  new_face_id[is_bnd_quad] = np.arange(quad_distri[0]+tri_distri[-1]+1, quad_distri[1]+tri_distri[-1]+1)

  all_pl = []
  for subset in PT.iter_all_subsets(zone, 'FaceCenter'):
    if (pl := PT.get_child_from_name(subset, 'PointList')) is not None:
      _pl = pl[1][0]
    elif (pr := PT.get_child_from_name(subset, 'PointRange')) is not None:
      distri = MT.getDistribution(subset, 'Index')[1]
      _pl = np_utils.single_dim_pr_to_pl(pr[1], distri)[0]
    all_pl.append(_pl - PT.Element.Range(ngon_n)[0])
    
  new_pl = EP.block_to_part(new_face_id, face_distri, all_pl, comm, legacy=False)
  
  for subset, _pl in zip(PT.iter_all_subsets(zone, 'FaceCenter'), new_pl):
    PT.rm_children_from_name(subset, 'PointList')
    PT.rm_children_from_name(subset, 'PointRange')
    PT.new_IndexArray(value=_pl.reshape((1,-1), order='F'), parent=subset)
    # NB : PointListDonor of GCs will be copied afterward (under usual assumption that PL are symmetric) 


  # Now take care of the cells 
  nface_n = PT.Zone.NFaceNode(zone)
  cell_face_idx = PT.get_child_from_name(nface_n, 'ElementStartOffset')[1]
  cell_face     = PT.get_child_from_name(nface_n, 'ElementConnectivity')[1]
  cell_distri   = MT.getDistribution(nface_n, 'Element')[1]
  _cell_face_idx = (cell_face_idx - cell_face_idx[0]).astype(np.int64, copy=False)
  cell_n = np.diff(cell_face_idx)

  # Design choice : get the number of vertices (with reps) for **all** cells,
  # thus we can check if elements seems to be standard. Otherwise, we could
  # do it only for cells having 5 faces to resolve prism / pyra ambiguity
  cell_nvtx_per_face = EP.block_to_part(face_n, face_distri, np.abs(cell_face)-1, comm, legacy=False)
  cell_nvtx_tot = np.add.reduceat(cell_nvtx_per_face, _cell_face_idx[:-1])

  n_treated = 0
  elt_shift = quad_range[1]
  cell_face_section = []
  mask = np.empty(cell_n.size, bool)
  new_cell_id = np.empty(cell_n.size, zone[1].dtype) # For subset renumbering
  # Gather (locally) the element per kind. In addition we prepare the renumbering table for cells
  for elt_kind, target_size in zip(['TETRA_4', 'PYRA_5', 'PENTA_6', 'HEXA_8'], [12, 16, 18, 24]):
    # Find corresponding cells
    np.equal(cell_nvtx_tot, target_size, out=mask)
    # Extract cell_face for this section
    cell_ids = np.nonzero(mask)[0]
    cell_face_section.append(np_utils.take_strided(_cell_face_idx, cell_face, cell_ids))
    
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

  assert n_treated == cell_n.size

  # Now get for each cell section the corresponding vertices, which will be
  # gathered to make nodal connectivity
  sections_stride, sections_face_vtx = EP.block_to_part_strided(face_n, face_vtx, face_distri, [np.abs(p)-1 for p in cell_face_section], comm, legacy=False)

  combine_funcs = [combine_to_tetra, combine_to_pyra, combine_to_penta, combine_to_hexa]

  for i, elt_kind in enumerate(['TETRA_4', 'PYRA_5', 'PENTA_6', 'HEXA_8']):
    elt = PT.get_child_from_name_and_label(zone, elt_kind, 'Elements_t')
    if elt is not None:
      ec = PT.get_child_from_name(elt, 'ElementConnectivity')[1]
      combine_funcs[i](sections_stride[i], sections_face_vtx[i], cell_face_section[i], ec) 

  # Renumber PointList indexing cells
  all_pl = []
  for subset in PT.iter_all_subsets(zone, 'CellCenter'):
    if (pl := PT.get_child_from_name(subset, 'PointList')) is not None:
      _pl = pl[1][0]
    elif (pr := PT.get_child_from_name(subset, 'PointRange')) is not None:
      distri = MT.getDistribution(subset, 'Index')[1]
      _pl = np_utils.single_dim_pr_to_pl(pr[1], distri)[0]
    all_pl.append(_pl - PT.Element.Range(nface_n)[0])

  # This last one is for fields supported by allCells (eg. FlowSolution)
  _pl = np.arange(cell_distri[0], cell_distri[1])
  all_pl.append(_pl)

  new_pl = EP.block_to_part(new_cell_id, cell_distri, all_pl, comm, legacy=False)
  
  # Update CellCentered PointList
  for subset, _pl in zip(PT.iter_all_subsets(zone, 'CellCenter'), new_pl[:-1]):
    PT.rm_children_from_name(subset, 'PointList')
    PT.rm_children_from_name(subset, 'PointRange')
    PT.new_IndexArray(value=_pl.reshape((1,-1), order='F'), parent=subset)

  # For allCells containers, we need an additional exchange to reorder data in cell_distri order
  is_cell_container = lambda n : PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t'] and PT.Subset.GridLocation(n) == 'CellCenter'
  cell_distri_f = par_utils.partial_to_full_distribution(cell_distri, comm)
  GI = EP.GlobalIndexer(cell_distri_f, new_pl[-1]-quad_range[1]-1, comm)

  for path in PT.predicates_to_paths(zone, [is_cell_container, 'DataArray_t']):
    data = PT.get_node_from_path(zone, path)[1]
    GI.Put_into(data, data) # Inplace update of node data


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
    dist_tree  (CGNSTree): 3D distributed tree with polyedric connectivity
    comm       (`MPIComm`) : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #convert_ngon_to_elements@start
        :end-before: #convert_ngon_to_elements@end
        :dedent: 2
  """

  # Function require NFACE + NGON with PE
  for zone in PT.get_all_Zone_t(dist_tree):
    if not PT.Zone.has_nface_elements(zone):
      maia.algo.pe_to_nface(zone, comm)
    ng = PT.Zone.NGonNode(zone)
    if PT.get_child_from_name(ng, 'ParentElements') is None:
      maia.algo.nface_to_pe(zone, comm)

  # Needed to update the joins afterward
  MJT.add_joins_donor_name(dist_tree, comm)

  for zone in PT.get_all_Zone_t(dist_tree):
    _ngon_to_elements_zone(zone, comm)

  MJT.copy_donor_subset(dist_tree)

def ngons_to_elements(dist_tree, comm):
  msg = "This function has been renamed into ``convert_ngon_to_elements``. Former name will be removed in next release."
  warnings.warn(msg, DeprecationWarning, stacklevel=2)
  return convert_ngon_to_elements(dist_tree, comm)