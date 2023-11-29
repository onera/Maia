import numpy              as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia.utils import np_utils, par_utils, layouts

from maia.algo.dist   import remove_element as RME
from maia.algo.dist   import matching_jns_tools as MJT
from maia.factory.partitioning.split_U.cgns_to_pdm_dmesh import cgns_dist_zone_to_pdm_dmesh_nodal

import Pypdm.Pypdm as PDM

def raise_if_possible_overflow(n_elt, n_rank):
  max_int = 2**31 - 1
  if n_elt > n_rank * max_int:
    req = n_elt // max_int + 1
    msg = f"Size of data seems to be too large regarding the number of MPI ranks. "\
          f"Please try with at least {req} processes."
    raise OverflowError(msg)

def _create_pe_global(flat_array, shift_value):
  pe = np.empty((flat_array.size//2, 2), dtype=flat_array.dtype, order='F')
  layouts.pdm_face_cell_to_pe_cgns(flat_array, pe)
  np_utils.shift_nonzeros(pe, shift_value)
  return pe

def predict_face_vtx_size(zone, dim):
  n_vtx_mult = {'BAR_2' : 2, 'TRI_3': 3, 'QUAD_4': 4, 
                'TETRA_4': 12, 'PYRA_5': 16, 'PENTA_6': 18, 'HEXA_8': 24}
  face_vtx_size = 0
  elt_predicate = lambda n : PT.get_label(n) == 'Elements_t' and PT.Element.Dimension(n) >= dim - 1 
  for elt in PT.iter_children_from_predicate(zone, elt_predicate):
    try:
      face_vtx_size += PT.Element.Size(elt) * n_vtx_mult[PT.Element.CGNSName(elt)]
    except KeyError:
      pass
  return face_vtx_size

def cgns_zone_to_pdm_dmesh_nodal(zone, comm, extract_dim):
  elts_per_dim = PT.Zone.get_ordered_elements_per_dim(zone)
  assert len(elts_per_dim[0]) == 0, "NODE elements are not supported in STD->NGON conversion"
  dmn = cgns_dist_zone_to_pdm_dmesh_nodal(zone, comm, needs_vertex=False)
  dmn.generate_distribution()
  return dmn
  
def pdm_dmesh_to_cgns_zone(result_dmesh, zone, comm, extract_dim):
  """
  """
  i_rank = comm.Get_rank()

  #Manage BCs : shift PL values to reach refer bnd elements
  if extract_dim == 2:
    group_idx, pdm_group = result_dmesh.dmesh_bound_get(PDM._PDM_BOUND_TYPE_EDGE)
    keep_location = 'EdgeCenter'
    skip_location = ['FaceCenter', 'CellCenter']
  elif extract_dim == 3:
    group_idx, pdm_group = result_dmesh.dmesh_bound_get(PDM._PDM_BOUND_TYPE_FACE)
    keep_location = 'FaceCenter'
    skip_location = ['EdgeCenter', 'CellCenter']
  converted_bc   = lambda n : PT.get_label(n) == 'BC_t' and PT.Subset.GridLocation(n) == keep_location
  unconverted_bc = lambda n : PT.get_label(n) == 'BC_t' and PT.Subset.GridLocation(n) in skip_location
  if pdm_group is not None:
    group = np.copy(pdm_group)
    for i_bc, bc in enumerate(PT.iter_children_from_predicates(zone, ['ZoneBC_t', converted_bc])):
      PT.rm_nodes_from_name(bc, 'PointRange')
      PT.rm_nodes_from_name(bc, 'PointList')
      start, end = group_idx[i_bc], group_idx[i_bc+1]
      PT.new_PointList(value=group[start:end].reshape((1,-1), order='F'), parent=bc)

  # Remove unconverted BCs
  for zbc in PT.get_nodes_from_label(zone, 'ZoneBC_t'):
    PT.rm_nodes_from_predicate(zbc, unconverted_bc)

  # Remove std elements
  PT.rm_children_from_label(zone, 'Elements_t')

  # Create polyedric elements
  if extract_dim == 3:
    dface_cell_idx, dface_cell = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_FACE_CELL)
    dface_vtx_idx,  dface_vtx  = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    dcell_face_idx, dcell_face = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
    distrib_face               = result_dmesh.dmesh_distrib_get(PDM._PDM_MESH_ENTITY_FACE)
    distrib_cell               = result_dmesh.dmesh_distrib_get(PDM._PDM_MESH_ENTITY_CELL)

    n_face  = distrib_face[-1]
    n_cell  = distrib_cell[-1]

    distrib_face_vtx  = par_utils.gather_and_shift(dface_vtx_idx[-1], comm, distrib_face.dtype)
    distrib_cell_face = par_utils.gather_and_shift(dcell_face_idx[-1], comm, distrib_cell.dtype)

    # Create NGON
    ngon_er  = np.array([1, n_face], dtype=dface_vtx.dtype)
    ngon_pe  = _create_pe_global(dface_cell, n_face)
    ngon_eso = dface_vtx_idx + distrib_face_vtx[i_rank]
    ngon_eso = np_utils.safe_int_cast(ngon_eso, ngon_er.dtype)

    ngon_n  = PT.new_NGonElements(erange=ngon_er, eso=ngon_eso, ec=dface_vtx, pe=ngon_pe, parent=zone)
    MT.newDistribution({'Element' :             par_utils.full_to_partial_distribution(distrib_face, comm),
                        'ElementConnectivity' : par_utils.full_to_partial_distribution(distrib_face_vtx, comm)},
                        ngon_n)

    # Create NFACE
    nface_er  = np.array([1, n_cell], dtype=dcell_face.dtype) + n_face
    nface_eso = dcell_face_idx + distrib_cell_face[i_rank]
    nface_eso = np_utils.safe_int_cast(nface_eso, nface_er.dtype)

    nfac_n = PT.new_NFaceElements(erange=nface_er, eso=nface_eso, ec=dcell_face, parent=zone)
    MT.newDistribution({'Element' :             par_utils.full_to_partial_distribution(distrib_cell, comm),
                        'ElementConnectivity' : par_utils.full_to_partial_distribution(distrib_cell_face, comm)},
                         nfac_n)

  elif extract_dim == 2:
    dedge_face_idx, dedge_face = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_EDGE_FACE)
    dedge_vtx_idx,  dedge_vtx  = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX)
    dface_edge_idx, dface_edge = result_dmesh.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_FACE_EDGE)
    distrib_edge               = result_dmesh.dmesh_distrib_get(PDM._PDM_MESH_ENTITY_EDGE)
    distrib_face               = result_dmesh.dmesh_distrib_get(PDM._PDM_MESH_ENTITY_FACE)

    n_edge  = distrib_edge[-1]
    n_face  = distrib_face[-1]

    distrib_face_vtx  = par_utils.gather_and_shift(dface_edge_idx[-1], comm, distrib_face.dtype) # Same as distri_face_edge
    
    edge_er  = np.array([1, n_edge], dtype=dedge_vtx.dtype)
    edge_pe  = _create_pe_global(dedge_face, n_edge)

    bar_n = PT.new_Elements('EdgeElements', 'BAR_2', erange=edge_er, econn=dedge_vtx, parent=zone)
    PT.new_DataArray('ParentElements', edge_pe, parent=bar_n)
    MT.newDistribution({'Element' : par_utils.full_to_partial_distribution(distrib_edge, comm)},
                         bar_n)

    # Create NGON (combine face_edge + edge_vtx)
    ngon_er = np.array([1, n_face], dtype=dface_edge.dtype) + n_edge
    ngon_ec = PDM.compute_dfacevtx_from_face_and_edge(comm, distrib_face, distrib_edge, dface_edge_idx, dface_edge, dedge_vtx)
    ngon_eso = dface_edge_idx + distrib_face_vtx[i_rank]
    ngon_eso = np_utils.safe_int_cast(ngon_eso, ngon_er.dtype)

    ngon_n  = PT.new_NGonElements(erange=ngon_er, eso=ngon_eso, ec=ngon_ec, parent=zone)
    MT.newDistribution({'Element' :             par_utils.full_to_partial_distribution(distrib_face, comm),
                        'ElementConnectivity' : par_utils.full_to_partial_distribution(distrib_face_vtx, comm)},
                        ngon_n)


  # > Remove internal holder state
  PT.rm_nodes_from_name(zone, ':CGNS#DMeshNodal#Bnd*')


def generate_ngon_from_std_elements(dist_tree, comm):
  """
  Transform an element based connectivity into a polyedric (NGon based)
  connectivity.
  
  Tree is modified in place : standard element are removed from the zones
  and Pointlist (under the BC_t nodes) are updated.

  Requirement : the ``Element_t`` nodes appearing in the distributed zones
  must be ordered according to their dimension (either increasing or 
  decreasing). 

  This function also works on 2d meshes.

  Args:
    dist_tree  (CGNSTree): Tree with connectivity described by standard elements
    comm       (`MPIComm`) : MPI communicator
  """
  MJT.add_joins_donor_name(dist_tree, comm)
  # Convert gcs into bc, so they will be converted by the function
  for zgc in PT.iter_nodes_from_label(dist_tree, 'ZoneGridConnectivity_t'):
    PT.set_label(zgc, 'ZoneBC_t')
    PT.new_node('__maia::isZGC', parent=zgc)
    for gc in PT.iter_children_from_label(zgc, 'GridConnectivity_t'):
      PT.set_label(gc, 'BC_t')

  for base in PT.iter_all_CGNSBase_t(dist_tree):
    extract_dim = PT.get_value(base)[0]
    zones_u = [zone for zone in PT.iter_all_Zone_t(base) if PT.Zone.Type(zone) == "Unstructured"]

    for zone in zones_u: #Raise if overflow is probable
      face_vtx_size = predict_face_vtx_size(zone, extract_dim)
      raise_if_possible_overflow(face_vtx_size*np.dtype(maia.npy_pdm_gnum_dtype).itemsize, comm.Get_size())

    dmn_to_dm = PDM.DMeshNodalToDMesh(len(zones_u), comm)
    for i_zone, zone in enumerate(zones_u):
      dmn = cgns_zone_to_pdm_dmesh_nodal(zone, comm, extract_dim)
      dmn_to_dm.add_dmesh_nodal(i_zone, dmn)

    # PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_FACE
    face = "EDGE" if extract_dim == 2 else "FACE"
    dmn_to_dm.compute(eval(f"PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_{face}"),
                      eval(f"PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSLATE_GROUP_TO_{face}"))

    for i_zone, zone in enumerate(zones_u):
      result_dmesh = dmn_to_dm.get_dmesh(i_zone)
      pdm_dmesh_to_cgns_zone(result_dmesh, zone, comm, extract_dim)

  # > Generate correctly zone_grid_connectivity
  for zbc in PT.iter_nodes_from_label(dist_tree, 'ZoneBC_t'):
    if PT.get_child_from_name(zbc, '__maia::isZGC'):
      PT.set_label(zbc, 'ZoneGridConnectivity_t')
      PT.rm_children_from_name(zbc, '__maia::isZGC')
      for bc in PT.iter_children_from_label(zbc, 'BC_t'):
          PT.set_label(bc, 'GridConnectivity_t')
  MJT.copy_donor_subset(dist_tree)

def convert_elements_to_ngon(dist_tree, comm, stable_sort=False):
  """
  Transform an element based connectivity into a polyedric (NGon based)
  connectivity.
  
  Tree is modified in place : standard element are removed from the zones
  and the PointList are updated. If ``stable_sort`` is True, face based PointList
  keep their original values.

  Requirement : the ``Element_t`` nodes appearing in the distributed zones
  must be ordered according to their dimension (either increasing or 
  decreasing). 

  Args:
    dist_tree  (CGNSTree): Tree with connectivity described by standard elements
    comm       (`MPIComm`) : MPI communicator
    stable_sort (bool, optional) : If True, 2D elements described in the
      elements section keep their original id. Defaults to False.

  Note that ``stable_sort`` is an experimental feature that brings the additional
  constraints:
    
    - 2D meshes are not supported;
    - 2D sections must have lower ElementRange than 3D sections.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #convert_elements_to_ngon@start
        :end-before: #convert_elements_to_ngon@end
        :dedent: 2
  """
  if stable_sort: 
    from .elements_to_ngons import elements_to_ngons
    elements_to_ngons(dist_tree, comm)
  else:
    generate_ngon_from_std_elements(dist_tree, comm)

