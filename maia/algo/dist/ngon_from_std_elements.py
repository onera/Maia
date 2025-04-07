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
    skip_location = ['FaceCenter']
  elif extract_dim == 3:
    group_idx, pdm_group = result_dmesh.dmesh_bound_get(PDM._PDM_BOUND_TYPE_FACE)
    keep_location = 'FaceCenter'
    skip_location = ['EdgeCenter']
  converted_bc   = lambda n : PT.get_label(n) == 'BC_t' and PT.Subset.GridLocation(n) == keep_location
  unconverted_bc = lambda n : PT.get_label(n) == 'BC_t' and PT.Subset.GridLocation(n) in skip_location
  if pdm_group is not None:
    group = np_utils.safe_int_cast(np.copy(pdm_group), zone[1].dtype)
    for i_bc, bc in enumerate(PT.iter_children_from_predicates(zone, ['ZoneBC_t', converted_bc])):
      PT.rm_children_from_name(bc, 'PointRange')
      PT.rm_children_from_name(bc, 'PointList')
      start, end = group_idx[i_bc], group_idx[i_bc+1]
      PT.new_IndexArray(value=group[start:end].reshape((1,-1), order='F'), parent=bc)

  # Remove unconverted BCs
  for zbc in PT.get_nodes_from_label(zone, 'ZoneBC_t'):
    PT.rm_nodes_from_predicate(zbc, unconverted_bc)

  # Remove std elements
  first_cell_id = PT.Zone.get_elt_range_per_dim(zone)[extract_dim][0] - 1 # Needed for CellCenter PL shift
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
    ngon_er  = np.array([1, n_face], dtype=zone[1].dtype)
    ngon_pe  = _create_pe_global(np_utils.safe_int_cast(dface_cell, ngon_er.dtype), n_face)
    ngon_ec  = np_utils.safe_int_cast(dface_vtx, ngon_er.dtype)
    ngon_eso = np_utils.safe_int_cast(dface_vtx_idx, ngon_er.dtype)
    ngon_eso += distrib_face_vtx[i_rank]

    ngon_n  = PT.new_NGonElements(erange=ngon_er, eso=ngon_eso, ec=ngon_ec, pe=ngon_pe, parent=zone)
    MT.newDistribution({'Element' :             par_utils.full_to_partial_distribution(distrib_face, comm),
                        'ElementConnectivity' : par_utils.full_to_partial_distribution(distrib_face_vtx, comm)},
                        ngon_n)

    # Create NFACE
    nface_er  = np.array([1, n_cell], dtype=zone[1].dtype) + n_face
    nface_ec  = np_utils.safe_int_cast(dcell_face, nface_er.dtype)
    nface_eso = np_utils.safe_int_cast(dcell_face_idx, nface_er.dtype)
    nface_eso += distrib_cell_face[i_rank]

    nfac_n = PT.new_NFaceElements(erange=nface_er, eso=nface_eso, ec=nface_ec, parent=zone)
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
    
    edge_er = np.array([1, n_edge], dtype=zone[1].dtype)
    edge_ec = np_utils.safe_int_cast(dedge_vtx, edge_er.dtype)
    edge_pe = _create_pe_global(np_utils.safe_int_cast(dedge_face, edge_er.dtype), n_edge)

    bar_n = PT.new_Elements('EdgeElements', 'BAR_2', erange=edge_er, econn=edge_ec, parent=zone)
    PT.new_DataArray('ParentElements', edge_pe, parent=bar_n)
    MT.newDistribution({'Element' : par_utils.full_to_partial_distribution(distrib_edge, comm)},
                         bar_n)

    # Create NGON (combine face_edge + edge_vtx)
    ngon_er = np.array([1, n_face], dtype=zone[1].dtype) + n_edge
    ngon_ec = PDM.compute_dfacevtx_from_face_and_edge(comm, distrib_face, distrib_edge, dface_edge_idx, dface_edge, dedge_vtx)
    ngon_ec  = np_utils.safe_int_cast(ngon_ec, ngon_er.dtype)
    ngon_eso = np_utils.safe_int_cast(dface_edge_idx, ngon_er.dtype)
    ngon_eso += distrib_face_vtx[i_rank]

    ngon_n  = PT.new_NGonElements(erange=ngon_er, eso=ngon_eso, ec=ngon_ec, parent=zone)
    MT.newDistribution({'Element' :             par_utils.full_to_partial_distribution(distrib_face, comm),
                        'ElementConnectivity' : par_utils.full_to_partial_distribution(distrib_face_vtx, comm)},
                        ngon_n)


  # > Shift CellCenter located pointlist
  shift = n_face if extract_dim == 3 else n_edge
  for node in PT.iter_all_subsets(zone, ['CellCenter']):
    pl = PT.get_child_from_predicate(node, lambda n : PT.get_name(n) in ['PointList', 'PointRange'])
    pl[1] += shift - first_cell_id # Last one shift back to 0 if mesh was increasing dim. numbered

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

  is_container = lambda n : PT.get_label(n) in ['FlowSolution_t', 'ZoneSubRegion_t', 'DiscreteData_t']
  is_fcenter   = lambda n : PT.Subset.GridLocation(n) not in ['CellCenter', 'Vertex']
  is_subset    = lambda n : PT.get_child_from_name(n, 'PointList') is not None \
                         or PT.get_child_from_name(n, 'PointRange') is not None
  
  # Convert data having PL into bc, so they will be converted by the function
  for dist_zone in PT.iter_all_Zone_t(dist_tree):
    # BCDS case is specific (they are included in BCs)
    for zbc in PT.iter_children_from_label(dist_zone, 'ZoneBC_t'):
      for bc in PT.get_children_from_label(zbc, 'BC_t'):
        for bcds in PT.get_children_from_predicate(bc, lambda n : PT.get_label(n) == 'BCDataSet_t' and is_subset(n)):
          bcds[0] = f'__maia::isBCDS#@#{bc[0]}#@#{bcds[0]}'
          bcds[3] = 'BC_t'
          PT.add_child(zbc, bcds)
        PT.rm_children_from_name(bc, '__maia::isBCDS#@#*')
    # GC case is specific (they have their own container)
    for zgc in PT.iter_children_from_label(dist_zone, 'ZoneGridConnectivity_t'):
      PT.set_label(zgc, 'ZoneBC_t')
      PT.new_node('__maia::isZGC', parent=zgc)
      for gc in PT.iter_children_from_label(zgc, 'GridConnectivity_t'):
        PT.set_label(gc, 'BC_t')
    # Other data (as ZSR) are self contained
    to_remove = list()
    container = PT.new_child(dist_zone, '__maia::isSubset', 'ZoneBC_t')
    for node in PT.get_children_from_predicate(dist_zone, lambda n: is_container(n) and is_fcenter(n) and is_subset(n)):
      PT.new_Descriptor('__maia::initialLabel', PT.get_label(node), parent=node)
      PT.set_label(node, 'BC_t')
      to_remove.append(PT.get_name(node))
      PT.add_child(container, node)
    PT.rm_children_from_predicate(dist_zone, lambda n : PT.get_name(n) in to_remove)

  is_zone     = lambda n : PT.get_label(n) == 'Zone_t'
  is_zone_elt = lambda n : is_zone(n) and PT.Zone.Type(n) == 'Unstructured' and not PT.Zone.has_ngon_elements(n)
  for base in PT.iter_all_CGNSBase_t(dist_tree):
    extract_dim = PT.get_value(base)[0]
    zones_u = PT.get_children_from_predicate(base, is_zone_elt)

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

  # Convert back "fake bc" containers
  for dist_zone in PT.iter_all_Zone_t(dist_tree):
    for zbc in PT.get_children_from_label(dist_zone, 'ZoneBC_t'):
      # > Zone_grid_connectivity
      if PT.get_child_from_name(zbc, '__maia::isZGC'):
        PT.set_label(zbc, 'ZoneGridConnectivity_t')
        PT.rm_children_from_name(zbc, '__maia::isZGC')
        for bc in PT.iter_children_from_label(zbc, 'BC_t'):
            PT.set_label(bc, 'GridConnectivity_t')
      # > Original BCs
      elif PT.get_name(zbc) != '__maia::isSubset':
        for bcds in PT.get_nodes_from_name(zbc, '__maia::isBCDS*'):
          _, bc_name, ds_name = bcds[0].split('#@#')
          bc = PT.get_child_from_name(zbc, bc_name)
          bcds[0] = ds_name
          bcds[3] = 'BCDataSet_t'
          if bc is not None: # BC may have been removed (eg. EdgeCenter BCs)
            PT.add_child(bc, bcds)
        PT.rm_children_from_label(zbc, 'BCDataSet_t')
    # > Subsets
    container = PT.get_child_from_name(dist_zone, '__maia::isSubset')
    for node in PT.get_children(container):
      old_label = PT.get_child_from_name(node, '__maia::initialLabel')
      PT.set_label(node, PT.get_value(old_label))
      PT.rm_child(node, old_label)
      PT.add_child(dist_zone, node)
    PT.rm_child(dist_zone, container)

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
  decreasing). Tree made of *Mixed* elements are supported
  (:func:`convert_mixed_to_elements` is called under the hood).

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
  # If tree has MIXED elements, first convert Mixed -> Elts
  is_mixed = lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'MIXED'
  has_mixed = PT.get_node_from_predicates(dist_tree, ['CGNSBase_t', 'Zone_t', is_mixed]) is not None
  if has_mixed:
    maia.algo.dist.convert_mixed_to_elements(dist_tree, comm)

  if stable_sort: 
    from .elements_to_ngons import elements_to_ngons
    elements_to_ngons(dist_tree, comm)
  else:
    generate_ngon_from_std_elements(dist_tree, comm)

  lib_version = PT.get_child_from_name(dist_tree, 'CGNSLibraryVersion')
  if PT.get_value(lib_version)[0] < 4:
    PT.set_value(lib_version, 4.2)
