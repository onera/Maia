import numpy              as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import np_utils, par_utils, layouts

from maia.algo.dist   import remove_element as RME
from maia.algo.dist   import matching_jns_tools as MJT
from maia.factory.partitioning.split_U.cgns_to_pdm_dmesh_nodal import cgns_dist_zone_to_pdm_dmesh_nodal

import Pypdm.Pypdm as PDM


def pdm_dmesh_to_cgns_zone(result_dmesh, zone, comm, extract_dim):
  """
  """

  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()

  # Get PDM output
  cell = "FACE" if extract_dim == 2 else "CELL"
  face = "EDGE" if extract_dim == 2 else "FACE"
  dface_cell_idx, dface_cell = result_dmesh.dmesh_connectivity_get(eval(f"PDM._PDM_CONNECTIVITY_TYPE_{face}_{cell}"))
  dface_vtx_idx,  dface_vtx  = result_dmesh.dmesh_connectivity_get(eval(f"PDM._PDM_CONNECTIVITY_TYPE_{face}_VTX"))
  dcell_face_idx, dcell_face = result_dmesh.dmesh_connectivity_get(eval(f"PDM._PDM_CONNECTIVITY_TYPE_{cell}_{face}"))
  distrib_face               = result_dmesh.dmesh_distrib_get(eval(f"PDM._PDM_MESH_ENTITY_{face}"))
  distrib_cell               = result_dmesh.dmesh_distrib_get(eval(f"PDM._PDM_MESH_ENTITY_{cell}"))
  group_idx, pdm_group       = result_dmesh.dmesh_bound_get(eval(f"PDM._PDM_BOUND_TYPE_{face}"))

  #Shift distribution if starting at 1
  if(distrib_face[0] == 1):
    distrib_face = distrib_face-1
  if(distrib_cell[0] == 1):
    distrib_cell = distrib_cell-1

  dn_face = distrib_face[i_rank+1] - distrib_face[i_rank]
  n_face  = distrib_face[n_rank]

  n_cell  = distrib_cell[n_rank]
  dn_cell = distrib_cell[i_rank+1] - distrib_cell[i_rank]

  ldistrib_face = distrib_face[[i_rank, i_rank+1, n_rank]]
  ldistrib_cell = distrib_cell[[i_rank, i_rank+1, n_rank]]

  distrib_face_vtx  = par_utils.gather_and_shift(dface_vtx_idx[dn_face], comm, np.int32)
  distrib_cell_face = par_utils.gather_and_shift(dcell_face_idx[dn_cell], comm, np.int32)

  # Create NGon node
  ermax   = max([PT.Element.Range(e)[1] for e in PT.iter_children_from_label(zone, 'Elements_t')])

  pe = np.empty((dface_cell.shape[0]//2, 2), dtype=dface_cell.dtype, order='F')
  layouts.pdm_face_cell_to_pe_cgns(dface_cell, pe)
  #NGon PE must refer to nFace indexes, we have to shift

  np_utils.shift_nonzeros(pe, ermax+n_face)
  # > Attention overflow I8
  eso_ngon = dface_vtx_idx + distrib_face_vtx[i_rank]

  ngon_n  = PT.new_NGonElements(erange=[ermax+np.int32(1), ermax+n_face], eso=eso_ngon, ec=dface_vtx, pe=pe, parent=zone)

  MT.newDistribution({'Element' : ldistrib_face, 
                      'ElementConnectivity' : distrib_face_vtx[[i_rank, i_rank+1, n_rank]]}, 
                      ngon_n)

  # Create NFace node
  ermax   = max([PT.Element.Range(e)[1] for e in PT.iter_children_from_label(zone, 'Elements_t')])
  eso_nfac = dcell_face_idx + distrib_cell_face[i_rank]

  nfac_n = PT.new_NFaceElements(erange=[ermax+np.int32(1), ermax+n_cell], eso=eso_nfac, parent=zone)
  if dcell_face is not None:
    PT.new_DataArray('ElementConnectivity', np.abs(dcell_face), parent=nfac_n)
  else:
    PT.new_DataArray('ElementConnectivity', np.empty(0, dtype=eso_ngon.dtype), parent=nfac_n)

  MT.newDistribution({'Element' : ldistrib_cell,
                      'ElementConnectivity' : distrib_cell_face[[i_rank, i_rank+1, n_rank]]}, 
                      nfac_n)

  #Manage BCs : shift PL values to reach refer ngon_elements
  if pdm_group is not None:
    group = np.copy(pdm_group) + (PT.Zone.get_range_of_ngon(zone)[0]-1)
    for i_bc, bc in enumerate(PT.iter_children_from_predicates(zone, 'ZoneBC_t/BC_t')):
      PT.rm_nodes_from_name(bc, 'PointRange')
      PT.rm_nodes_from_name(bc, 'PointList')
      start, end = group_idx[i_bc], group_idx[i_bc+1]
      PT.new_PointList(value=group[start:end].reshape((1,-1), order='F'), parent=bc)


# -----------------------------------------------------------------
def compute_ngon_from_std_elements(dist_tree, comm):
  """
  """
  MJT.add_joins_donor_name(dist_tree, comm)
  for zgc in PT.iter_nodes_from_label(dist_tree, 'ZoneGridConnectivity_t'):
    PT.set_label(zgc, 'ZoneBC_t')
    PT.new_node('__maia::isZGC', parent=zgc)
    for gc in PT.iter_children_from_label(zgc, 'GridConnectivity_t'):
      PT.set_label(gc, 'BC_t')
  for base in PT.iter_all_CGNSBase_t(dist_tree):
    extract_dim = PT.get_value(base)[0]
    #print("extract_dim == ", extract_dim)
    zones_u = [zone for zone in PT.iter_all_Zone_t(base) if PT.Zone.Type(zone) == "Unstructured"]

    dmn_to_dm = PDM.DMeshNodalToDMesh(len(zones_u), comm)
    dmesh_nodal_list = list()
    for i_zone, zone in enumerate(zones_u):
      dmn = cgns_dist_zone_to_pdm_dmesh_nodal(zone, comm, needs_vertex=False)
      dmn.generate_distribution()
      dmn_to_dm.add_dmesh_nodal(i_zone, dmn)

    # PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_FACE
    face = "EDGE" if extract_dim == 2 else "FACE"
    dmn_to_dm.compute(eval(f"PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSFORM_TO_{face}"),
                      eval(f"PDM._PDM_DMESH_NODAL_TO_DMESH_TRANSLATE_GROUP_TO_{face}"))

    for i_zone, zone in enumerate(zones_u):
      result_dmesh = dmn_to_dm.get_dmesh(i_zone)
      pdm_dmesh_to_cgns_zone(result_dmesh, zone, comm, extract_dim)

      # > Remove internal holder state
      PT.rm_nodes_from_name(zone, ':CGNS#DMeshNodal#Bnd*')

  # > Generate correctly zone_grid_connectivity
  for zbc in PT.iter_nodes_from_label(dist_tree, 'ZoneBC_t'):
    if PT.get_child_from_name(zbc, '__maia::isZGC'):
      PT.set_label(zbc, 'ZoneGridConnectivity_t')
      PT.rm_children_from_name(zbc, '__maia::isZGC')
      for bc in PT.iter_children_from_label(zbc, 'BC_t'):
          PT.set_label(bc, 'GridConnectivity_t')
  MJT.copy_donor_subset(dist_tree)

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
  #Possible optimisation : remove all the element at the same time
  #instead of looping
  compute_ngon_from_std_elements(dist_tree,comm)
  for zone in PT.get_all_Zone_t(dist_tree):
    elts_to_remove = [elt for elt in PT.iter_children_from_label(zone, 'Elements_t') if\
        PT.Element.CGNSName(elt) not in ["NGON_n", "NFACE_n"]]
    #2D element should be removed first, to avoid probleme coming from ParentElements
    for elt in sorted(elts_to_remove, key = PT.Element.Dimension):
      RME.remove_element(zone, elt)
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

