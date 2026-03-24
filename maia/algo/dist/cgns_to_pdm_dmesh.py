import numpy          as np
from   mpi4py import MPI

import maia.pytree        as PT
import maia.pytree.maia   as MT
from   maia.pytree.maia   import pdm_elts

from maia.utils import py_utils, np_utils, layouts, as_pdm_gnum
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype

from Pypdm.Pypdm import DistributedMesh, DistributedMeshNodal
from Pypdm.Pypdm import _PDM_CONNECTIVITY_TYPE_FACE_VTX, _PDM_BOUND_TYPE_FACE, _PDM_BOUND_TYPE_EDGE, \
                        _PDM_CONNECTIVITY_TYPE_FACE_CELL, _PDM_CONNECTIVITY_TYPE_CELL_FACE, \
                        _PDM_CONNECTIVITY_TYPE_EDGE_VTX, _PDM_CONNECTIVITY_TYPE_EDGE_FACE


def cgns_dist_zone_to_pdm_dmesh_vtx(dist_zone, comm):
  """
  Create a pdm_dmesh structure for distributed having only vertices
  """
  dn_vtx      = MT.Zone.dn_vtx(dist_zone)

  if dn_vtx > 0:
    cx, cy, cz = PT.Zone.coordinates(dist_zone)
    dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])
  else:
    dvtx_coord = np.empty(0, dtype='float64', order='F')

  dmesh = DistributedMesh(comm, 0, 0, 0, dn_vtx)

  dmesh.dmesh_vtx_coord_set(dvtx_coord)

  # > Create an older --> To Suppress after all
  multi_part_node = PT.update_child(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
  PT.new_DataArray('dvtx_coord', dvtx_coord, parent=multi_part_node)

  return dmesh

def cgns_dist_zone_to_pdm_dmesh(dist_zone, comm, needs_bc=False):
  """
  Create a pdm_dmesh structure from a distributed zone
  """

  # > Try to hook NGon
  ngon_node = PT.Zone.NGonNode(dist_zone)
  ngon_first = PT.Element.Range(ngon_node)[0] == 1
  has_nface  = PT.Zone.has_nface_elements(dist_zone)
  has_pe     = PT.get_node_from_name(ngon_node, 'ParentElements') is not None
  dface_vtx = as_pdm_gnum(PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1])
  ngon_eso  = PT.get_child_from_name(ngon_node, 'ElementStartOffset' )[1]

  if has_nface:
    nface_node = PT.Zone.NFaceNode(dist_zone)
    nface_ec  = as_pdm_gnum(PT.get_child_from_name(nface_node, 'ElementConnectivity')[1])
    nface_eso = PT.get_child_from_name(nface_node, 'ElementStartOffset' )[1]
  if has_pe:
    ngon_pe = as_pdm_gnum(PT.get_child_from_name(ngon_node, 'ParentElements')[1])

  dn_vtx  = MT.Zone.dn_vtx(dist_zone)
  dn_cell = MT.Zone.dn_cell(dist_zone)
  dn_face = MT.Element.dn_elt(ngon_node)
  dn_edge = -1 #Not used

  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])

  dface_vtx_idx = np.add(ngon_eso, -ngon_eso[0], dtype=np.int32) #Local index is int32bits

  if has_nface: #Use NFace to set cell_face
    dcell_face_idx = np.add(nface_eso, -nface_eso[0], dtype=np.int32) # Local index is int32bits
    if ngon_first:
      dcell_face = nface_ec
    else:
      dcell_face = nface_ec - PT.Zone.n_cell(dist_zone)
  if has_pe: #Use PE to set face_cell
    dface_cell = np.empty(2*dn_face, dtype=pdm_gnum_dtype) # Respect pdm_gnum_type
    layouts.pe_cgns_to_pdm_face_cell(ngon_pe, dface_cell)
    if ngon_first:
      np_utils.shift_nonzeros(dface_cell, -PT.Element.Size(ngon_node))


  # > Prepare bnd (needed for HPC renumbering)
  if needs_bc:
    bcs = PT.get_children_from_predicates(dist_zone, ['ZoneBC_t', PT.pred.is_bc_of_location('FaceCenter')])
    point_lists = [MT.Subset.distributed_pointlist(bc) for bc in bcs]
    dface_bound_idx, dface_bound = np_utils.concatenate_point_list(point_lists, pdm_gnum_dtype)
  else:
    dface_bound_idx = np.zeros(1, dtype=np.int32)
    dface_bound     = np.empty(0, dtype=pdm_gnum_dtype)

  dmesh = DistributedMesh(comm, dn_cell, dn_face, dn_edge, dn_vtx)

  dmesh.dmesh_vtx_coord_set(dvtx_coord)
  dmesh.dmesh_connectivity_set(_PDM_CONNECTIVITY_TYPE_FACE_VTX, dface_vtx_idx, dface_vtx)
  dmesh.dmesh_bound_set(_PDM_BOUND_TYPE_FACE, dface_bound_idx, dface_bound)

  if has_nface:
    dmesh.dmesh_connectivity_set(_PDM_CONNECTIVITY_TYPE_CELL_FACE, dcell_face_idx, dcell_face)
  if has_pe:
    dmesh.dmesh_connectivity_set(_PDM_CONNECTIVITY_TYPE_FACE_CELL, None, dface_cell)

  # > Create an older --> To Suppress after all
  multi_part_node = PT.update_child(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
  PT.new_DataArray('dvtx_coord'     , dvtx_coord     , parent=multi_part_node)
  PT.new_DataArray('dface_vtx_idx'  , dface_vtx_idx  , parent=multi_part_node)
  PT.new_DataArray('dface_vtx'      , dface_vtx      , parent=multi_part_node)
  PT.new_DataArray('dface_bound_idx', dface_bound_idx, parent=multi_part_node)
  PT.new_DataArray('dface_bound'    , dface_bound    , parent=multi_part_node)
  if has_nface:
    PT.new_DataArray('dcell_face_idx', dcell_face_idx, parent=multi_part_node)
    PT.new_DataArray('dcell_face'    , dcell_face    , parent=multi_part_node)
  if has_pe:
    PT.new_DataArray('dface_cell'    , dface_cell    , parent=multi_part_node)

  return dmesh

def cgns_dist_zone_to_pdm_dmesh_2d(dist_zone, comm, needs_bc=False):
  """
  Create a pdm_dmesh structure from a 2d distributed zone
  """
  distrib_vtx  = MT.Zone.vtx_distribution(dist_zone)
  distrib_face = MT.Zone.cell_distribution(dist_zone) #In 2d, cell == face

  # Try to hook Edge nodes
  edge_node  = MT.Zone.EdgeNode(dist_zone)
  edge_first = PT.Element.Range(edge_node)[0] == 1
  has_pe = PT.get_child_from_name(edge_node, 'ParentElements') is not None

  dedge_vtx = as_pdm_gnum(PT.get_child_from_name(edge_node, 'ElementConnectivity')[1])
  if not has_pe:
    from maia.algo.dist import ngon_tools
    ngon_tools.ngon_to_edge_pe(dist_zone, comm)
  edge_pe = as_pdm_gnum(PT.get_child_from_name(edge_node, 'ParentElements')[1])

  distrib_edge = MT.Element.distribution(edge_node)

  dn_vtx  = distrib_vtx[1] - distrib_vtx[0]
  dn_face = distrib_face[1] - distrib_face[0]
  dn_edge = distrib_edge[1] - distrib_edge[0]


  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  if cz is None:
    cz = np.zeros_like(cx)
  dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])


  dedge_face = np.empty(2*dn_edge, dtype=pdm_gnum_dtype) # Respect pdm_gnum_type
  layouts.pe_cgns_to_pdm_face_cell(edge_pe, dedge_face)
  if edge_first:
    np_utils.shift_nonzeros(dedge_face, -distrib_edge[2])

  # Fix in PDM for dmesh_extract not yet integrated
  dedge_vtx_idx = 2*np.arange(dedge_vtx.shape[0]//2+1, dtype=np.int32)

  #Create DMesh
  dmesh = DistributedMesh(comm, 0, dn_face, dn_edge, dn_vtx)

  dmesh.dmesh_vtx_coord_set(dvtx_coord)
  dmesh.dmesh_connectivity_set(_PDM_CONNECTIVITY_TYPE_EDGE_VTX,  dedge_vtx_idx, dedge_vtx)
  dmesh.dmesh_connectivity_set(_PDM_CONNECTIVITY_TYPE_EDGE_FACE, None, dedge_face)

  # > Prepare bnd (needed for HPC renumbering)
  if needs_bc:
    bcs = PT.get_children_from_predicates(dist_zone, ['ZoneBC_t', PT.pred.is_bc_of_location('EdgeCenter')])
    point_lists = [MT.Subset.distributed_pointlist(bc) for bc in bcs]
    dedge_bound_idx, dedge_bound = np_utils.concatenate_point_list(point_lists, pdm_gnum_dtype)
  else:
    dedge_bound_idx = np.zeros(1, dtype=np.int32)
    dedge_bound     = np.empty(0, dtype=pdm_gnum_dtype)
  dmesh.dmesh_bound_set(_PDM_BOUND_TYPE_EDGE, dedge_bound_idx, dedge_bound)

  # keep dvtx_coord object alive for ParaDiGM
  multi_part_node = PT.update_child(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
  PT.new_DataArray('dvtx_coord'     , dvtx_coord     , parent=multi_part_node)
  PT.new_DataArray('dedge_vtx'      , dedge_vtx      , parent=multi_part_node)
  PT.new_DataArray('dedge_face'     , dedge_face     , parent=multi_part_node)
  PT.new_DataArray('dedge_vtx_idx'  , dedge_vtx_idx  , parent=multi_part_node)
  PT.new_DataArray('dedge_bound_idx', dedge_bound_idx, parent=multi_part_node)
  PT.new_DataArray('dedge_bound'    , dedge_bound    , parent=multi_part_node)

  return dmesh


def cgns_dist_zone_to_pdm_dmesh_poly2d(dist_zone, comm):
  """
  This function was used to split 2D meshes having only a
  face_vtx (NGON) connectivity (without edge => without bc)
  It is unused now, but we save it in case of need
  """
  distrib_vtx = MT.Zone.vtx_distribution(dist_zone)
  distrib_face = MT.Zone.cell_distribution(dist_zone) #In 2d, cell == face
  n_vtx   = distrib_vtx[2]
  n_face  = distrib_face[2]
  dn_vtx  = distrib_vtx[1] - distrib_vtx[0]
  dn_face = distrib_face[1] - distrib_face[0]

  #Create DMeshNodal
  dmesh_nodal = DistributedMeshNodal(comm, n_vtx, 0, n_face, 0, mesh_dimension=2)

  if dn_vtx > 0:
    cx, cy, cz = PT.Zone.coordinates(dist_zone)
    dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])
  else:
    dvtx_coord = np.empty(0, dtype='float64', order='F')
  dmesh_nodal.set_coordinates(dvtx_coord)

  ngon_node = PT.Zone.NGonNode(dist_zone)
  face_vtx  = MT.Element.connectivity(ngon_node)

  dface_vtx_idx = face_vtx.displs.astype(np.int32, copy=False)
  dface_vtx     = as_pdm_gnum(face_vtx.values)
  dmesh_nodal.set_poly2d_section(dface_vtx_idx, dface_vtx)

  # keep dvtx_coord object alive for ParaDiGM
  multi_part_node = PT.update_child(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
  PT.new_DataArray('dvtx_coord', dvtx_coord, parent=multi_part_node)
  PT.new_DataArray('dface_vtx_idx', dface_vtx_idx, parent=multi_part_node)
  PT.new_DataArray('dface_vtx', dface_vtx, parent=multi_part_node)

  return dmesh_nodal

def cgns_dist_zone_to_pdm_dmesh_nodal(dist_zone, comm, needs_vertex=True, needs_bc=True):
  """
  Create a pdm_dmesh_nodal structure from a distributed zone
  """
  distrib_vtx = MT.Zone.vtx_distribution(dist_zone)
  n_vtx   = distrib_vtx[2]
  dn_vtx  = distrib_vtx[1] - distrib_vtx[0]

  n_elt_per_dim  = [0,0,0]
  sorted_elts_by_dim = PT.Zone.get_ordered_elements_per_dim(dist_zone)
  for elt_dim in sorted_elts_by_dim:
    for elt in elt_dim:
      assert PT.Element.Type(elt) not in ["NGON_n", "NFACE_n"]
      if PT.Element.Dimension(elt) > 0:
        n_elt_per_dim[PT.Element.Dimension(elt)-1] += PT.Element.Size(elt)

  if PT.Zone.elt_ordering_by_dim(dist_zone) == 0:
    raise ValueError(f"Sections of unstructured zone '{PT.get_name(dist_zone)}' are not ordered by dimension," \
                     f" which is not supported. Please reorder the sections using maia.algo.dist.reorder_elt_sections_from_dim.")

  #Create DMeshNodal
  mesh_dimension = 3
  for n_elt_dim in n_elt_per_dim[::-1]:
    if n_elt_dim != 0:
      break
    mesh_dimension -= 1
  dmesh_nodal = DistributedMeshNodal(comm, n_vtx, *n_elt_per_dim[::-1], mesh_dimension)

  #Vertices
  if needs_vertex:
    if dn_vtx > 0:
      cx, cy, cz = PT.Zone.coordinates(dist_zone)
      if cz is None:
        cz = np.zeros_like(cx)
      dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])
    else:
      dvtx_coord = np.empty(0, dtype='float64', order='F')
    dmesh_nodal.set_coordinates(dvtx_coord)

    # keep dvtx_coord object alive for ParaDiGM
    multi_part_node = PT.update_child(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
    PT.new_DataArray('dvtx_coord', dvtx_coord, parent=multi_part_node)

  #Elements
  for i_dim, elts in enumerate(sorted_elts_by_dim):
    elt_pdm_types = np.array([pdm_elts.cgns_elt_name_to_pdm_element_type(PT.Element.Type(e)) for e in elts], dtype=np.int32)
    elt_lengths   = np.array([MT.Element.dn_elt(e) for e in elts], dtype=np.int32)
    elmts_connectivities = [as_pdm_gnum(PT.get_child_from_name(e, "ElementConnectivity")[1]) for e in elts]
    dmesh_nodal.set_sections(pdm_elts.elements_dim_to_pdm_kind[i_dim], elmts_connectivities, elt_pdm_types, elt_lengths)

  # Boundaries
  if needs_bc:
    range_by_dim = PT.Zone.get_elt_range_per_dim(dist_zone)

    cell_dim = PT.Zone.CellDimension(dist_zone)
    if cell_dim == 1:
      elts_loc = ['Vertex', 'CellCenter']
    elif cell_dim == 2:
      elts_loc = ['Vertex', 'EdgeCenter', 'CellCenter']
    else:
      elts_loc = ['Vertex', 'EdgeCenter', 'FaceCenter', 'CellCenter']

    # Skip Vertex-located BCs, because they are not in Element numbering
    bc_nodes = PT.get_children_from_predicates(dist_zone, ['ZoneBC_t', PT.pred.label_is('BC_t') & ~PT.pred.has_location('Vertex')])
    bc_nodes_by_dim = py_utils.bucket_split(bc_nodes, lambda bc: elts_loc.index(PT.Subset.GridLocation(bc)), size=4)

    bc_point_lists_by_dim = [[MT.Subset.distributed_pointlist(node) for node in nodes] for nodes in bc_nodes_by_dim]

    for i_dim, bc_pl in enumerate(bc_point_lists_by_dim):
      if(len(bc_pl) > 0 ):
        delmt_bound_idx, delmt_bound = np_utils.concatenate_point_list(bc_pl, pdm_gnum_dtype)
        # Shift because CGNS global numbering is for all elements / ParaDiGM is by dimension
        delmt_bound -= (range_by_dim[i_dim][0] - 1)
        n_elmt_group = delmt_bound_idx.shape[0] - 1
        #Need an holder to prevent memory deletion
        pdm_node = PT.update_child(dist_zone, ':CGNS#DMeshNodal#Bnd{0}'.format(i_dim), 'UserDefinedData_t')
        PT.new_DataArray('delmt_bound_idx', delmt_bound_idx, parent=pdm_node)
        PT.new_DataArray('delmt_bound'    , delmt_bound    , parent=pdm_node)

        dmesh_nodal.set_group_elmt(pdm_elts.elements_dim_to_pdm_kind[i_dim], n_elmt_group, delmt_bound_idx, delmt_bound)

  return dmesh_nodal

