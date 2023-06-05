import numpy          as np
from   mpi4py import MPI

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import py_utils, np_utils, layouts, as_pdm_gnum
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.transfer.dist_to_part.index_exchange import collect_distributed_pl

from Pypdm.Pypdm import DistributedMesh, DistributedMeshNodal
from Pypdm.Pypdm import _PDM_CONNECTIVITY_TYPE_FACE_VTX, _PDM_BOUND_TYPE_FACE, \
                        _PDM_CONNECTIVITY_TYPE_FACE_CELL, _PDM_CONNECTIVITY_TYPE_CELL_FACE

def _split_point_list_by_dim(pl_list, range_by_dim, comm):
  """
  Split a list of PointList nodes into 4 sublists depending on the dimension
  of each PointList.
  Dimension is recovered using the values of the PointList and the range_by_dim
  array (GridLocation may be a better choice ?)
  """
  def _get_dim(pl):
    min_l_pl = np.amin(pl[0,:], initial=np.iinfo(pl.dtype).max)
    max_l_pl = np.amax(pl[0,:], initial=-1)

    min_pl    = comm.allreduce(min_l_pl, op=MPI.MIN)
    max_pl    = comm.allreduce(max_l_pl, op=MPI.MAX)
    for i_dim in range(len(range_by_dim)):
      if(min_pl >= range_by_dim[i_dim][0] and max_pl <= range_by_dim[i_dim][1]):
        return i_dim

  return py_utils.bucket_split(pl_list, lambda pl: _get_dim(pl), size=4)

def cgns_dist_zone_to_pdm_dmesh_vtx(dist_zone, comm):
  """
  Create a pdm_dmesh structure for distributed having only vertices
  """
  distrib_vtx = PT.get_value(MT.getDistribution(dist_zone, 'Vertex'))
  dn_vtx      = distrib_vtx[1] - distrib_vtx[0]

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

def cgns_dist_zone_to_pdm_dmesh(dist_zone, comm):
  """
  Create a pdm_dmesh structure from a distributed zone
  """
  distrib_vtx      = PT.get_value(MT.getDistribution(dist_zone, 'Vertex'))
  distrib_cell     = PT.get_value(MT.getDistribution(dist_zone, 'Cell'))

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
    distrib_cell_face = as_pdm_gnum(PT.get_value(MT.getDistribution(nface_node, 'ElementConnectivity')))
  if has_pe:
    ngon_pe = as_pdm_gnum(PT.get_child_from_name(ngon_node, 'ParentElements')[1])

  distrib_face     = as_pdm_gnum(PT.get_value(MT.getDistribution(ngon_node, 'Element')))
  distrib_face_vtx = as_pdm_gnum(PT.get_value(MT.getDistribution(ngon_node, 'ElementConnectivity')))

  dn_vtx  = distrib_vtx [1] - distrib_vtx [0]
  dn_cell = distrib_cell[1] - distrib_cell[0]
  dn_face = distrib_face[1] - distrib_face[0]
  dn_edge = -1 #Not used

  cx, cy, cz = PT.Zone.coordinates(dist_zone)
  dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])

  dface_vtx_idx = np.add(ngon_eso, -distrib_face_vtx[0], dtype=np.int32) #Local index is int32bits

  if has_nface: #Use NFace to set cell_face
    dcell_face_idx = np.add(nface_eso, -distrib_cell_face[0], dtype=np.int32) # Local index is int32bits
    if ngon_first:
      dcell_face = nface_ec
    else:
      dcell_face = nface_ec - distrib_cell[2]
  if has_pe: #Use PE to set face_cell
    dface_cell = np.empty(2*dn_face, dtype=pdm_gnum_dtype) # Respect pdm_gnum_type
    layouts.pe_cgns_to_pdm_face_cell(ngon_pe, dface_cell)
    if ngon_first:
      np_utils.shift_nonzeros(dface_cell, -distrib_face[2])
    

  # > Prepare bnd
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

def cgns_dist_zone_to_pdm_dmesh_poly2d(dist_zone, comm):
  distrib_vtx = PT.get_value(MT.getDistribution(dist_zone, 'Vertex'))
  distrib_face = PT.get_value(MT.getDistribution(dist_zone, 'Cell')) #In 2d, cell == face
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
  dmesh_nodal.set_coordinnates(dvtx_coord)

  ngon_node = PT.Zone.NGonNode(dist_zone)
  ngon_eso  = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
  ngon_ec   = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]

  ngon_distri = MT.getDistribution(ngon_node, 'ElementConnectivity')[1]

  dface_vtx_idx = np.add(ngon_eso, -ngon_distri[0], dtype=np.int32) #Local index is int32bits
  dface_vtx = as_pdm_gnum(ngon_ec)
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
  distrib_vtx = PT.get_value(MT.getDistribution(dist_zone, 'Vertex'))
  n_vtx   = distrib_vtx[2]
  dn_vtx  = distrib_vtx[1] - distrib_vtx[0]

  n_elt_per_dim  = [0,0,0]
  sorted_elts_by_dim = PT.Zone.get_ordered_elements_per_dim(dist_zone)
  for elt_dim in sorted_elts_by_dim:
    for elt in elt_dim:
      assert PT.Element.CGNSName(elt) not in ["NGON_n", "NFACE_n"]
      if PT.Element.Dimension(elt) > 0:
        n_elt_per_dim[PT.Element.Dimension(elt)-1] += PT.Element.Size(elt)

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
      dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])
    else:
      dvtx_coord = np.empty(0, dtype='float64', order='F')
    dmesh_nodal.set_coordinnates(dvtx_coord)

    # keep dvtx_coord object alive for ParaDiGM
    multi_part_node = PT.update_child(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
    PT.new_DataArray('dvtx_coord', dvtx_coord, parent=multi_part_node)

  #Elements
  to_elmt_size = lambda e : PT.get_value(MT.getDistribution(e, 'Element'))[1] - PT.get_value(MT.getDistribution(e, 'Element'))[0]

  for i_dim, elts in enumerate(sorted_elts_by_dim):
    elt_pdm_types = np.array([MT.pdm_elts.element_pdm_type(PT.Element.Type(e)) for e in elts], dtype=np.int32)
    elt_lengths   = np.array([to_elmt_size(e) for e in elts], dtype=np.int32)
    elmts_connectivities = [as_pdm_gnum(PT.get_child_from_name(e, "ElementConnectivity")[1]) for e in elts]
    dmesh_nodal.set_sections(MT.pdm_elts.elements_dim_to_pdm_kind[i_dim], elmts_connectivities, elt_pdm_types, elt_lengths)

  # Boundaries
  if needs_bc:
    range_by_dim = PT.Zone.get_elt_range_per_dim(dist_zone)
    # print("range_by_dim --> ", range_by_dim)

    bc_point_lists = collect_distributed_pl(dist_zone, [['ZoneBC_t', 'BC_t']])
    # Find out in which dim the boundary refers
    bc_point_lists_by_dim = _split_point_list_by_dim(bc_point_lists, range_by_dim, comm)

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

        dmesh_nodal.set_group_elmt(MT.pdm_elts.elements_dim_to_pdm_kind[i_dim], n_elmt_group, delmt_bound_idx, delmt_bound)

  return dmesh_nodal

