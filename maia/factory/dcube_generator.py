from mpi4py import MPI
import numpy as np
import Pypdm.Pypdm as PDM

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.pytree.sids import elements_utils as EU

import maia
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils import np_utils, par_utils, layouts

def _add_sections_to_zone(dist_zone, section, shift_elmt, comm):
  """
  """
  if section == None: return shift_elmt
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  for i_section, section in enumerate(section["sections"]):
    # print("section = ", section)
    cgns_elmt_type = MT.pdm_elts.pdm_elt_name_to_cgns_element_type(section["pdm_type"])
    # print("cgns_elmt_type :", cgns_elmt_type)

    elmt = I.newElements(f"{cgns_elmt_type}.{i_section}", cgns_elmt_type,
                         erange = [shift_elmt, shift_elmt + section["np_distrib"][n_rank]-1], parent=dist_zone)
    I.newDataArray('ElementConnectivity', section["np_connec"], parent=elmt)

    shift_elmt += section["np_distrib"][n_rank]

    distrib   = section["np_distrib"][[i_rank, i_rank+1, n_rank]]
    MT.newDistribution({'Element' : distrib}, parent=elmt)

  return shift_elmt

# --------------------------------------------------------------------------
def dcube_generate(n_vtx, edge_length, origin, comm):
  """
  This function calls paradigm to generate a distributed mesh of a cube, and
  return a CGNS PyTree
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  dcube = PDM.DCubeGenerator(n_vtx, edge_length, *origin, comm)

  dcube_dims = dcube.dcube_dim_get()
  dcube_val  = dcube.dcube_val_get()

  distrib_cell    = par_utils.gather_and_shift(dcube_dims['dn_cell'],   comm, pdm_gnum_dtype)
  distrib_vtx     = par_utils.gather_and_shift(dcube_dims['dn_vtx'],    comm, pdm_gnum_dtype)
  distrib_face    = par_utils.gather_and_shift(dcube_dims['dn_face'],   comm, pdm_gnum_dtype)
  distrib_facevtx = par_utils.gather_and_shift(dcube_dims['sface_vtx'], comm, pdm_gnum_dtype)

  # > Generate dist_tree
  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase(parent=dist_tree)
  dist_zone = I.newZone('zone', [[distrib_vtx[n_rank], distrib_cell[n_rank], 0]],
                        'Unstructured', parent=dist_base)

  # > Grid coordinates
  cx, cy, cz = layouts.interlaced_to_tuple_coords(dcube_val['dvtx_coord'])
  grid_coord = I.newGridCoordinates(parent=dist_zone)
  I.newDataArray('CoordinateX', cx, parent=grid_coord)
  I.newDataArray('CoordinateY', cy, parent=grid_coord)
  I.newDataArray('CoordinateZ', cz, parent=grid_coord)

  # > NGon node
  dn_face = dcube_dims['dn_face']

  # > For Offset we have to shift to be global
  eso = distrib_facevtx[i_rank] + dcube_val['dface_vtx_idx'].astype(pdm_gnum_dtype)

  pe     = dcube_val['dface_cell'].reshape(dn_face, 2)
  np_utils.shift_nonzeros(pe, distrib_face[n_rank])
  ngon_n = I.newElements('NGonElements', 'NGON',
                         erange = [1, distrib_face[n_rank]], parent=dist_zone)

  I.newDataArray('ElementConnectivity', dcube_val['dface_vtx'], parent=ngon_n)
  I.newDataArray('ElementStartOffset' , eso                   , parent=ngon_n)
  I.newDataArray('ParentElements'     , pe                    , parent=ngon_n)

  # > BCs
  zone_bc = I.newZoneBC(parent=dist_zone)

  face_group_idx = dcube_val['dface_group_idx']

  face_group = dcube_val['dface_group']

  bc_names = ['Zmin', 'Zmax', 'Xmin', 'Xmax', 'Ymin', 'Ymax']
  for i_bc in range(dcube_dims['n_face_group']):
    bc_n = I.newBC(bc_names[i_bc], btype='Null', parent=zone_bc)
    I.newGridLocation('FaceCenter', parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    I.newPointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)

    bc_distrib = par_utils.gather_and_shift(dn_face_bnd, comm, pdm_gnum_dtype)
    distrib   = np.array([bc_distrib[i_rank], bc_distrib[i_rank+1], bc_distrib[n_rank]])
    MT.newDistribution({'Index' : distrib}, parent=bc_n)

  # > Distributions
  np_distrib_cell    = np.array([distrib_cell[i_rank]   , distrib_cell[i_rank+1]   , distrib_cell[n_rank]]   )
  np_distrib_vtx     = np.array([distrib_vtx[i_rank]    , distrib_vtx[i_rank+1]    , distrib_vtx[n_rank]]    )
  np_distrib_face    = np.array([distrib_face[i_rank]   , distrib_face[i_rank+1]   , distrib_face[n_rank]]   )
  np_distrib_facevtx = np.array([distrib_facevtx[i_rank], distrib_facevtx[i_rank+1], distrib_facevtx[n_rank]])

  MT.newDistribution({'Cell' : np_distrib_cell, 'Vertex' : np_distrib_vtx}, parent=dist_zone)
  MT.newDistribution({'Element' : np_distrib_face, 'ElementConnectivity' : np_distrib_facevtx}, parent=ngon_n)

  return dist_tree

# --------------------------------------------------------------------------
def dcube_nodal_generate(n_vtx, edge_length, origin, cgns_elmt_name, comm):
  """
  This function calls paradigm to generate a distributed mesh of a cube with various type of elements, and
  return a CGNS PyTree
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  t_elmt = MT.pdm_elts.cgns_elt_name_to_pdm_element_type(cgns_elmt_name)
  cgns_elt_index = [prop[0] for prop in EU.elements_properties].index(cgns_elmt_name)
  cell_dim = EU.element_dim(cgns_elt_index)

  # Manage 2D meshes with 2D PhyDim
  phy_dim = 3
  if len(origin) == 2:
    phy_dim  = 2
    origin = list(origin) + [0.]
  if cell_dim == 3:
    assert phy_dim == 3

  if isinstance(n_vtx, int):
    n_vtx = [n_vtx, n_vtx, n_vtx]
  assert isinstance(n_vtx, list) and len(n_vtx) == 3

  dcube = PDM.DCubeNodalGenerator(*n_vtx, edge_length, *origin, t_elmt, 1, comm)
  dcube.set_ordering("PDM_HO_ORDERING_CGNS".encode('utf-8'))
  dcube.compute()

  dmesh_nodal = dcube.get_dmesh_nodal()

  g_dims      = dmesh_nodal.dmesh_nodal_get_g_dims()
  n_vtx_out   = g_dims['n_vtx_abs']
  n_cell_out  = g_dims['n_face_abs'] if cell_dim == 2 else g_dims['n_cell_abs']


  sections_vol   = None
  sections_surf  = None
  sections_ridge = None
  if cgns_elmt_name.split('_')[0] in ["TRI", "QUAD"]:
    sections_surf  = dmesh_nodal.dmesh_nodal_get_sections(PDM._PDM_GEOMETRY_KIND_SURFACIC, comm)
    sections_ridge = dmesh_nodal.dmesh_nodal_get_sections(PDM._PDM_GEOMETRY_KIND_RIDGE   , comm)
    groups         = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_RIDGE)
  else:
    sections_vol   = dmesh_nodal.dmesh_nodal_get_sections(PDM._PDM_GEOMETRY_KIND_VOLUMIC , comm)
    sections_surf  = dmesh_nodal.dmesh_nodal_get_sections(PDM._PDM_GEOMETRY_KIND_SURFACIC, comm)
    groups         = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_SURFACIC)

  # > Generate dist_tree
  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase('Base', cellDim=cell_dim, physDim=phy_dim, parent=dist_tree)
  dist_zone = I.newZone('zone', [[n_vtx_out, n_cell_out, 0]],
                        'Unstructured', parent=dist_base)

  # > Grid coordinates
  vtx_data = dmesh_nodal.dmesh_nodal_get_vtx(comm)
  cx, cy, cz = layouts.interlaced_to_tuple_coords(vtx_data['np_vtx'])
  grid_coord = I.newGridCoordinates(parent=dist_zone)
  I.newDataArray('CoordinateX', cx, parent=grid_coord)
  I.newDataArray('CoordinateY', cy, parent=grid_coord)
  if phy_dim == 3:
    I.newDataArray('CoordinateZ', cz, parent=grid_coord)

  # > Section implicitement range donc on maintiens un compteur
  shift_elmt       = 1
  shift_elmt_vol   = _add_sections_to_zone(dist_zone, sections_vol  , shift_elmt     , comm)
  shift_elmt_surf  = _add_sections_to_zone(dist_zone, sections_surf , shift_elmt_vol , comm)
  shift_elmt_ridge = _add_sections_to_zone(dist_zone, sections_ridge, shift_elmt_surf, comm)

  # > BCs
  shift_bc = shift_elmt_vol - 1 if sections_vol is not None else shift_elmt_surf - 1
  zone_bc = I.newZoneBC(parent=dist_zone)

  face_group_idx = groups['dgroup_elmt_idx']

  face_group = shift_bc + groups['dgroup_elmt']
  distri = np.empty(n_rank, dtype=face_group.dtype)
  n_face_group = face_group_idx.shape[0] - 1

  if cell_dim == 2:
    bc_names = ['Zmin', 'Zmax', 'Ymin', 'Ymax']
    bc_loc = 'EdgeCenter'
  else:
    bc_names = ['Zmin', 'Zmax', 'Xmin', 'Xmax', 'Ymin', 'Ymax']
    bc_loc = 'FaceCenter'
  for i_bc in range(n_face_group):
    bc_n = I.newBC(bc_names[i_bc], btype='Null', parent=zone_bc)
    I.newGridLocation(bc_loc, parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    I.newPointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)

    bc_distrib = par_utils.gather_and_shift(dn_face_bnd, comm, pdm_gnum_dtype)
    distrib    = bc_distrib[[i_rank, i_rank+1, n_rank]]
    MT.newDistribution({'Index' : distrib}, parent=bc_n)

  # > Distributions
  np_distrib_cell = par_utils.uniform_distribution(g_dims["n_cell_abs"], comm)

  distri_vtx     = vtx_data['np_vtx_distrib']
  np_distrib_vtx = distri_vtx[[i_rank, i_rank+1, n_rank]]

  MT.newDistribution({'Cell' : np_distrib_cell, 'Vertex' : np_distrib_vtx}, parent=dist_zone)

  return dist_tree


def generate_dist_block(n_vtx, cgns_elmt_name, comm, origin=np.zeros(3), edge_length=1.):
  """Generate a distributed mesh with a cartesian topology.
  
  Returns a distributed CGNSTree containing a single :cgns:`CGNSBase_t` and
  :cgns:`Zone_t`. The kind 
  and cell dimension of the zone is controled by the cgns_elmt_name parameter: 

  - ``"Structured"`` (or ``"S"``) produces a 3d structured zone (not yet implemented),
  - ``"Poly"`` produces an unstructured 3d zone with a NGon+PE connectivity,
  - ``"NFACE_n"`` produces an unstructured 3d zone with a NFace+NGon connectivity,
  - ``"NGON_n"``  produces an unstructured 2d zone with faces described by a NGon
    node (not yet implemented),
  - Other names must be in ``["TRI_3", "QUAD_4", "TETRA_4", "PENTA_6", "HEXA_8"]``
    and produces an unstructured 2d or 3d zone with corresponding standard elements.

  In all cases, the created zone contains the cartesian grid coordinates and the relevant number
  of boundary conditions.

  When creating 2 dimensional zones, the
  `physical dimension <https://cgns.github.io/CGNS_docs_current/sids/cgnsbase.html#CGNSBase>`_
  is set equal to the length of the origin parameter.

  Args:
    n_vtx (int or array of int) : Number of vertices in each direction. Scalars
      automatically extend to uniform array.
    cgns_elmt_name (str) : requested kind of elements
    comm       (MPIComm) : MPI communicator
    origin (array, optional) : Coordinates of the origin of the generated mesh. Defaults
        to zero vector.
    edge_length (float, optional) : Edge size of the generated mesh. Defaults to 1.
  Returns:
    CGNSTree: distributed cgns tree

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #generate_dist_block@start
        :end-before: #generate_dist_block@end
        :dedent: 2
  """
  if cgns_elmt_name is None or cgns_elmt_name in ["Structured", "S"]:
    raise NotImplementedError
  elif cgns_elmt_name.upper() in ["POLY", "NFACE_N"]:
    dist_tree = dcube_generate(n_vtx, edge_length, origin, comm)
    if cgns_elmt_name.upper() == "NFACE_N":
      for zone in PT.get_all_Zone_t(dist_tree):
        maia.algo.pe_to_nface(zone, comm, removePE=True)
    return dist_tree
  else:
    return dcube_nodal_generate(n_vtx, edge_length, origin, cgns_elmt_name, comm)

