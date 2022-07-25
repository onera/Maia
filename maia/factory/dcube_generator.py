import numpy as np
import Pypdm.Pypdm as PDM

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.pytree.sids import elements_utils as EU

import maia
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils import np_utils, par_utils, layouts

def _dmesh_nodal_to_cgns_zone(dmesh_nodal, comm, elt_min_dim=0):

  g_dims  = dmesh_nodal.dmesh_nodal_get_g_dims()
  n_vtx   = g_dims['n_vtx_abs']
  n_cell  = g_dims['n_face_abs'] if g_dims['n_cell_abs'] == 0 else g_dims['n_cell_abs']
  max_dim = 3 if g_dims['n_cell_abs'] != 0 else 2

  zone = I.newZone('zone', [[n_vtx, n_cell, 0]], 'Unstructured') 

  # > Grid coordinates
  vtx_data = dmesh_nodal.dmesh_nodal_get_vtx(comm)
  cx, cy, cz = layouts.interlaced_to_tuple_coords(vtx_data['np_vtx'])
  grid_coord = I.newGridCoordinates(parent=zone)
  I.newDataArray('CoordinateX', cx, parent=grid_coord)
  I.newDataArray('CoordinateY', cy, parent=grid_coord)
  I.newDataArray('CoordinateZ', cz, parent=grid_coord)

  # Carefull ! Getting elt of dim > cell_dim is not allowed
  pdm_section_kind = [PDM._PDM_GEOMETRY_KIND_CORNER, PDM._PDM_GEOMETRY_KIND_RIDGE,
                      PDM._PDM_GEOMETRY_KIND_SURFACIC, PDM._PDM_GEOMETRY_KIND_VOLUMIC]
  pdm_section_kind = pdm_section_kind[elt_min_dim:max_dim+1][::-1]

  sections_per_dim = [dmesh_nodal.dmesh_nodal_get_sections(kind, comm) for kind in pdm_section_kind]

  elt_shift = 1
  for dim_sections in sections_per_dim:
    for i_section, section in enumerate(dim_sections["sections"]):
      cgns_elmt_name = MT.pdm_elts.pdm_elt_name_to_cgns_element_type(section["pdm_type"])
      distrib   = par_utils.full_to_partial_distribution(section["np_distrib"], comm)

      elmt = I.newElements(f"{cgns_elmt_name}.{i_section}", cgns_elmt_name, parent=zone)
      I.newPointRange('ElementRange', [elt_shift, elt_shift + distrib[-1]-1], parent=elmt)
      I.newDataArray('ElementConnectivity', section["np_connec"], parent=elmt)
      MT.newDistribution({'Element' : distrib}, parent=elmt)
      elt_shift += distrib[-1]

  # > Distributions
  np_distrib_cell = par_utils.uniform_distribution(g_dims["n_cell_abs"], comm)
  np_distrib_vtx  = par_utils.full_to_partial_distribution(vtx_data['np_vtx_distrib'], comm)

  MT.newDistribution({'Cell' : np_distrib_cell, 'Vertex' : np_distrib_vtx}, parent=zone)

  return zone
    

# --------------------------------------------------------------------------
def dcube_generate(n_vtx, edge_length, origin, comm):
  """
  This function calls paradigm to generate a distributed mesh of a cube, and
  return a CGNS PyTree
  """
  dcube = PDM.DCubeGenerator(n_vtx, edge_length, *origin, comm)

  dcube_dims = dcube.dcube_dim_get()
  dcube_val  = dcube.dcube_val_get()

  distrib_cell    = par_utils.dn_to_distribution(dcube_dims['dn_cell'],   comm)
  distrib_vtx     = par_utils.dn_to_distribution(dcube_dims['dn_vtx'],    comm)
  distrib_face    = par_utils.dn_to_distribution(dcube_dims['dn_face'],   comm)
  distrib_facevtx = par_utils.dn_to_distribution(dcube_dims['sface_vtx'], comm)

  # > Generate dist_tree
  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase(parent=dist_tree)
  dist_zone = I.newZone('zone', [[distrib_vtx[-1], distrib_cell[-1], 0]],
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
  eso = distrib_facevtx[0] + dcube_val['dface_vtx_idx'].astype(pdm_gnum_dtype)

  pe     = dcube_val['dface_cell'].reshape(dn_face, 2)
  np_utils.shift_nonzeros(pe, distrib_face[-1])
  ngon_n = I.newElements('NGonElements', 'NGON',
                         erange = [1, distrib_face[-1]], parent=dist_zone)

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

    distrib  = par_utils.dn_to_distribution(dn_face_bnd, comm)
    MT.newDistribution({'Index' : distrib}, parent=bc_n)

  # > Distributions
  MT.newDistribution({'Cell' : distrib_cell, 'Vertex' : distrib_vtx}, parent=dist_zone)
  MT.newDistribution({'Element' : distrib_face, 'ElementConnectivity' : distrib_facevtx}, parent=ngon_n)

  return dist_tree

# --------------------------------------------------------------------------
def dcube_nodal_generate(n_vtx, edge_length, origin, cgns_elmt_name, comm, get_ridges=False):
  """
  This function calls paradigm to generate a distributed mesh of a cube with various type of elements, and
  return a CGNS PyTree
  """

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

  # > Generate dist_tree
  dist_tree = I.newCGNSTree()
  dist_base = I.newCGNSBase('Base', cellDim=cell_dim, physDim=phy_dim, parent=dist_tree)

  min_elt_dim = 0 if get_ridges else cell_dim - 1
  dist_zone = _dmesh_nodal_to_cgns_zone(dmesh_nodal, comm, min_elt_dim)
  I._addChild(dist_base, dist_zone)

  if phy_dim == 2:
    I.rmNodeByPath(dist_zone, 'GridCoordinates/CoordinateZ')

  # > BCs
  if cell_dim == 2:
    bc_names = ['Zmin', 'Zmax', 'Ymin', 'Ymax']
    bc_loc = 'EdgeCenter'
    groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_RIDGE)
  else:
    bc_names = ['Zmin', 'Zmax', 'Xmin', 'Xmax', 'Ymin', 'Ymax']
    bc_loc = 'FaceCenter'
    groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_SURFACIC)

  range_per_dim = PT.Zone.get_elt_range_per_dim(dist_zone)
  shift_bc = range_per_dim[cell_dim][1]

  zone_bc = I.newZoneBC(parent=dist_zone)

  face_group_idx = groups['dgroup_elmt_idx']

  face_group = shift_bc + groups['dgroup_elmt']
  n_face_group = face_group_idx.shape[0] - 1

  for i_bc in range(n_face_group):
    bc_n = I.newBC(bc_names[i_bc], btype='Null', parent=zone_bc)
    I.newGridLocation(bc_loc, parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    I.newPointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)
    MT.newDistribution({'Index' : par_utils.dn_to_distribution(dn_face_bnd, comm)}, parent=bc_n)

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

