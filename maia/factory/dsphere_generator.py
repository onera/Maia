import numpy as np
import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils     import par_utils, layouts

from .dcube_generator import _dmesh_nodal_to_cgns_zone

def dsphere_vol_nodal_generate(n_vtx, radius, origin, comm):
  """ Generate a nodal (TETRA_4 + TRI_3) filled sphere """
  dmesh_nodal = PDM.sphere_vol_icosphere_gen_nodal(comm, n_vtx, *origin, radius)

  # > Generate dist_tree
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase('Base', cell_dim=3, phy_dim=3, parent=dist_tree)

  dist_zone = _dmesh_nodal_to_cgns_zone(dmesh_nodal, comm, elt_min_dim=2)
  PT.add_child(dist_base, dist_zone)

  bc_names = ['Skin']
  groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_SURFACIC)

  range_per_dim = PT.Zone.get_elt_range_per_dim(dist_zone)
  shift_bc = range_per_dim[3][1]

  zone_bc = PT.new_ZoneBC(parent=dist_zone)

  face_group_idx = groups['dgroup_elmt_idx']

  face_group = shift_bc + groups['dgroup_elmt']
  n_face_group = face_group_idx.shape[0] - 1

  for i_bc in range(n_face_group):
    bc_n = PT.new_BC(bc_names[i_bc], type='Null', parent=zone_bc)
    PT.new_GridLocation('FaceCenter', parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    PT.new_PointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)
    MT.newDistribution({'Index' : par_utils.dn_to_distribution(dn_face_bnd, comm)}, parent=bc_n)

  return dist_tree

def dsphere_surf_nodal_generate(n_vtx, radius, origin, comm):
  """ Generate a nodal (TRI_3) surfacic sphere """
  dmesh_nodal = PDM.sphere_surf_icosphere_gen_nodal(comm, n_vtx, *origin, radius)

  # > Generate dist_tree
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase('Base', cell_dim=2, phy_dim=3, parent=dist_tree)

  dist_zone = _dmesh_nodal_to_cgns_zone(dmesh_nodal, comm, elt_min_dim=1)
  PT.add_child(dist_base, dist_zone)

  return dist_tree

# --------------------------------------------------------------------------
def dsphere_hollow_nodal_generate(n_vtx, radius_int, radius_ext, origin, comm, n_layer=1, geometric_ratio=1):
  """ Generate a nodal (TETRA_4 + TRI_3) partially filled sphere (Spherical crown)
  nlayer = number of cell layers in the filled part
  geometric_ratio : control the size of cell layers in the filled part (geometric progression)
  """
  dmesh_nodal = PDM.sphere_vol_hollow_gen_nodal(comm, n_vtx, n_layer, *origin, radius_int, radius_ext, geometric_ratio)

  # > Generate dist_tree
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase('Base', cell_dim=3, phy_dim=3, parent=dist_tree)

  dist_zone = _dmesh_nodal_to_cgns_zone(dmesh_nodal, comm, elt_min_dim=2)
  PT.add_child(dist_base, dist_zone)

  bc_names = ['Skin', 'Farfield']
  groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_SURFACIC)

  range_per_dim = PT.Zone.get_elt_range_per_dim(dist_zone)
  shift_bc = range_per_dim[3][1]

  zone_bc = PT.new_ZoneBC(parent=dist_zone)

  face_group_idx = groups['dgroup_elmt_idx']

  face_group = shift_bc + groups['dgroup_elmt']
  n_face_group = face_group_idx.shape[0] - 1

  for i_bc in range(n_face_group):
    bc_n = PT.new_BC(bc_names[i_bc], type='Null', parent=zone_bc)
    PT.new_GridLocation('FaceCenter', parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    PT.new_PointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)
    MT.newDistribution({'Index' : par_utils.dn_to_distribution(dn_face_bnd, comm)}, parent=bc_n)

  return dist_tree


# --------------------------------------------------------------------------

def generate_dist_sphere(m, cgns_elmt_name, comm, origin=np.zeros(3), radius=1.):
  """Generate a distributed mesh with a spherical topology.
  
  Returns a distributed CGNSTree containing a single :cgns:`CGNSBase_t` and
  :cgns:`Zone_t`. The kind 
  and cell dimension of the zone is controled by the cgns_elmt_name parameter: 

  - ``"NFACE_n"`` produces an unstructured 3d zone with a NFace+NGon connectivity,
  - ``"NGON_n"``  produces an unstructured 2d zone with a NGon+Bar connectivity,
  - Other names must be in ``["TRI_3", "TETRA_4"]``
    and produces an unstructured 2d or 3d zone with corresponding standard elements.

  In all cases, the created zone contains the grid coordinates and the relevant number
  of boundary conditions.

  Spherical meshes are
  `class I geodesic polyhedra <https://en.wikipedia.org/wiki/Geodesic_polyhedron>`_
  (icosahedral). Number of vertices on the external surface is equal to
  :math:`10m^2+2`.


  Args:
    m (int) : Strict. positive integer who controls the number of vertices (see above)
    cgns_elmt_name (str) : requested kind of elements
    comm       (MPIComm) : MPI communicator
    origin (array, optional) : Coordinates of the origin of the generated mesh. Defaults
        to zero vector.
    radius (float, optional) : Radius of the generated sphere. Defaults to 1.
  Returns:
    CGNSTree: distributed cgns tree

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #generate_dist_sphere@start
        :end-before: #generate_dist_sphere@end
        :dedent: 2
  """
  from maia.algo.dist import convert_elements_to_ngon # Cyclic import

  assert m >= 1
  if cgns_elmt_name in ['TETRA_4', 'NFACE_n']:
    tree = dsphere_vol_nodal_generate(m-1, radius, origin, comm)
    if cgns_elmt_name == 'NFACE_n':
      convert_elements_to_ngon(tree, comm)
  elif cgns_elmt_name in ['TRI_3', 'NGON_n']:
    tree = dsphere_surf_nodal_generate(m-1, radius, origin, comm)
    if cgns_elmt_name == 'NGON_n':
      convert_elements_to_ngon(tree, comm)
  else:
    raise ValueError("Unvalid cgns_elmt_name")

  return tree
