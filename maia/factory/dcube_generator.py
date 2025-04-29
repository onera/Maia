import numpy as np

import maia
from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.pytree.sids import elements_utils as EU
from maia.utils import np_utils, par_utils, layouts
from maia       import npy_pdm_gnum_dtype           as pdm_gnum_dtype

from .dline_generator import generate_dist_line
import Pypdm.Pypdm as PDM

def _dmesh_nodal_to_cgns_zone(dmesh_nodal, comm: MPIComm, elt_min_dim: int =0) -> CGNSTree:

  g_dims  = dmesh_nodal.dmesh_nodal_get_g_dims()
  n_vtx   = g_dims['n_vtx_abs']
  n_cell  = g_dims['n_face_abs'] if g_dims['n_cell_abs'] == 0 else g_dims['n_cell_abs']
  max_dim = 3 if g_dims['n_cell_abs'] != 0 else 2

  zone = PT.new_Zone('zone', size=np.array([[n_vtx, n_cell, 0]],dtype=pdm_gnum_dtype), type='Unstructured')

  # > Grid coordinates
  vtx_data = dmesh_nodal.dmesh_nodal_get_vtx(comm)
  cx, cy, cz = layouts.interlaced_to_tuple_coords(vtx_data['np_vtx'])
  coords = {'CoordinateX' : cx, 'CoordinateY' : cy, 'CoordinateZ' : cz}
  grid_coord = PT.new_GridCoordinates(fields=coords, parent=zone)

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

      _erange = np.array([elt_shift, elt_shift+distrib[-1]-1], section["np_connec"].dtype)
      elmt = PT.new_Elements(f"{cgns_elmt_name}.{i_section}", cgns_elmt_name, \
          erange=_erange, econn=section["np_connec"], parent=zone)
      MT.newDistribution({'Element' : distrib}, parent=elmt)
      elt_shift += distrib[-1]

  # > Distributions
  np_distrib_cell = par_utils.uniform_distribution(n_cell, comm)
  np_distrib_vtx  = par_utils.uniform_distribution(n_vtx,  comm)

  MT.newDistribution({'Cell' : np_distrib_cell, 'Vertex' : np_distrib_vtx}, parent=zone)

  return zone
    

# --------------------------------------------------------------------------
def dcube_generate(n_vtx: int,
                   edge_length: float,
                   origin: Tuple[float, float, float],
                   comm: MPIComm) -> CGNSDistTree:
  """
  This function calls paradigm to generate a distributed mesh of a cube, and
  return a CGNS PyTree
  """
  if not isinstance(n_vtx, int):
    raise NotImplementedError("Poly/NFACE_n generation does not supports variable number of vertices")

  dcube = PDM.DCubeGenerator(n_vtx, edge_length, *origin, comm)
  dcube_dims = dcube.dcube_dim_get()
  dcube_val  = dcube.dcube_val_get()

  distrib_cell    = par_utils.dn_to_distribution(dcube_dims['dn_cell'],   comm)
  distrib_vtx     = par_utils.dn_to_distribution(dcube_dims['dn_vtx'],    comm)
  distrib_face    = par_utils.dn_to_distribution(dcube_dims['dn_face'],   comm)
  distrib_facevtx = par_utils.dn_to_distribution(dcube_dims['sface_vtx'], comm)

  # > Generate dist_tree
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase(parent=dist_tree)
  dist_zone = PT.new_Zone('zone', size=[[distrib_vtx[-1], distrib_cell[-1], 0]],
                        type='Unstructured', parent=dist_base)

  # > Grid coordinates
  cx, cy, cz = layouts.interlaced_to_tuple_coords(dcube_val['dvtx_coord'])
  coords = {'CoordinateX' : cx, 'CoordinateY' : cy, 'CoordinateZ' : cz}
  grid_coord = PT.new_GridCoordinates(fields=coords, parent=dist_zone)

  # > NGon node
  dn_face = dcube_dims['dn_face']

  # > For Offset we have to shift to be global
  eso = distrib_facevtx[0] + np_utils.safe_int_cast(dcube_val['dface_vtx_idx'], distrib_face.dtype)

  pe     = dcube_val['dface_cell'].reshape(dn_face, 2)
  np_utils.shift_nonzeros(pe, distrib_face[-1])
  ngon_n = PT.new_NGonElements('NGonElements', 
                               erange = [1, distrib_face[-1]], parent=dist_zone,
                               eso = eso, ec = dcube_val['dface_vtx'], pe = pe)
  # > BCs
  zone_bc = PT.new_ZoneBC(parent=dist_zone)

  face_group_idx = dcube_val['dface_group_idx']

  face_group = dcube_val['dface_group']

  bc_names = ['Zmin', 'Zmax', 'Xmin', 'Xmax', 'Ymin', 'Ymax']
  for i_bc in range(dcube_dims['n_face_group']):
    bc_n = PT.new_BC(bc_names[i_bc], type='Null', parent=zone_bc)
    PT.new_GridLocation('FaceCenter', parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    PT.new_IndexArray(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)

    distrib  = par_utils.dn_to_distribution(dn_face_bnd, comm)
    MT.newDistribution({'Index' : distrib}, parent=bc_n)

  # > Distributions
  MT.newDistribution({'Cell' : distrib_cell, 'Vertex' : distrib_vtx}, parent=dist_zone)
  MT.newDistribution({'Element' : distrib_face, 'ElementConnectivity' : distrib_facevtx}, parent=ngon_n)

  return dist_tree

# --------------------------------------------------------------------------
def dcube_nodal_generate(n_vtx: Union[int, Sequence[int]], 
                         edge_length: float,
                         origin: Sequence[float],
                         cgns_elmt_name: str,
                         comm: MPIComm, 
                         get_ridges: bool=False) -> CGNSDistTree:
  """
  This function calls paradigm to generate a distributed mesh of a cube with various type of elements, and
  return a CGNS PyTree
  """

  t_elmt = MT.pdm_elts.cgns_elt_name_to_pdm_element_type(cgns_elmt_name)
  cgns_elt_index = [prop[0] for prop in EU.elements_properties].index(cgns_elmt_name)
  cell_dim = EU.element_dim(cgns_elt_index)
  assert cell_dim is not None

  # Manage 2D meshes with 2D PhyDim
  phy_dim = 3
  if len(origin) == 2:
    phy_dim  = 2
    origin = list(origin) + [0.]
  if cell_dim == 3:
    assert phy_dim == 3

  if isinstance(n_vtx, int):
    n_vtx = [n_vtx, n_vtx, n_vtx]
  elif phy_dim==2 and len(n_vtx)==2:
    n_vtx = list(n_vtx)+[1]
  assert len(n_vtx) == 3

  dcube = PDM.DCubeNodalGenerator(*n_vtx, edge_length, *origin, t_elmt, 1, comm)
  dcube.set_ordering("PDM_HO_ORDERING_CGNS".encode('utf-8'))
  dcube.compute()

  dmesh_nodal = dcube.get_dmesh_nodal()

  # > Generate dist_tree
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase('Base', cell_dim=cell_dim, phy_dim=phy_dim, parent=dist_tree)

  min_elt_dim = 0 if get_ridges else cell_dim - 1
  dist_zone = _dmesh_nodal_to_cgns_zone(dmesh_nodal, comm, min_elt_dim)
  PT.add_child(dist_base, dist_zone)

  if phy_dim == 2:
    PT.rm_node_from_path(dist_zone, 'GridCoordinates/CoordinateZ')

  # > BCs
  if cell_dim == 2:
    bc_names = ['Ymin', 'Ymax', 'Xmin', 'Xmax']
    bc_loc = 'EdgeCenter'
    groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_RIDGE)
  else:
    bc_names = ['Zmin', 'Zmax', 'Xmin', 'Xmax', 'Ymin', 'Ymax']
    bc_loc = 'FaceCenter'
    groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_SURFACIC)

  range_per_dim = PT.Zone.get_elt_range_per_dim(dist_zone)
  shift_bc = range_per_dim[cell_dim][1]

  zone_bc = PT.new_ZoneBC(parent=dist_zone)

  face_group_idx = groups['dgroup_elmt_idx']

  face_group = shift_bc + groups['dgroup_elmt']
  n_face_group = face_group_idx.shape[0] - 1

  for i_bc in range(n_face_group):
    bc_n = PT.new_BC(bc_names[i_bc], type='Null', parent=zone_bc)
    PT.new_GridLocation(bc_loc, parent=bc_n)
    start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
    dn_face_bnd = end - start
    PT.new_IndexArray(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)
    MT.newDistribution({'Index' : par_utils.dn_to_distribution(dn_face_bnd, comm)}, parent=bc_n)

  return dist_tree

def dcube_struct_generate(n_vtx: Union[int, Sequence[int]], 
                          edge_length: Union[float, Sequence[float]], 
                          origin: Sequence[float],
                          comm: MPIComm, 
                          bc_location: str='Vertex') -> CGNSDistTree:
  max_coords = np.asarray(origin).copy() + np.asarray(edge_length)

  dist_tree = maia.factory.generate_dist_points(n_vtx, "Structured", comm, origin, max_coords)
  dist_base = PT.request_node_from_label(dist_tree, 'CGNSBase_t')
  dist_zone = PT.request_node_from_label(dist_tree, 'Zone_t')

  # Update zone dims
  zone_dims = PT.get_value(dist_zone, raw=True)
  assert zone_dims is not None
  cell_dim = zone_dims.shape[0]
  for dim in range(cell_dim):
    zone_dims[dim,1] = zone_dims[dim, 0] - 1

  # If n_vtx == 1 in one dir, we remove this dir
  while zone_dims[-1][1] == 0:
    zone_dims = zone_dims[:-1,:]
    cell_dim -= 1
  # Update
  dist_base[1][0] = cell_dim #type:ignore[index] #(base value is not None)
  PT.set_value(dist_zone, zone_dims)

  # Update Cell distribution and add face distribution
  distrib = {'Cell' : par_utils.uniform_distribution(PT.Zone.n_cell(dist_zone), comm)}
  if cell_dim == 3:
      distrib['Face'] =  par_utils.uniform_distribution(PT.Zone.n_face(dist_zone), comm)
  MT.newDistribution(distrib, dist_zone)

  # Create BCs
  zbc = PT.new_child(dist_zone, 'ZoneBC', 'ZoneBC_t')

  xyz_to_faceloc = {'X':'IFaceCenter', 'Y':'JFaceCenter', 'Z':'KFaceCenter'}
  offset = 1 if bc_location=='FaceCenter' else 0
  for idim, dir in enumerate(['X', 'Y', 'Z']):
    if cell_dim > idim:
      location = bc_location if bc_location=='Vertex' else xyz_to_faceloc[dir]
      mask = np_utils.others_mask(np.arange(cell_dim), [idim])

      pr = np.ones((cell_dim, 2), dtype=zone_dims.dtype)
      pr[mask, 1] = zone_dims[mask, 0]-offset
      bc = PT.new_BC(f'{dir}min', point_range=pr, loc=location, parent=zbc)

      pr = np.ones((cell_dim, 2), dtype=zone_dims.dtype)
      pr[idim, :] = zone_dims[idim,0]
      pr[mask, 1] = zone_dims[mask, 0]-offset
      bc = PT.new_BC(f'{dir}max', point_range=pr, loc=location, parent=zbc)

  for bc in PT.get_children(zbc):
    distri = par_utils.uniform_distribution(PT.Subset.n_elem(bc), comm)
    MT.newDistribution({'Index' : distri}, bc)

  return dist_tree


def generate_dist_block(n_vtx: Union[int, Sequence[int]],
                        cgns_elmt_name: str,
                        comm: MPIComm,
                        origin: Sequence[float] = (0,0,0), 
                        length: Union[float, Sequence[float], List[Sequence[float]]] = 1.) -> CGNSDistTree:
  """Generate a distributed mesh with a block shape (line, parallelogram or parallelepiped). 
  
  This function returns a distributed CGNSTree containing a single :cgns:`CGNSBase_t` and
  :cgns:`Zone_t`. The created zone contains the grid coordinates and the relevant number
  of boundary conditions.
  
  The kind and CGNS cell dimension :math:`d_m` of the zone is controled by the ``cgns_elmt_name`` parameter: 

  - ``"Structured"`` (or ``"S"``) produces a structured zone, which dimension is equal to ``n_vtx.size``,
  - ``"NFACE_n"`` produces a hexa filled unstructured 3d zone with NFace+NGon connectivity,
  - ``"NGON_n"``  produces a quad filled unstructured 2d zone with NGon+Bar connectivity
    (**not yet implemented**),
  - Other names must be in ``["BAR_2", "TRI_3", "QUAD_4", "TETRA_4", "PYRA_5", "PENTA_6", "HEXA_8"]``
    and produces an unstructured 1d, 2d or 3d zone with corresponding standard elements.

  The `CGNS physical dimension <https://cgns.github.io/CGNS_docs_current/sids/cgnsbase.html#CGNSBase>`_
  :math:`d_\\phi` is deduced from the shape of the ``origin`` parameter. Note that the physical dimension must be
  upper or equal to the cell dimension.

  The number of vertices in each direction is given by ``n_vtx`` parameter, which is a tuple of 
  size :math:`d_m`. If a scalar is provided, its value is broadcasted to a uniform tuple.

  Lastly, the geometric size and the position of the zone is computed from the combination of ``origin`` and
  ``length`` parameters. The first one, which is an array of size :math:`d_\\phi`, set the position of the
  'first vertex' of the zone. The length parameter can be either:

  - a scalar, which leads to a line, a square or a cube aligned with the canonical axes. Its length is
    then the same in each direction;
  - a tuple of size :math:`d_m`, which leads to a line, a rectangle or a rectangular cuboid aligned with
    the canonical axes. Its length is then equal to the specified value in each direction;
  - a list of :math:`d_m` vectors, each one of size :math:`d_\\phi`. In this case, the generated line, parallelogram
    or parallelepiped is no more aligned with the canonical axes, but with the provided basis.
    Its length is equal to the norm of the basis vector in each direction.

  Note that ``length`` can contain negative values.

  Args:
    n_vtx (int or tuple of int) : Number of vertices in each direction. Scalars
      automatically extend to uniform array.
    cgns_elmt_name (str) : requested kind of elements
    comm       (MPIComm) : MPI communicator
    origin (array, optional) : Coordinates of the origin of the generated mesh. Defaults
        to zero vector.
    length (float or tuple(s) of floats, optional) : Length for each dimension of the generated mesh
      (see above). Defaults to 1.
  Returns:
    CGNSTree: distributed cgns tree

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #generate_dist_block@start
        :end-before: #generate_dist_block@end
        :dedent: 2
  """
  # > Retrive entry dimensions
  phy_dim = len(origin)
  if cgns_elmt_name in ['Structured', 'S']:
    if isinstance(n_vtx, Iterable):
      cell_dim = len(n_vtx)
      for k in n_vtx[::-1]:
        if k != 1:
          break
        cell_dim -= 1
    else: # Can not guess from scalar n_vtx --> use origin vector
      cell_dim = len(origin)
  elif cgns_elmt_name == 'BAR_2':
    cell_dim = 1
  elif cgns_elmt_name in ['TRI_3', 'QUAD_4', 'NGON_n']:
    cell_dim = 2
  else:
    cell_dim = 3

  # > Convert length to full vector
  need_matrix = False
  need_scaling = False
  # Special case of BAR_2 : we allow [l1,l2,l3] to be converted in [[l1,l2,l3]]
  if cgns_elmt_name == 'BAR_2' and isinstance(length, Iterable) and not isinstance(length[0], Iterable) and len(length) == phy_dim:
    length = [length] #type:ignore[assignment] #(length is of kind Sequence[float])

  if not isinstance(length, Iterable): # Scalar case : extend to tuple case
    need_scaling = length != 1.
    #length = np.full(cell_dim, length, dtype=np.float64)
    length = cell_dim * [float(length)]
  elif not isinstance(length[0], Iterable): # tuple case
    assert len(length) == cell_dim, f"length argument is a tuple (case 2), but its size is not equal to CellDimension ({len(length)} vs {cell_dim})"
    need_scaling = True
  else: # List of Sequence case
    inner_lens = [len(ld) for ld in length] #type:ignore[arg-type] #(ld is of type Sequence[float])
    msg_outer = f"length argument is a list of tuple (case 3), but its size is not equal to CellDimension ({len(length)} vs {cell_dim})"
    msg_inner = f"length argument is a list of tuple (case 3), but the size of each tuple is not equal to PhysicalDimension ({inner_lens}) vs {phy_dim})"
    assert len(length) == cell_dim, msg_outer
    assert all([l == phy_dim for l in inner_lens]), msg_inner

    need_matrix = True

  # First case manage correctly origin / length --> direct return of disttree
  # For BAR_2 we have lenght = List[Sequence[float]] (need_matrix=True) or Sequence[float] (size 1) (need_matrix=False)
  if cgns_elmt_name in ["BAR_2"]:
    if isinstance(length[0], Iterable):
      end = [c + length[0][i] for i,c in enumerate(origin)]
    else:
      end = [c + (i==0)*length[0] for i,c in enumerate(origin)]
    _n_vtx = n_vtx[0] if isinstance(n_vtx, Iterable) else n_vtx
    return generate_dist_line(_n_vtx, origin, end, comm)
  
  # Structured case manage case 2, but not general case --> generate unit mesh in general case to rescale afterward
  elif cgns_elmt_name in ["Structured", "S"]:
    if not need_matrix:
      _length:List[float] = list(length) + [0.] * (phy_dim-cell_dim) #type:ignore[assignment] #(not need_matrix => single list)
      return dcube_struct_generate(n_vtx, _length, origin, comm)
    else:
      dist_tree = dcube_struct_generate(n_vtx, 1., phy_dim*[0.], comm)

  # Other cases do not manage anything: use origin=0., length=1. (unit mesh), and rescale afterward
  elif cgns_elmt_name.upper() in ["POLY", "NFACE_N"]:
    if not isinstance(n_vtx, int):
      raise NotImplementedError("Poly/NFACE_n generation does not supports variable number of vertices")
    dist_tree = dcube_generate(n_vtx, 1., (0,0,0), comm)
    if cgns_elmt_name.upper() == "NFACE_N":
      for zone in PT.get_all_Zone_t(dist_tree):
        maia.algo.pe_to_nface(zone, comm, removePE=True)
  else:
    dist_tree = dcube_nodal_generate(n_vtx, 1., [0.]*phy_dim, cgns_elmt_name, comm)

  # > Apply scaling and transform
  if need_matrix:
    matrix = np.eye(phy_dim)
    matrix[:,0:cell_dim] = np.asarray(length).T
    zone = PT.get_all_Zone_t(dist_tree)[0]
    coords = [PT.request_node_from_path(zone, f'GridCoordinates/Coordinate{dir}') for dir in 'XYZ'[0:phy_dim]]
    coords_val:List[NDArray] = [PT.get_value(c) for c in coords] #type:ignore #(coords should not be None)
    tr_coords = np_utils.matmul_cart_vectors(coords_val, matrix)
    for coord_n, new_c in zip(coords, tr_coords):
      PT.set_value(coord_n, new_c)
  elif need_scaling:
    scale_length:List[float] = list(length) + [1.]*(3-cell_dim) #type:ignore[assignment] #(need_scaling => single list)
    maia.algo.scale_mesh(dist_tree, scale_length)

  for dim, coord_name in enumerate(['CoordinateX', 'CoordinateY', 'CoordinateZ'][:phy_dim]):
    coord_n = PT.request_node_from_name(dist_tree, coord_name)
    assert (coord_val:=PT.get_value(coord_n, True)) is not None
    PT.set_value(coord_n, coord_val+origin[dim])

  return dist_tree
