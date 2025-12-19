import numpy as np

from maia.typing import *
import maia.pytree      as PT
import maia.pytree.maia as MT
from maia.utils           import py_utils, np_utils, par_utils, pr_utils
from maia.utils.numbering import range_to_slab          as HFR2S
from maia.transfer import protocols as EP

from maia.utils import logging as mlog
from .geometry import _compute_elements_center

def _to_xyz(r, theta, z):
  return r*np.cos(theta), r*np.sin(theta), z
def _to_xyz_vectors(vr, vtheta, vz, theta):
  return vr*np.cos(theta)-vtheta*np.sin(theta), vtheta*np.cos(theta)+vr*np.sin(theta), vz
def _to_rthetaz(x, y, z):
  return np.sqrt(x**2+y**2), np.arctan2(y, x), z
def _to_rthetaz_vectors(vx, vy, vz, theta):
  return vx*np.cos(theta)+vy*np.sin(theta), vy*np.cos(theta)-vx*np.sin(theta), vz


def update_fields(node, vtx_mask, phy_dim, rotation_center_np, rotation_angle_np, translation_np, positional_vectors, constant_vectors):
  transform_func = {2: np_utils.transform_cart_vectors_2d, 3: np_utils.transform_cart_vectors}[phy_dim]

  container_paths = set()
  def add_cnt_path(nodes):
    last_node = nodes[-1]
    if PT.get_child_from_label(last_node, 'DataArray_t') is not None \
       and PT.get_name(last_node) not in [":CGNS#Distribution", ":CGNS#GlobalNumbering", "GridCoordinates", "Periodic"]:
      path='/'.join(PT.get_name(node) for node in nodes[1:])
      container_paths.add(path)
  PT.scan(node, add_cnt_path, ancestors=True)
  fields_nodes = [PT.find_node_from_path(node, path) for path in container_paths]

  for fields_node in fields_nodes:
    is_full_vtx = PT.get_label(fields_node) in ['FlowSolution_t', 'DiscreteData_t'] and \
                  PT.Container.GridLocation(fields_node) == 'Vertex' and \
                  not PT.pred.IS_SUBSET(fields_node)
    data_names = [PT.get_name(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
    cartesian_vectors_basenames = py_utils.find_cartesian_vector_names(data_names, phy_dim)
    for basename in cartesian_vectors_basenames:
      if basename in constant_vectors:
        continue
      vectors_n = [PT.find_node_from_name_and_label(fields_node, f"{basename}{c}", 'DataArray_t')  for c in ['X', 'Y', 'Z'][:phy_dim]]
      if is_full_vtx:
        vectors = [PT.get_np_value(n)[vtx_mask] for n in vectors_n]
      else:
        vectors = [PT.get_np_value(n) for n in vectors_n]
      if basename in positional_vectors:
        tr_vectors = transform_func(*vectors, translation=translation_np, 
                                              rotation_center=rotation_center_np,
                                              rotation_angle=rotation_angle_np) #type:ignore[operator] #(signature of 2 funcs differs)
      else:
        tr_vectors = transform_func(*vectors, rotation_center=rotation_center_np,
                                              rotation_angle=rotation_angle_np) #type:ignore[operator]
      for vector_n, tr_vector in zip(vectors_n, tr_vectors):
        vector_val = PT.get_np_value(vector_n)
        if is_full_vtx:
          vector_val[vtx_mask] = tr_vector
        else:
          vector_val[:] = tr_vector


def transform_affine_zone(zone: CGNSTree,
                          vtx_mask: NDArray,
                          rotation_center: Iterable[float],
                          rotation_angle: Union[float, Iterable[float]],
                          translation: Iterable[float],
                          apply_to_fields: bool,
                          positional_fields: List[str] = ['Coordinate'],
                          constant_fields: List[str] = []) -> None:
  """
  Implementation of transform affine (see associated documentation) for
  a given zone.
  
  In addition, this function takes a bool array of shaped as coords array and
  apply the periodicity only to the vertices evaluating to True.
  """

  #Global information
  phy_dim = PT.Zone.PhysicalDimension(zone)
  transform_func = {2: np_utils.transform_cart_vectors_2d, 3: np_utils.transform_cart_vectors}[phy_dim]
  assert translation is not None
  assert rotation_angle is not None
  assert rotation_center is not None
  translation_np = np.asarray(translation)
  rotation_center_np = np.asarray(rotation_center)
  rotation_angle_np = np.asarray(rotation_angle) if phy_dim == 3 else rotation_angle

  # Transform coords
  for grid_co in PT.iter_children_from_label(zone, "GridCoordinates_t"):
    maybe_coords_n = [PT.get_child_from_name(grid_co, f"Coordinate{c}")  for c in ['X', 'Y', 'Z']]
    coords_n = [PT.find_child_from_name(grid_co, f"Coordinate{c}")  for c in ['X', 'Y', 'Z'][:phy_dim]]
    coords = [PT.get_np_value(n)[vtx_mask] for n in coords_n]

    tr_coords = transform_func(*coords, translation_np, rotation_center_np, rotation_angle_np) #type:ignore[operator] #(signature of 2 funcs differs)
    for coord_n, tr_coord in zip(coords_n, tr_coords):
      coord_value = PT.get_np_value(coord_n)
      coord_value[vtx_mask] = tr_coord

  # Transform GC/Periodic data
  # To update Periodic values of GCs, it is simpler to use homogeneous matrices
  # For a given GC, we have v_opp = M_gc * v_cur
  # and we apply to the whole mesh M_tr transformation v' = M_tr * v
  # We search M_gcnew such that v_opp' = M_gcnew * v_cur'
  # --> This leads to M_gcnew = M_tr * M_gc * (M_tr)^-1
  transf_mat = np_utils._transform_to_homogeneous_matrix(translation_np, rotation_center_np, rotation_angle_np)
  transf_mat_inv = np.linalg.inv(transf_mat)
  for gc in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', PT.pred.IS_GC]):
    if PT.GridConnectivity.isperiodic(gc):
      gc_center = PT.find_node_from_name(gc, 'RotationCenter')
      gc_angle  = PT.find_node_from_name(gc, 'RotationAngle')
      gc_trans  = PT.find_node_from_name(gc, 'Translation')
      
      gc_angle_value = PT.get_np_value(gc_angle)
      if phy_dim == 2: # 2D : angle may be in slot 0 or 1
        gc_angle_value = gc_angle_value[0] if gc_angle_value[0] != 0 else gc_angle_value[1]

      perio_mat  = np_utils._transform_to_homogeneous_matrix(gc_trans[1], gc_center[1], gc_angle_value)
      perio_mat_new = np.dot(transf_mat, np.dot(perio_mat, transf_mat_inv))
      gc_trans_new, gc_center_new, gc_angle_new = np_utils._homogeneous_matrix_to_transform(perio_mat_new)
      if phy_dim == 2:
        gc_angle_new = np.array([gc_angle_new, 0])
      PT.set_value(gc_center, gc_center_new)
      PT.set_value(gc_angle, gc_angle_new)
      PT.set_value(gc_trans, gc_trans_new)

  # Transform fields
  if apply_to_fields:
    update_fields(zone, vtx_mask, phy_dim, rotation_center_np, rotation_angle_np, translation_np, positional_fields, constant_fields)



def transform_affine(t: CGNSTree,
                     rotation_center: Optional[Iterable[float]] = None,
                     rotation_angle: Union[None, float, Iterable[float]] = None,
                     translation: Optional[Iterable[float]] = None,
                     apply_to_fields: bool = True,
                     positional_fields: List[str] = ['Coordinate']) -> None:
  """Apply the affine transformation to the coordinates of the given zone.

  Input zone(s) can be either structured or unstructured, but must have cartesian coordinates.
  Transformation is defined by

  .. math::
     \\tilde v = R \\cdot (v - c) + c + t

  where :math:`c, t` are the rotation center and translation vectors and :math:`R` is the rotation matrix.
  The rotation matrix is computed from ``rotation_angle``, whose kind depends on physical
  dimension of the mesh:

  - if ``phy_dim == 3``, it must be a vector of 3 floats, storing the
    `Euler rotation angles <https://en.wikipedia.org/wiki/Euler_angles>`_ 
    :math:`\\alpha, \\beta \\text{ and } \\gamma`; :math:`R` is then the combination of 
    
    - intrinsic elemental rotations :math:`X_\\alpha, Y^{\\prime}_\\beta, Z^{\\prime\\prime}_\\gamma`, 
      or, equivalently, 
    - extrinsic elemental rotations :math:`Z_\\gamma, Y_\\beta, X_\\alpha`. 

  - if ``phy_dim == 2``, a scalar float :math:`\\theta` is expected, defining the rotation angle in the XY plane.

  Input tree is modified inplace.

  Args:
    t    (CGNSTree): Tree starting at Zone_t level or higher.
    rotation_center (array): Center coordinates of the rotation
    rotation_angle (array): Angles of the rotation
    translation (array):  Translation vector components
    apply_to_fields (bool, optional) : 
        If ``True``, apply the rotation part of the transformation to all the vectorial fields (DataArray_t)
        found in the input tree. Defaults to ``True``.
    positional_fields (list of str, optional): 
        If ``apply_to_fields`` is ``True``, add the translation part for these specific vectorial fields.
        Defaults to ``['Coordinate']``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #transform_affine@start
        :end-before: #transform_affine@end
        :dedent: 2
  """
  for zone in PT.iter_all_Zone_t(t):
    any_gc_n = PT.find_child_from_label(zone, 'GridCoordinates_t')
    cart_names = ['CoordinateX', 'CoordinateY' ,'CoordinateZ']
    phy_dim = len(PT.get_children_from_predicate(any_gc_n, PT.pred.name_in(cart_names)))
    assert phy_dim in [2,3]
    if rotation_center is None:
      rotation_center = [0.] * phy_dim
    if rotation_angle is None:
      rotation_angle = [0.] * phy_dim if phy_dim == 3 else 0.
    if translation is None:
      translation = [0.] * phy_dim
    # Don't use PT.Zone.VertexSize because it won't work on dist_tree
    any_coord = PT.find_child_from_predicate(any_gc_n, PT.pred.name_in(cart_names))
    vtx_mask = np.ones(PT.get_np_value(any_coord).shape, bool)
    transform_affine_zone(zone, vtx_mask, rotation_center, rotation_angle, translation, apply_to_fields, positional_fields)

  # Deal vectors that are outside zones: family, UserDefined, Convergence history, ...
  if apply_to_fields:
    for base in PT.iter_all_CGNSBase_t(t):
      phy_dim = PT.Base.PhysicalDimension(base)
      assert phy_dim in [2,3]
      if rotation_center is None:
        rotation_center = [0.] * phy_dim
      if rotation_angle is None:
        rotation_angle = [0.] * phy_dim if phy_dim == 3 else 0.
      if translation is None:
        translation = [0.] * phy_dim
      for child in PT.iter_children_from_predicate(base, ~PT.pred.label_is('Zone_t')):
        update_fields(child, None, phy_dim, np.asarray(rotation_center), np.asarray(rotation_angle),
                              np.asarray(translation), positional_fields, list())

def scale_mesh(t: CGNSTree, s: Union[float, Sequence[float]] = 1.) -> None:
  """Rescale the GridCoordinates of the input mesh.

  Input zone(s) can be either structured or unstructured, but must have cartesian coordinates.
  Transformation is defined by

  .. math::
     \\tilde v = S \\cdot v

  where S is the scaling matrix.
  Input tree is modified inplace.

  Args:
    t    (CGNSTree): Tree starting at Zone_t level or higher
    s (float or array of float): Scaling factor in each physical dimension. Scalars automatically
      extend to uniform array.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: scale_mesh@start
        :end-before: #scale_mesh@end
        :dedent: 2
  """
  scaling = 3 * [s] if isinstance(s, (int, float)) else s 
  fields_found = False
  is_container = PT.pred.label_in(['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t'])
  for zone in PT.iter_all_Zone_t(t):
    for grid_co in PT.get_children_from_label(zone, 'GridCoordinates_t'):
      for idir, dir in enumerate(['X', 'Y', 'Z']):
        node = PT.get_child_from_name(grid_co, f'Coordinate{dir}')
        if node is not None:
          node_val = PT.get_np_value(node)
          node_val *= scaling[idir]

    if PT.get_child_from_predicate(zone, is_container) is not None or \
       PT.get_child_from_predicates(zone, 'ZoneBC_t/BC_t/BCDataSet_t/BCData_t') is not None:
      fields_found = True
  
  if fields_found:
    mlog.warning(f"Scaling mesh does not affect fields, and some are present in tree. Update their value if needed.")



# Belows are helper functions to compute entity theta coordinate, depending of GridLocation
def _compute_cellcenter_theta(z: CGNSTree, comm: Optional[MPIComm]) -> NDArray:
  theta = _compute_elements_center(z, 3, comm)[1::3]
  if PT.Zone.Type(z) == 'Structured' and MT.get_Distribution(z) is None:
    theta = theta.reshape(PT.Zone.CellSize(z), order='F')
  return theta

COMPUTE_THETA:Dict[str, Callable[[CGNSTree, Optional[MPIComm]], NDArray]]
COMPUTE_THETA = {'CellCenter'  : _compute_cellcenter_theta,
                 'FaceCenter'  : lambda z,comm : _compute_elements_center(z,2,comm)[1::3], #Only partial subsets -> no reshape needed
                 'Vertex'      : lambda z,c : PT.get_node_from_predicates(z, 'GridCoordinates_t/CoordinateTheta')[1] #type:ignore #(request not yet av)
                 }

def shrink_to_subset(array, zone, subset, comm):
  """
  Extract a subpart of a full array (eg defined on all Vertex) on a specific 
  patch (U/PointList or S/PointRange). Array / subset can be distributed or partitioned.
  """
  pl = PT.get_child_from_name(subset, 'PointList')
  pr = PT.get_child_from_name(subset, 'PointRange')
  if pl is None and pr is None: # Subset is not partial
    return array
  loc = PT.Subset.GridLocation(subset)
  is_partitioned = MT.get_Distribution(zone) is None
  subset_distri = None if is_partitioned else MT.distribution_value(subset, 'Index')
  if PT.Zone.Type(zone) == 'Unstructured':
    if pl is None:
      assert pr is not None
      pl = np_utils.single_dim_pr_to_pl(PT.get_np_value(pr), subset_distri)
    if loc == 'Vertex':
      shift = 1
    else:
      _to_index = {'CellCenter' : 2 ,'EdgeCenter' : 1} if PT.Zone.CellDimension(zone) == 2 else {'CellCenter' : 3 , 'FaceCenter' :2, 'EdgeCenter' : 1}
      range_per_dim = PT.Zone.get_elt_range_per_dim(zone)
      if PT.Zone.CellDimension(zone) == 3 and PT.Zone.has_ngon_elements(zone) and not PT.Zone.has_nface_elements(zone):
        range_per_dim[3][0] = range_per_dim[2][1] + 1 # Implicit nface 
        range_per_dim[3][1] = range_per_dim[2][1] + PT.Zone.n_cell(zone)
      shift = range_per_dim[_to_index[loc]][0] 
    if is_partitioned:
      return array[pl[1][0]-shift] 
    else:
      distri = par_utils.dn_to_distribution(array.size, comm)
      return EP.block_to_part(array, distri, pl[1][0]-shift, comm)
  else: # Structured zones
    assert pl is None, "PointList are not managed for unstructured meshes"
    if PT.get_label(subset) in ["FlowSolution_t", "DiscreteData_t"]:
      raise NotImplementedError(f"Partial containers are not supported for structured {PT.get_label(subset)}")
    
    # We use compute_pointList_from_pointRanges to expand indices corresponding 
    # to the input PointRange. In partitioned case, create a fake "full" distribution
    bc_size = np.abs(pr[1][:,1] - pr[1][:,0]) + 1
    bc_range = subset_distri if not is_partitioned else np.array([0, bc_size.prod(), bc_size.prod()])

    bc_slabs = HFR2S.compute_slabs(bc_size, bc_range)

    sub_pr_list = [np.asarray(slab) for slab in bc_slabs]
    for sub_pr in sub_pr_list:
      sub_pr[:,0] += pr[1][:,0]
      sub_pr[:,1] += pr[1][:,0] - 1
    idx = pr_utils.compute_pointList_from_pointRanges(sub_pr_list, 
                                                      PT.Zone.VertexSize(zone), 
                                                      PT.Subset.GridLocation(subset))[0]

    if is_partitioned:
      # If loc == CellCenter or Vertex, array is shaped -> flatten it
      return array.reshape(-1, order='F')[idx-1]
    else:
      distri = par_utils.dn_to_distribution(array.size, comm)
      return EP.block_to_part(array, distri, idx-1, comm)
      
def cartesian_to_cylindrical_from_unit_revolution_axis(t: CGNSTree,
                                                       revolution_axis: Sequence[int],
                                                       comm: Optional[MPIComm],
                                                       apply_to_fields: bool,
                                                       abs_tol:float=1.e-8) -> None:
  """ Implementation of cartesian_to_cylindrical for a unit revolution axis.

  Transformation is defined by

  .. math::
     \\ r = np.sqrt(a² + b²)
     \\ theta = np.arctan(b/a)
     \\ z = c

  where a, b, c are coordinates on a plan depending of input revolution axis
  """

  np_revolution_axis = np.asarray(revolution_axis)
  if np.array_equal(np_revolution_axis, [1, 0, 0]):
    idx_order = [1,2,0]
    axis_idx = 0
  elif np.array_equal(np_revolution_axis, [0, 1, 0]):  
    idx_order = [0,2,1]
    axis_idx = 1
  elif np.array_equal(np_revolution_axis, [0, 0, 1]):
    idx_order = [0,1,2]
    axis_idx = 2
  else:
    raise AssertionError("Revolution axis is not unitary")
  cyl_suffix = ['R', 'Theta', 'Z']
  
  non_axis_idx=[0,1,2]
  non_axis_idx.pop(axis_idx)

  for zone in PT.iter_all_Zone_t(t):

    transform_matrix_n = PT.get_child_from_predicates(zone, 'GridCoordinates_t/CoordinateTransform')
    coords_suffix = ['Xi', 'Eta', 'Zeta'] if transform_matrix_n is not None else ['X', 'Y', 'Z']

    predicates = ['GridCoordinates_t'] # Always treat coordinates, + fields if apply_to_fields
    if apply_to_fields:
      predicates += ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t']

    loc_to_theta:Dict[str, Optional[NDArray]]  = {key: None for key in COMPUTE_THETA.keys()}

    for predicate in predicates:
      for container in PT.get_children_from_predicates(zone, predicate):
        datapaths = PT.Container.fields(container).keys()
        vectors_basepaths = py_utils.find_vector_names(datapaths, coords_suffix)
        if PT.get_label(container) != "GridCoordinates_t" and len(vectors_basepaths) > 0:
          loc_container = PT.Container.GridLocation(container, zone)
          loc_container = 'FaceCenter' if loc_container.endswith("FaceCenter") else loc_container #Remove I,J,K prefix
          if loc_to_theta[loc_container] is None:
            loc_to_theta[loc_container] = COMPUTE_THETA[loc_container](zone, comm)
          theta = loc_to_theta[loc_container]
          if PT.Container._is_partial(container):
            theta = shrink_to_subset(theta, zone, PT.Container.SubsetNode(container, zone), comm)
        for basepath in vectors_basepaths:
          basename = basepath.split('/')[-1]
          fields_n = [PT.find_node_from_path(container, f'{basepath}{suffix}') for suffix in coords_suffix]
          ordered_fields = [fields_n[i] for i in idx_order]
          if basename == "Coordinate":
            cyl_values = _to_rthetaz(*[PT.get_np_value(n) for n in ordered_fields])
          else:
            cyl_values = _to_rthetaz_vectors(*[PT.get_np_value(n) for n in ordered_fields], theta) #type:ignore #(unpacking is not correctly infered)
          for i, val in enumerate(cyl_values):
            PT.update_node(ordered_fields[i], f'{basename}{cyl_suffix[i]}', value=val)

    for gc in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', PT.pred.IS_GC]):
      if PT.GridConnectivity.isperiodic(gc):
        gc_angle  = PT.find_node_from_name(gc, 'RotationAngle')
        gc_trans  = PT.find_node_from_name(gc, 'Translation')
        gc_angle_value = PT.get_np_value(gc_angle)
        gc_trans_value = PT.get_np_value(gc_trans)
        
        # Only allowed transformations are managed
        # > only rotation around axis in cartesian system
        # > only translation around axis in cartesian system
        if not (np.abs(gc_angle_value[non_axis_idx]) < abs_tol).all():
          raise AssertionError(f"Rotation axis of periodic interface {PT.get_name(gc)} is not aligned with revolution axis")
        if not (np.abs(gc_trans_value[non_axis_idx]) < abs_tol).all():
          raise AssertionError(f"Translation axis of periodic interface {PT.get_name(gc)} is not aligned with revolution axis")
        
        gc_angle_new = np.zeros_like(gc_angle_value)
        gc_trans_new = np.zeros_like(gc_trans_value)
        # Rotation around axis (cart) becomes translation in theta (cyl)
        # Translation in axis (cart) becomes translation in Z (cyl)
        gc_trans_new[1] = gc_angle_value[axis_idx]
        gc_trans_new[2] = gc_trans_value[axis_idx]
        PT.set_value(gc_angle, gc_angle_new)
        PT.set_value(gc_trans, gc_trans_new)
     
def cylindrical_to_cartesian_from_unit_revolution_axis(t: CGNSTree,
                                                       revolution_axis: Sequence[float],
                                                       comm: Optional[MPIComm],
                                                       apply_to_fields: bool,
                                                       abs_tol:float=1.e-8) -> None:
  """Compute the cartesian coordinates from a unit revolution axis.

  Transformation is defined by

  .. math::
     \\ a = r*cos(theta)
     \\ b = r*sin(theta)
     \\ c = z

  where r, theta are respectively the radius and the angle and
  a, b, c are coordinates on a plan depending of input revolution axis
  """
  np_revolution_axis = np.asarray(revolution_axis)
  if np.array_equal(np_revolution_axis,[1, 0, 0]):
    idx_order = [2,0,1]
    axis_idx = 0
  elif np.array_equal(np_revolution_axis, [0, 1, 0]):
    idx_order = [0,2,1]
    axis_idx = 1
  elif np.array_equal(np_revolution_axis, [0, 0, 1]):
    idx_order = [0,1,2]
    axis_idx = 2
  else:
    raise AssertionError("Revolution axis is not unitary")
  
  non_axis_idx=[0,1,2]
  non_axis_idx.pop(axis_idx)

  for zone in PT.iter_all_Zone_t(t):

    transform_matrix_n = PT.get_child_from_predicates(zone, 'GridCoordinates_t/CoordinateTransform')
    coords_suffix = ['Xi', 'Eta', 'Zeta'] if transform_matrix_n is not None else ['X', 'Y', 'Z']

    if apply_to_fields:
      predicates = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t']
    else:
      predicates = []
    predicates += ['GridCoordinates_t'] # Always treat coordinates (last because needed for centers)

    loc_to_theta:Dict[str, Optional[NDArray]]  = {key: None for key in COMPUTE_THETA.keys()}
  
    for predicate in predicates:
      for container in PT.get_children_from_predicates(zone, predicate):
        datapaths = PT.Container.fields(container).keys()
        cylindric_vectors_basepaths = py_utils.find_vector_names(datapaths, ['R', 'Theta', 'Z'])
        if PT.get_label(container) != "GridCoordinates_t" and len(cylindric_vectors_basepaths) > 0:
          loc_container = PT.Container.GridLocation(container, zone)
          loc_container = 'FaceCenter' if loc_container.endswith("FaceCenter") else loc_container #Remove I,J,K prefix
          if loc_to_theta[loc_container] is None:
            loc_to_theta[loc_container] = COMPUTE_THETA[loc_container](zone, comm)
          theta = loc_to_theta[loc_container]
          if PT.Container._is_partial(container):
            theta = shrink_to_subset(theta, zone, PT.Container.SubsetNode(container, zone), comm)
        for basepath in cylindric_vectors_basepaths:
          basename = basepath.split('/')[-1]
          fields_n = [PT.find_node_from_path(container, f'{basepath}{suffix}') for suffix in ['R', 'Theta', 'Z']]
          if basename == "Coordinate":
            cart_values = _to_xyz(*[PT.get_value(n) for n in fields_n])
          else:
            cart_values = _to_xyz_vectors(*[PT.get_value(n) for n in fields_n], theta) #type:ignore #(unpacking badly infered)

          for i, idx in enumerate(idx_order):
            PT.update_node(fields_n[idx], f'{basename}{coords_suffix[i]}', value=cart_values[idx])

    for gc in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', PT.pred.IS_GC]):
      if PT.GridConnectivity.isperiodic(gc):
        gc_angle  = PT.find_node_from_name(gc, 'RotationAngle')
        gc_trans  = PT.find_node_from_name(gc, 'Translation')
        gc_angle_value = PT.get_np_value(gc_angle)
        gc_trans_value = PT.get_np_value(gc_trans)
        
        # Only allowed transformations are managed
        # > no periodic by rotation in cylindrical system
        # > only translation on theta or z in cylindrical system
        if not np.allclose(gc_angle_value, [0., 0., 0.], atol=abs_tol):
          raise AssertionError(f"Rotation of periodic interface {PT.get_name(gc)} is not empty")
        if not abs(gc_trans_value[0]) < abs_tol:
          raise AssertionError(f"Translation axis of periodic interface {PT.get_name(gc)} is not orthogonal to er vector")
        
        # Translation in theta (cyl) becomes rotation around axis (cart)
        gc_angle_new = np.zeros_like(gc_angle_value)
        gc_trans_new = np.zeros_like(gc_trans_value)
        # Translation in Z (cyl) becomes translation in axis (cart)
        gc_angle_new[axis_idx] = gc_trans_value[1]
        gc_trans_new[axis_idx] = gc_trans_value[2]
        PT.set_value(gc_angle, gc_angle_new)
        PT.set_value(gc_trans, gc_trans_new)

def auxiliary_coords_system(t: CGNSTree,
                            transition_matrix: Optional[NDArray],
                            apply_to_fields: bool = True) -> None: 
  """Convert the input tree from or to an auxiliary coordinate system.

  Input zone(s) in the tree can be either structured or unstructured, and can have cartesian or 
  auxiliary coordinates system. In the later case, suffixes ``Xi``, ``Eta`` and ``Zeta``
  are used for coordinates and vectorial fields. In addition, the ``CoordinateTransform`` node
  must be used to record the transition matrix.

  Depending of the type of ``transition_matrix``, this function can be used to:

  - go to (or stay in) auxiliary coordinates (using a matrix of size 3x3);
  - go back to cartesian coordinates (using None). The operation is done by inverting the 
    transition matrix stored in the CoordinateTransform node.

  Args:
    t    (CGNSTree): Tree starting at Zone_t level or higher
    transition_matrix (array or None) : 3x3 array of floats or None (see above)
    apply_to_fields (bool) : If True, apply the transformation to the vectorial fields found under
      the following nodes : ``FlowSolution_t``, ``DiscreteData_t``, ``ZoneSubRegion_t``, ``BCDataset_t``.
      Defaults to ``True``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #change_basis@start
        :end-before: #change_basis@end
        :dedent: 2
  """

  for zone in PT.iter_all_Zone_t(t):

    # Assert that CoordinateTransform is the same for all GridCoordinates_t nodes
    coord_transform_n = PT.get_child_from_predicates(zone, 'GridCoordinates_t/CoordinateTransform')
    is_aux = coord_transform_n is not None
    reverse = transition_matrix is None

    if reverse:
      assert coord_transform_n is not None
      transition_matrix = PT.get_np_value(coord_transform_n)
      transition_matrix = np.linalg.inv(transition_matrix)

    assert transition_matrix is not None
    in_suffix = ['Xi', 'Eta', 'Zeta'] if is_aux else ['X', 'Y', 'Z']
    out_suffix = ['X', 'Y', 'Z'] if reverse else ['Xi', 'Eta', 'Zeta']

    if reverse:
      PT.rm_nodes_from_name(zone, 'CoordinateTransform', depth=2)
    else:
      new_transform_matrix = transition_matrix if coord_transform_n is None else np.dot(transition_matrix, PT.get_np_value(coord_transform_n))
      for gc_n in PT.get_children_from_predicate(zone, 'GridCoordinates_t'):
        PT.update_child(gc_n, 'CoordinateTransform', 'DataArray_t', new_transform_matrix)
    
    predicates = ['GridCoordinates_t'] # Always treat coordinates
    if apply_to_fields:
      predicates.extend(['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t/BCData_t'])

    for predicate in predicates:
      for fields_node in PT.get_children_from_predicates(zone, predicate):
        datanames = [PT.get_name(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
        vectors_basenames = py_utils.find_vector_names(datanames, in_suffix)
        for basename in vectors_basenames:
          vectors_n = [PT.find_node_from_name(fields_node, f"{basename}{c}")  for c in in_suffix]
          tr_fields = np_utils.matmul_cart_vectors([PT.get_np_value(n) for n in vectors_n], transition_matrix)
          for node, s, new_val in zip(vectors_n, out_suffix, tr_fields):
            PT.update_node(node, f'{basename}{s}', value=new_val)
            
    # Transform GC/Periodic data
    # To update Periodic values of GCs, it is simpler to use homogeneous matrices
    # For a given GC, we have v_opp = M_gc * v_cur
    # and we apply to the whole mesh M_tr transformation v' = M_tr * v
    # We search M_gcnew such that v_opp' = M_gcnew * v_cur'
    # --> This leads to M_gcnew = M_tr * M_gc * (M_tr)^-1
    transf_mat           = np.zeros((4,4))
    transf_mat[0:3, 0:3] = transition_matrix
    transf_mat[3,3]      = 1
    transf_mat_inv       = np.linalg.inv(transf_mat)
    
    coords_n = PT.Zone.coordinates(zone)
    phy_dim = 2 if coords_n[2] is None else 3
    
    for gc in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', PT.pred.IS_GC]):
      if PT.GridConnectivity.isperiodic(gc):
        gc_center = PT.find_node_from_name(gc, 'RotationCenter')
        gc_angle  = PT.find_node_from_name(gc, 'RotationAngle')
        gc_trans  = PT.find_node_from_name(gc, 'Translation')
      
        gc_angle_value = PT.get_np_value(gc_angle)
        if phy_dim == 2: # 2D : angle may be in slot 0 or 1
          gc_angle_value = gc_angle_value[0] if gc_angle_value[0] != 0 else gc_angle_value[1]
  
        perio_mat  = np_utils._transform_to_homogeneous_matrix(gc_trans[1], gc_center[1], gc_angle_value)
        perio_mat_new = np.dot(transf_mat, np.dot(perio_mat, transf_mat_inv))
        gc_trans_new, gc_center_new, gc_angle_new = np_utils._homogeneous_matrix_to_transform(perio_mat_new)
        if phy_dim == 2:
          gc_angle_new = np.array([gc_angle_new, 0])
        PT.set_value(gc_center, gc_center_new)
        PT.set_value(gc_angle, gc_angle_new)
        PT.set_value(gc_trans, gc_trans_new)
    

def cartesian_to_cylindrical(t: CGNSTree,
                             axis: Sequence[float],
                             comm: Optional[MPIComm] = None,
                             apply_to_fields: bool = True) -> None:
  """Convert the input tree into a cylindrical coordinate system.

  Input zone(s) in the tree can be either structured or unstructured, but must have cartesian coordinates.
  The revolution axis to be used for the cylindrical coordinate system must be specified thought the ``axis`` argument,
  and will always be denoted ``Z`` in the cylindrical system.

  Input tree is modified inplace; suffixes ``R``, ``Theta`` and ``Z`` are used for coordinates
  and vectorial fields (note that cyl. Z axis and cart. Z axis may differs).

  Args:
    t    (CGNSTree): Tree starting at Zone_t level or higher
    axis (array of 3 floats) : Revolution axis, which can by any non zero vector
    comm       (MPIComm) : MPI communicator, mandatory only for distributed trees
    apply_to_fields (bool) : If True, apply the transformation to the vectorial fields found under
      the following nodes : ``FlowSolution_t``, ``DiscreteData_t``, ``ZoneSubRegion_t``, ``BCDataset_t``.
      Defaults to ``True``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #cartesian_to_cylindrical@start
        :end-before: #cartesian_to_cylindrical@end
        :dedent: 2
  """

  np_axis = np.asarray(axis)

  if np.count_nonzero(np_axis) != 1:
    transform_matrix = np_utils.create_transform_matrix(np_axis) #type:ignore[arg-type] #(ndarray is compliant)
    auxiliary_coords_system(t, transform_matrix, apply_to_fields)
    np_axis = np.dot(transform_matrix, np_axis)
 
  revolution_axis_unit = np_axis / np.linalg.norm(np_axis)
  cartesian_to_cylindrical_from_unit_revolution_axis(t, revolution_axis_unit, comm, apply_to_fields)

def cylindrical_to_cartesian(t: CGNSTree,
                             axis: Sequence[float],
                             comm: Optional[MPIComm] = None,
                             apply_to_fields: bool = True) -> None:
  """Convert the input tree into a cartesian coordinate system.

  Input zone(s) in the tree can be either structured or unstructured, but must have cylindrical coordinates.
  The expression of the revolution axis (*ie* the ``Z`` axis) of the cylindrical coordinate system in the cartesian basis must be
  specified thought the ``axis`` argument.

  Input tree is modified inplace; suffixes ``R``, ``Theta`` and ``Z`` must be used for coordinates
  and vectorial fields (note that cyl. Z axis and cart. Z axis may differs).

  Args:
    t    (CGNSTree): Tree starting at Zone_t level or higher
    axis (array of 3 floats) : Revolution axis, which can by any non zero vector
    comm       (MPIComm) : MPI communicator, mandatory only for distributed trees
    apply_to_fields (bool) : If True, apply the transformation to the vectorial fields found under
      the following nodes : ``FlowSolution_t``, ``DiscreteData_t``, ``ZoneSubRegion_t``, ``BCDataset_t``.
      Defaults to ``True``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #cylindrical_to_cartesian@start
        :end-before: #cylindrical_to_cartesian@end
        :dedent: 2
  """
  
  np_axis = np.asarray(axis)
  need_change_basis = np.count_nonzero(np_axis) != 1

  if need_change_basis:
    transform_matrix_n = PT.get_child_from_predicates(t, 'CGNSBase_t/Zone_t/GridCoordinates_t/CoordinateTransform')
    assert transform_matrix_n is not None
    transform_matrix = PT.get_np_value(transform_matrix_n)
    np_axis = np.dot(transform_matrix, np_axis)

  revolution_axis_unit = np_axis / np.linalg.norm(np_axis)
  cylindrical_to_cartesian_from_unit_revolution_axis(t, revolution_axis_unit, comm, apply_to_fields)

  if need_change_basis:
    auxiliary_coords_system(t, None, apply_to_fields)
