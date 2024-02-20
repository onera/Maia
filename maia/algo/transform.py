import numpy as np

import maia.pytree as PT
from maia.utils import py_utils, np_utils
from maia.algo.apply_function_to_nodes import zones_iterator

from maia.utils import logging as mlog

def _to_xyz(r, theta, z):
  return r*np.cos(theta), r*np.sin(theta), z
def _to_rthetaz(x, y, z):
  return np.sqrt(x**2+y**2), np.arctan2(y, x), z

def transform_affine_zone(zone,
                          vtx_mask,
                          rotation_center,
                          rotation_angle,
                          translation,
                          apply_to_fields):
  """
  Implementation of transform affine (see associated documentation) for
  a given zone.
  
  In addition, this function takes a bool array of shaped as coords array and
  apply the periodicity only to the vertices evaluating to True.
  """
  # Transform coords
  for grid_co in PT.iter_children_from_label(zone, "GridCoordinates_t"):
    coords_n = [PT.get_child_from_name(grid_co, f"Coordinate{c}")  for c in ['X', 'Y', 'Z']]
    phy_dim = 2 if coords_n[2] is None else 3
    coords_n = coords_n[:phy_dim]
    coords = [PT.get_value(n)[vtx_mask] for n in coords_n]
  
    if phy_dim == 3:
      tr_coords = np_utils.transform_cart_vectors(*coords, translation, rotation_center, rotation_angle)
    else:
      tr_coords = np_utils.transform_cart_vectors_2d(*coords, translation, rotation_center, rotation_angle)
    for coord_n, tr_coord in zip(coords_n, tr_coords):
      coord_n[1][vtx_mask] = tr_coord

  # Transform fields
  if apply_to_fields:
    fields_nodes  = PT.get_children_from_label(zone, "FlowSolution_t")
    fields_nodes += PT.get_children_from_label(zone, "DiscreteData_t")
    fields_nodes += PT.get_children_from_label(zone, "ZoneSubRegion_t")
    for bc in PT.iter_children_from_predicates(zone, "ZoneBC_t/BC_t"):
      fields_nodes += PT.get_children_from_label(bc, "BCDataSet_t")
    for fields_node in fields_nodes:
      is_full_vtx = PT.Subset.GridLocation(fields_node) == 'Vertex' and \
                    PT.get_label(fields_node) in ['FlowSolution_t', 'DiscreteData_t'] and \
                    PT.get_child_from_name(fields_node, 'PointList') is None and \
                    PT.get_child_from_name(fields_node, 'PointRange') is None
      data_names = [PT.get_name(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
      cartesian_vectors_basenames = py_utils.find_cartesian_vector_names(data_names, phy_dim)
      for basename in cartesian_vectors_basenames:
        vectors_n = [PT.get_node_from_name_and_label(fields_node, f"{basename}{c}", 'DataArray_t')  for c in ['X', 'Y', 'Z'][:phy_dim]]
        if is_full_vtx:
          vectors = [PT.get_value(n)[vtx_mask] for n in vectors_n]
        else:
          vectors = [PT.get_value(n) for n in vectors_n]
        # Assume that vectors are position independant
        # Be careful, if coordinates vector needs to be transform, the translation is not applied !
        if phy_dim == 3:
          tr_vectors = np_utils.transform_cart_vectors(*vectors, rotation_center=rotation_center, rotation_angle=rotation_angle)
        else:
          tr_vectors = np_utils.transform_cart_vectors_2d(*vectors, rotation_center=rotation_center, rotation_angle=rotation_angle)
        for vector_n, tr_vector in zip(vectors_n, tr_vectors):
          if is_full_vtx:
            vector_n[1][vtx_mask] = tr_vector
          else:
            vector_n[1] = tr_vector



def transform_affine(t,
                     rotation_center = np.zeros(3),
                     rotation_angle  = np.zeros(3),
                     translation     = np.zeros(3),
                     apply_to_fields = False):
  """Apply the affine transformation to the coordinates of the given zone.

  Input zone(s) can be either structured or unstructured, but must have cartesian coordinates.
  Transformation is defined by

  .. math::
     \\tilde v = R \\cdot (v - c) + c + t

  where c, t are the rotation center and translation vectors and R is the rotation matrix.
  Note that when the physical dimension of the mesh is set to 2, rotation_angle must
  be a scalar float.

  Input tree is modified inplace.

  Args:
    t    (CGNSTree(s)): Tree (or sequences of) starting at Zone_t level or higher.
    rotation_center (array): center coordinates of the rotation
    rotation_angler (array): angles of the rotation
    translation (array):  translation vector components
    apply_to_fields (bool, optional) : 
        if True, apply the rotation vector to the vectorial fields found under 
        following nodes : ``FlowSolution_t``, ``DiscreteData_t``, ``ZoneSubRegion_t``, ``BCDataset_t``.
        Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #transform_affine@start
        :end-before: #transform_affine@end
        :dedent: 2
  """
  for zone in zones_iterator(t):
    any_coord = PT.get_node_from_predicates(zone, 'GridCoordinates_t/DataArray_t')
    # Don't use PT.Zone.VertexSize because it won't work on dist_tree
    vtx_mask = np.ones(PT.get_value(any_coord).shape, bool)
    transform_affine_zone(zone, vtx_mask, rotation_center, rotation_angle, translation, apply_to_fields)

def scale_mesh(t, s=1.):
  """Rescale the GridCoordinates of the input mesh.

  Input zone(s) can be either structured or unstructured, but must have cartesian coordinates.
  Transformation is defined by

  .. math::
     \\tilde v = S \\cdot v

  where S is the scaling matrix.
  Input tree is modified inplace.

  Args:
    t    (CGNSTree(s)): Tree (or sequences of) starting at Zone_t level or higher
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
  is_container = lambda n: PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t']
  for zone in zones_iterator(t):
    for grid_co in PT.get_children_from_label(zone, 'GridCoordinates_t'):
      for idir, dir in enumerate(['X', 'Y', 'Z']):
        node = PT.get_child_from_name(grid_co, f'Coordinate{dir}')
        if node is not None:
          node[1] *= scaling[idir]

    if PT.get_child_from_predicate(zone, is_container) is not None or \
       PT.get_child_from_predicates(zone, 'ZoneBC_t/BC_t/BCDataSet_t/BCData_t') is not None:
      fields_found = True
  
  if fields_found:
    mlog.warning(f"Scaling mesh does not affect fields, and some are present in tree. Update their value if needed.")



def cartesian_to_cylindric_from_unit_revolution_axis(t, revolution_axis=(0, 0, 1), gc_name='GridCoordinates', apply_to_fields=True):
  """Compute cylindric coordinates from a unit revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cartesian coordinates.

  Transformation is defined by

  .. math::
     \\ r = np.sqrt(x² + y²)
     \\ theta = np.arctan(y/x)
     \\ z = z

  where x, y are coordinates on (x,y) plan.

  Args:
    t (Tree) : Recover GridCoordinates from the zones in the tree
               Tree can be a distributed or partitioned tree.
    revolution_axis (tuple, list, array) : Constant axis
                                           By default it set on z-axis.
    gc_name (str) : Name of the GridCoordinates to transform into cylindric coordinates and containing the transformation matrix
                    By default it searches the GridCoordinates node
    apply_to_fields (bool) : Apply the transformation to fields 
                             By default it set on True
  """

  revolution_axis = np.asarray(revolution_axis)
  if np.array_equal(revolution_axis, [1, 0, 0]):
    idx_order = [1,2,0]
  elif np.array_equal(revolution_axis, [0, 1, 0]):  
    idx_order = [0,2,1]
  elif np.array_equal(revolution_axis, [0, 0, 1]):
    idx_order = [0,1,2]
  else:
    raise AssertionError("Revolution axis is not unitary")
  cyl_suffix = ['R', 'Theta', 'Z']

  for zone in PT.get_all_Zone_t(t):

    if PT.get_node_from_name(zone, gc_name) is None:
      continue

    transform_matrix_n = PT.get_child_from_predicates(zone, f'{gc_name}/CoordinateTransform')
    coords_suffix = ['Xi', 'Eta', 'Zeta'] if transform_matrix_n is not None else ['X', 'Y', 'Z']

    predicates = [gc_name] # Always treat coordinates, + fields if apply_to_fields
    if apply_to_fields:
      predicates += ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t']

    for predicate in predicates:
      for container in PT.get_children_from_predicates(zone, predicate):
        datanames = [PT.get_name(data) for data in PT.iter_nodes_from_label(container, "DataArray_t")]
        vectors_basenames = py_utils.find_vector_names(datanames, coords_suffix)
        for basename in vectors_basenames:
          
          fields_n = [PT.get_child_from_name(container, f'{basename}{suffix}') for suffix in coords_suffix]
          ordered_fields = [fields_n[i] for i in idx_order]

          cyl_values = _to_rthetaz(*[PT.get_value(n) for n in ordered_fields])
          for i, val in enumerate(cyl_values):
            PT.update_node(ordered_fields[i], f'{basename}{cyl_suffix[i]}', value=val)
     
def cylindric_to_cartesian_from_unit_revolution_axis(t, revolution_axis=(0, 0, 1), gc_name='GridCoordinates', apply_to_fields=True):
  """Compute the cartesian coordinates from a unit revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cylindric coordinates.

  Transformation is defined by

  .. math::
     \\ x = r*cos(theta)
     \\ y = r*sin(theta)
     \\ z = z

  where r, theta are respectively the radius and the angle.

  Args:
    t (Tree) : Recover GridCoordinates from the zones in the tree
               Tree can be a distributed or partitioned tree.
    revolution_axis (tuple, list, array) : Constant axis
                                           By default it set on z-axis.
    gc_name (str) : Name of the GridCoordinates to transform into cartesian coordinates and containing the transformation matrix
                    By default it searches the GridCoordinates node
    apply_to_fields (bool) : Apply the transformation to fields
                             By default it set on True               
  """
  revolution_axis = np.asarray(revolution_axis)
  if np.array_equal(revolution_axis,[1, 0, 0]):
    idx_order = [2,0,1]
  elif np.array_equal(revolution_axis, [0, 1, 0]):
    idx_order = [0,2,1]
  elif np.array_equal(revolution_axis, [0, 0, 1]):
    idx_order = [0,1,2]
  else:
    raise AssertionError("Revolution axis is not unitary")

  for zone in PT.get_all_Zone_t(t):

    if PT.get_node_from_name(zone, gc_name) is None:
      continue

    transform_matrix_n = PT.get_child_from_predicates(zone, f'{gc_name}/CoordinateTransform')
    coords_suffix = ['Xi', 'Eta', 'Zeta'] if transform_matrix_n is not None else ['X', 'Y', 'Z']

    predicates = [gc_name] # Always treat coordinates, + fields if apply_to_fields
    if apply_to_fields:
      predicates += ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t']

    for predicate in predicates:
      for container in PT.get_children_from_predicates(zone, predicate):
        datanames = [PT.get_name(data) for data in PT.iter_nodes_from_label(container, "DataArray_t")]
        cylindric_vectors_basenames = py_utils.find_vector_names(datanames, ['R', 'Theta', 'Z'])
        for basename in cylindric_vectors_basenames:

          fields_n = [PT.get_child_from_name(container, f'{basename}{suffix}') for suffix in ['R', 'Theta', 'Z']]
          cart_values = _to_xyz(*[PT.get_value(n) for n in fields_n])

          for i, idx in enumerate(idx_order):
            PT.update_node(fields_n[idx], f'{basename}{coords_suffix[i]}', value=cart_values[idx])

def change_basis(t, transform_matrix, gc_name='GridCoordinates', apply_to_fields=True):
  """Compute the coorindates in the new basis.

  Input is transform matrix from the former basis to the new basis.
  Transform matrix is defined by a plane equation.

  .. math::
     \\ X' = TX
     \\ X  = T⁻¹X'

  where X is the vector in the former basis
        X' is the vector in the new basis
        T is the transformation matrix from the former basis to the new basis
        T⁻¹ is the transformation matrix from the new basis to the former basis

  Args:
    t (Tree) : Recover GridCoordinates from the zones in the tree
               Tree can be a distributed or partioned tree
    transformation_matrix (array) : Transformation matrix from the former basis to the new basis
                                    By default it set on identity matrix.
    gc_name (str) : Name of the coordinates to convert in the new basis and containing the transformation matrix
                    By default it searches the GridCoordinates node
    apply_to fields (bool) : Apply the transformation to fields
                             By default it set on True
  """

  for zone in PT.get_all_Zone_t(t): 

    gc_n = PT.get_child_from_name(zone, gc_name)
    if gc_n is None:
      continue

    coord_transform_n = PT.get_child_from_name(gc_n, 'CoordinateTransform')
    is_aux = coord_transform_n is not None
    reverse = transform_matrix is None

    if reverse:
      transform_matrix = PT.get_value(coord_transform_n)
      transform_matrix = np.linalg.inv(transform_matrix)

    in_suffix = ['Xi', 'Eta', 'Zeta'] if is_aux else ['X', 'Y', 'Z']
    out_suffix = ['X', 'Y', 'Z'] if reverse else ['Xi', 'Eta', 'Zeta']

    if reverse:
      PT.rm_nodes_from_name(zone, 'CoordinateTransform')
    else:
      new_transform_matrix = transform_matrix if coord_transform_n is None else np.dot(transform_matrix, coord_transform_n[1])
      PT.update_child(gc_n, 'CoordinateTransform', 'DataArray_t', new_transform_matrix)
    
    predicates = [gc_name] # Always treat coordinates
    if apply_to_fields:
      predicates.extend(['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t'])

    for predicate in predicates:
      for fields_node in PT.get_children_from_predicates(zone, predicate):
        datanames = [PT.get_name(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
        vectors_basenames = py_utils.find_vector_names(datanames, in_suffix)
        for basename in vectors_basenames:
          vectors_n = [PT.get_node_from_name(fields_node, f"{basename}{c}")  for c in in_suffix]
          tr_fields = np_utils.matmul_cart_vectors(*[PT.get_value(n) for n in vectors_n], transform_matrix)
          for node, s, new_val in zip(vectors_n, out_suffix, tr_fields):
            PT.update_node(node, f'{basename}{s}', value=new_val)
    

def cartesian_to_cylindric(t, revolution_axis, gc_name='GridCoordinates', apply_to_fields=True):
  """Compute cylindric coordinates from any revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cartesian coordinates.

  Args:
    t (Tree) : Recover GridCoordinates from the zones in the tree.
               Tree can be a distributed or partitioned tree.
    revolution_axis (tuple, list, array) : Constant axis.
                                           By default it set on z-axis.  
    gc_name (str) : Name of the coordinates to transform into cylindric coordinates and containing the transformation matrix
                    By default it searches the GridCoordinates node
    apply_to_fields (bool) : Apply the transformation to fields
                             By default it set on True                                              
  """

  revolution_axis = np.asarray(revolution_axis)

  if np.count_nonzero(revolution_axis) != 1:
    transform_matrix = np_utils.create_transform_matrix(revolution_axis)
    change_basis(t, transform_matrix, gc_name, apply_to_fields)
    revolution_axis = np.dot(transform_matrix, revolution_axis)
 
  revolution_axis_unit = revolution_axis / np.linalg.norm(revolution_axis)
  cartesian_to_cylindric_from_unit_revolution_axis(t, revolution_axis_unit, gc_name, apply_to_fields)

def cylindric_to_cartesian(t, revolution_axis, gc_name='GridCoordinates', apply_to_fields=True):
  """Compute cartesian coordinates from any revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cylindric coordinates.

  Args:
    t (Tree) : Recover GridCoordinates from the zones in the tree.
               Tree can be a distributed or partitioned tree.
    revolution_axis (tuple, list, array) : Constant axis
                                           By default it set on z-axis.
    gc_name (str) : Name of the coordinates to transform into cartesian coordinates and containing the transformation matrix
                    By default it searches the GridCoordinates node
    apply_to_fields (bool) : Apply the transformation to fields   
                             By default it set on True             
  """
  
  revolution_axis = np.asarray(revolution_axis)
  need_change_basis = np.count_nonzero(revolution_axis) != 1

  if need_change_basis:
    transform_matrix_n = PT.get_child_from_predicates(t, f'CGNSBase_t/Zone_t/{gc_name}/CoordinateTransform')
    if transform_matrix_n is None: # ???
      return
    transform_matrix = PT.get_value(transform_matrix_n)
    revolution_axis = np.dot(transform_matrix, revolution_axis)

  revolution_axis_unit = revolution_axis / np.linalg.norm(revolution_axis)
  cylindric_to_cartesian_from_unit_revolution_axis(t, revolution_axis_unit, gc_name)

  if need_change_basis:
    change_basis(t, None, gc_name, apply_to_fields)