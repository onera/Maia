import numpy as np

import maia.pytree as PT
from maia.utils import py_utils, np_utils
from maia.algo.apply_function_to_nodes import zones_iterator

from maia.utils import logging as mlog

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

def _transform_cartesian_to_cylindric_unit(zone, revolution_axis=(0, 0, 1), gc_name='GridCoordinates', name='GridCoordinates', basename='Coordinate'):

  """Compute cylindric coordinates from a unit revolution axis and update the node coordinates in the new coordinate system.

  Input zone(s) in the tree can be either structured or unstructured, but must have cartesian coordinates.
  Transformation is defined by

  .. math::
     \\ r = np.sqrt(x² + y²)
     \\ theta = np.arctan(y/x)
     \\ z = z

  where x, y are coordinates on (x,y) plan.

  Args:
    zone (Tree) : Recover GridCoordinates from the zones in the tree
                  Zone can be a distributed or partitioned tree.
    revolution_axis (tuple, list, array) : Constant axis
                                           By default it set on z-axis.
    gc_name (str) : Name of the GridCoordinates to transform in cylindric coordinates and containing the transformation matrix
                    By default it searches GridCoordinates node
    name (str) : Name of the node containing cartesian vectors to transform into cylindric coordinate
                 By default it set searches the GridCoordinates node
    basename (str) : Vector name without suffix of coordinate system
    apply_to_fields (bool) : Apply the transformation to fields 
                             By default it set on True
  """

  if PT.get_node_from_name(zone, gc_name) is None:
      return
  
  if isinstance(revolution_axis, (tuple, list)):
    revolution_axis = np.array(revolution_axis)

  assert (np.array_equal(revolution_axis, np.array([1, 0, 0]))) or (np.array_equal(revolution_axis, np.array([0, 1, 0]))) or (np.array_equal(revolution_axis, np.array([0, 0, 1])))

  transform_matrix_n = PT.get_node_from_predicates(zone, f'{gc_name}/CoordinateTransform')
  transform_node = PT.get_node_from_name(zone, f'{name}')

  if transform_matrix_n is not None:
    coords_suffix = ['Xi', 'Eta', 'Zeta']
  else:
    coords_suffix = ['X', 'Y', 'Z']
  
  coords_n = [PT.get_child_from_name(transform_node, f'{basename}{suffix}') for suffix in coords_suffix]
  coords   = [PT.get_value(node) for node in coords_n]

  if np.array_equal(revolution_axis, np.array([1, 0, 0])):
    c1 = coords[1]
    c2 = coords[2]
    c3 = coords[0]
    c1_n = coords_n[1]
    c2_n = coords_n[2]
    c3_n = coords_n[0]

  elif np.array_equal(revolution_axis, np.array([0, 1, 0])):  
    c1 = coords[0]
    c2 = coords[2]
    c3 = coords[1]
    c1_n = coords_n[0]
    c2_n = coords_n[2]
    c3_n = coords_n[1]

  elif np.array_equal(revolution_axis, np.array([0, 0, 1])):
    c1 = coords[0]
    c2 = coords[1]
    c3 = coords[2]
    c1_n = coords_n[0]
    c2_n = coords_n[1]
    c3_n = coords_n[2]

  radius = np.sqrt(c1**2+c2**2)
  theta  = np.arctan2(c2, c1)

  PT.update_node(c1_n, f'{basename}R', value=radius)
  PT.update_node(c2_n, f'{basename}Theta', value=theta)
  PT.update_node(c3_n, f'{basename}Z', value=c3)

  ct_n = PT.get_node_from_predicates(zone, f'{gc_name}/CoordinateTransform')
  if ct_n is None:
    gc_n = PT.get_node_from_name(zone, f'{gc_name}')
    PT.add_child(gc_n, ct_n)
  
  return coords_suffix

def _transform_cylindric_to_cartesian_unit(zone, revolution_axis=(0, 0, 1), gc_name='GridCoordinates', name='GridCoordinates', basename='Coordinate'):

  """Compute cylindric coordinates from a unit revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cartesian coordinates.
  Transformation is defined by

  .. math::
     \\ x = r*cos(theta)
     \\ y = r*sin(theta)
     \\ z = z

  where r, theta are respectively the radius and the angle.

  Args:
    zone (Tree) : Recover GridCoordinates from the zones in the tree
                  Zone can be a distributed or partitioned tree.
    revolution_axis (tuple, list, array) : Constant axis
                                           By default it set on z-axis.
    gc_name (str) : Name of the GridCoordinates to transform in cartesian coordinates and containing the transformation matrix
                    By default it searches the GridCoordinates node
    name (str) : Name of the node containing cartesian vectors to transform into cylindric coordinate
                 By default it searches GridCoordinates node
    basename (str) : Vector name without suffix of coordinate system
    apply_to_fields (bool) : Apply the transformation to fields 
                             By default it set on True
  """

  if PT.get_node_from_name(zone, gc_name) is None:
      return

  if isinstance(revolution_axis, (tuple, list)):
    revolution_axis = np.array(revolution_axis)
  
  assert (np.array_equal(revolution_axis, np.array([1, 0, 0]))) or (np.array_equal(revolution_axis, np.array([0, 1, 0]))) or (np.array_equal(revolution_axis, np.array([0, 0, 1])))

  transform_node = PT.get_node_from_name(zone, f'{name}')

  coords_n = [PT.get_child_from_name(transform_node, f'{basename}{suffix}') for suffix in ['R', 'Theta', 'Z']]
  coords   = [PT.get_value(node) for node in coords_n]

  cyl_1 = coords[0]*np.cos(coords[1])
  cyl_2 = coords[0]*np.sin(coords[1])

  if np.array_equal(revolution_axis, np.array([1, 0, 0])):
    c1 = coords[2]
    c2 = cyl_1
    c3 = cyl_2
    c1_n = coords_n[2]
    c2_n = coords_n[0]
    c3_n = coords_n[1]
        
  elif np.array_equal(revolution_axis, np.array([0, 1, 0])):
    c1 = cyl_1
    c2 = coords[2]
    c3 = cyl_2
    c1_n = coords_n[0]
    c2_n = coords_n[2]
    c3_n = coords_n[1]

  elif np.array_equal(revolution_axis, np.array([0, 0, 1])):
    c1 = cyl_1
    c2 = cyl_2
    c3 = coords[2]
    c1_n = coords_n[0]
    c2_n = coords_n[1]
    c3_n = coords_n[2]

  transform_matrix_n = PT.get_node_from_predicates(zone, f'{gc_name}/CoordinateTransform')
  if transform_matrix_n is not None:
    coords_suffix = ['Xi', 'Eta', 'Zeta']
  else : 
    coords_suffix = ['X', 'Y', 'Z']

  PT.update_node(c1_n, f'{basename}{coords_suffix[0]}', value=c1)
  PT.update_node(c2_n, f'{basename}{coords_suffix[1]}', value=c2)
  PT.update_node(c3_n, f'{basename}{coords_suffix[2]}', value=c3)
        
def cartesian_to_cylindric_from_unit_revolution_axis(t, revolution_axis=(0, 0, 1), gc_name='GridCoordinates', apply_to_fields=True):

  """Compute cylindric coordinates from a unit revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cartesian coordinates.

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
  assert (np.array_equal(revolution_axis, np.array([1, 0, 0]))) or (np.array_equal(revolution_axis, np.array([0, 1, 0]))) or (np.array_equal(revolution_axis, np.array([0, 0, 1])))

  if isinstance(revolution_axis, (tuple, list)):
    revolution_axis = np.array(revolution_axis)

  for zone in PT.get_all_Zone_t(t):

    if PT.get_node_from_name(zone, gc_name) is None:
      continue

    coords_suffix = _transform_cartesian_to_cylindric_unit(zone, revolution_axis, gc_name=gc_name)

    if apply_to_fields:
      for predicate in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t']:
        for fields_node in PT.get_children_from_predicates(zone, predicate):
          datanames = [PT.get_name(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
          cartesian_vectors_basenames = py_utils.find_vector_names(datanames, coords_suffix)
          for basename in cartesian_vectors_basenames:
            _transform_cartesian_to_cylindric_unit(zone, revolution_axis, gc_name=gc_name, name=fields_node[0], basename=basename)
     
def cylindric_to_cartesian_from_unit_revolution_axis(t, revolution_axis=(0, 0, 1), gc_name='GridCoordinates', apply_to_fields=True):

  """Compute the cartesian coordinates from a unit revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cylindric coordinates.

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

  assert (np.array_equal(revolution_axis, np.array([1, 0, 0]))) or (np.array_equal(revolution_axis, np.array([0, 1, 0]))) or (np.array_equal(revolution_axis, np.array([0, 0, 1])))

  for zone in PT.get_all_Zone_t(t):

    if PT.get_node_from_name(zone, gc_name) is None:
      continue

    _transform_cylindric_to_cartesian_unit(zone, revolution_axis, gc_name=gc_name)

    if apply_to_fields:
      for predicate in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t']:
        for fields_node in PT.get_children_from_predicates(zone, predicate):
          datanames = [PT.get_name(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
          cylindric_vectors_basenames = py_utils.find_vector_names(datanames, ['R', 'Theta', 'Z'])
          for basename in cylindric_vectors_basenames:
            _transform_cylindric_to_cartesian_unit(zone, revolution_axis, gc_name=gc_name, name=fields_node[0], basename=basename) 

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

    if PT.get_node_from_name(zone, gc_name) is None:
      continue

    coord_transform_n = PT.get_node_from_predicates(zone, f'{gc_name}/CoordinateTransform')
    is_aux = coord_transform_n is not None
    gc_n = PT.get_node_from_name(zone, f'{gc_name}')

    reverse = transform_matrix is None

    if reverse:
      transform_matrix = PT.get_value(coord_transform_n)
      transform_matrix = np.linalg.inv(transform_matrix)

    coords_suffix = ['Xi', 'Eta', 'Zeta'] if is_aux else ['X', 'Y', 'Z']

    coords_n = [PT.get_child_from_name(gc_n, f'Coordinate{suffix}') for suffix in coords_suffix]
    coords = [PT.get_value(node) for node in coords_n]

    co_1, co_2, co_3 = py_utils.apply_cart_vectors(*coords, transform_matrix)

    if reverse:
      coords_name = ['CoordinateX', 'CoordinateY', 'CoordinateZ']
      PT.rm_nodes_from_name(zone, 'CoordinateTransform')
    else:
      coords_name = ['CoordinateXi', 'CoordinateEta', 'CoordinateZeta']
      new_transform_matrix = transform_matrix if coord_transform_n is None else np.dot(transform_matrix, coord_transform_n[1])
      PT.update_child(gc_n, 'CoordinateTransform', 'DataArray_t', new_transform_matrix)
    
    PT.update_node(coords_n[0], coords_name[0], value=co_1)
    PT.update_node(coords_n[1], coords_name[1], value=co_2)
    PT.update_node(coords_n[2], coords_name[2], value=co_3)

    if apply_to_fields:
      for predicate in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'ZoneBC_t/BC_t/BCDataSet_t']:
        for fields_node in PT.get_children_from_predicates(zone, predicate):
          datanames = [PT.get_name(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
          cartesian_vectors_basenames = py_utils.find_vector_names(datanames, coords_suffix)
          for basename in cartesian_vectors_basenames:
            vectors_n = [PT.get_node_from_name_and_label(fields_node, f"{basename}{c}", 'DataArray_t')  for c in coords_suffix]
            vectors = [PT.get_value(n) for n in vectors_n]

            transform_fields = py_utils.apply_cart_vectors(vectors[0], vectors[1], vectors[2], transform_matrix)

            if reverse:
              fields_name = [f'{basename}X', f'{basename}Y', f'{basename}Z']
            else:
              fields_name = [f'{basename}Xi', f'{basename}Eta', f'{basename}Zeta']

            PT.update_node(vectors_n[0], fields_name[0], value=transform_fields[0])
            PT.update_node(vectors_n[1], fields_name[1], value=transform_fields[1])
            PT.update_node(vectors_n[2], fields_name[2], value=transform_fields[2])