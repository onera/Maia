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


def create_transform_matrix(revolution_axis=(0, 0, 1)):  
  
  """Create a transform matrix from any axis revolution.

  Input is any revolution axis but must have cartesian coordinates.
  Transform matrix is defined by a plane equation

  .. math::
     \\ ax + by + cz = 0

  where (a, b, c) is the direction vector of the plane equation.

  Args:
    revolution_axis (tuple, list, array) : Constant axis
                                           By default it set on z-axis.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: scale_mesh@start
        :end-before: #scale_mesh@end
        :dedent: 2

  Return the transformation matrix from the former basis toward the new basis  
  """


  assert not (np.array_equal(np.array(revolution_axis), np.zeros(3)))

  if isinstance(revolution_axis, (tuple, list)):
    revolution_axis = np.array(revolution_axis)

  if revolution_axis[0] != 0:
    revolution_axis_bis = np.array([-revolution_axis[1]/revolution_axis[0], 1, 0])   
  elif revolution_axis[1] != 0:
    revolution_axis_bis = np.array([0, -revolution_axis[2]/revolution_axis[1], 1]) 
  elif revolution_axis[2] != 0:
    revolution_axis_bis = np.array([1, 0, -revolution_axis[1]/revolution_axis[2]])
  
  revolution_axis_ter = np.cross(revolution_axis, revolution_axis_bis)

  transform_matrix = np.array([revolution_axis, revolution_axis_bis, revolution_axis_ter])
     
  return transform_matrix

def change_basis(t, transform_matrix=np.eye(3, dtype=float), name_in='GridCoordinates', name_out='GridCoordinatesTransform'):

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
    transformation_matrix (array) : Transfomration matrix from the former basis to the new basis
                                    By default it's the identity matrix.
    name_in (str) : Name of the coordinates to recover in the former basis
    name_out (str) : Name of the coordinates in the new basis

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: scale_mesh@start
        :end-before: #scale_mesh@end
        :dedent: 2
  """

  for zone in PT.get_all_Zone_t(t): 
    transform_matrix_n = PT.get_node_from_predicates(zone, f'{name_in}/CoordinateTransform')
  
    if transform_matrix_n is not None:
      gc_n = PT.get_node_from_name(zone, f'{name_in}')
      ci_1 = PT.get_node_from_name(gc_n, 'CoordinateXi')[1]
      ci_2 = PT.get_node_from_name(gc_n, 'CoordinateEta')[1]
      ci_3 = PT.get_node_from_name(gc_n, 'CoordinateZeta')[1]
    else:
      ci_1, ci_2, ci_3 = PT.Zone.coordinates(zone, name=name_in)
  
    if PT.Zone.Type(zone) == 'Structured':
       ci_1 = ci_1.flatten(order='F')
       ci_2 = ci_2.flatten(order='F')
       ci_3 = ci_3.flatten(order='F')

    vectors = np.array([ci_1, ci_2, ci_3], order='F')
    tranform_vectors = np.dot(transform_matrix, vectors)
    
    if PT.Zone.Type(zone) == 'Structured':
      co_1 = tranform_vectors[0].reshape(PT.Zone.VertexSize(zone), order='F')
      co_2 = tranform_vectors[1].reshape(PT.Zone.VertexSize(zone), order='F')
      co_3 = tranform_vectors[2].reshape(PT.Zone.VertexSize(zone), order='F')
    else:
      co_1 = tranform_vectors[0]
      co_2 = tranform_vectors[1]
      co_3 = tranform_vectors[2]
  
    if transform_matrix_n is not None:
       fields = {'CoordinateX': co_1, 'CoordinateY': co_2, 'CoordinateZ' : co_3}
    else:
       fields = {'CoordinateXi': co_1, 'CoordinateEta': co_2, 'CoordinateZeta' : co_3, 'CoordinateTransform': transform_matrix}

    PT.new_GridCoordinates(name_out, fields=fields, parent=zone)


def cartesian_to_cylindric_from_unit_revolution_axis(t, revolution_axis=(0, 0, 1), name_in = 'GridCoordinates', name_out='GridCoordinatesCyl'):

  """Compute cylinder coordinates from a unit revolution axis.

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
    name_in (str) : Name of the GridCoordinates to transform in cylinder coordinates
                    By default it searches GridCoordinates
    name_out (str) : Name of the GridCoordinates transforms into cylinder coordinates
                     By default name is set to GridCoordinatesCyl

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: scale_mesh@start
        :end-before: #scale_mesh@end
        :dedent: 2
  """
  assert (np.array_equal(revolution_axis, np.array([1, 0, 0]))) or (np.array_equal(revolution_axis, np.array([0, 1, 0]))) or (np.array_equal(revolution_axis, np.array([0, 0, 1])))

  for zone in PT.get_all_Zone_t(t):
    tranform_matrix_n = PT.get_node_from_predicates(zone, f'{name_in}/CoordinateTransform')

    if tranform_matrix_n is not None:
      cx = PT.get_child_from_predicates(zone, f'{name_in}/CoordinateXi')[1]
      cy = PT.get_child_from_predicates(zone, f'{name_in}/CoordinateEta')[1]
      cz = PT.get_child_from_predicates(zone, f'{name_in}/CoordinateZeta')[1]

    else:
       cx, cy, cz = PT.Zone.coordinates(zone, name=name_in)

    if np.array_equal(revolution_axis, np.array([1, 0, 0])):
      c1 = cy
      c2 = cz
      c3 = cx

    elif np.array_equal(revolution_axis, np.array([0, 1, 0])):  
      c1 = cx
      c2 = cz  
      c3 = cy

    elif np.array_equal(revolution_axis, np.array([0, 0, 1])):
      c1 = cx
      c2 = cy
      c3 = cz
    
    else:
       AssertionError

    radius = np.sqrt(c1**2+c2**2)
    theta  = np.arctan2(c2, c1)
    theta[np.isnan(theta)] = np.pi/2

    gc_cyl = PT.new_GridCoordinates(name_out, fields={'CoordinateR':radius, 'CoordinateTheta':theta, 'CoordinateZ': c3}, parent=zone)

    ct_n = PT.get_node_from_predicates(zone, f'{name_in}/CoordinateTransform')
    if ct_n is not None:
       PT.add_child(gc_cyl, ct_n)


def cylinder_to_cartesian_from_unit_revolution_axis(t, revolution_axis=(0, 0, 1), name_in='GridCoordinatesCyl', name_out='GridCoordinates'):

  """Compute the cartesian coordinates from a unit revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cylinder coordinates.
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
    name_in (str) : Name of the GridCoordinates to transform in cartesian coordinates
                    By default it searches GridCoordinatesCyl.
    name_out (str) : Name of the GridCoordinates transforms into cartesian coordinates
                     By default name is set to GridCoordinates.                

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: scale_mesh@start
        :end-before: #scale_mesh@end
        :dedent: 2
  """

  assert (np.array_equal(revolution_axis, np.array([1, 0, 0]))) or (np.array_equal(revolution_axis, np.array([0, 1, 0]))) or (np.array_equal(revolution_axis, np.array([0, 0, 1])))

  for zone in PT.get_all_Zone_t(t):

    radius = PT.get_child_from_predicates(zone, f'{name_in}/CoordinateR')[1]
    theta  = PT.get_child_from_predicates(zone, f'{name_in}/CoordinateTheta')[1]
    z      = PT.get_child_from_predicates(zone, f'{name_in}/CoordinateZ')[1]

    ci_1 = radius*np.cos(theta)
    ci_2 = radius*np.sin(theta)

    if np.array_equal(revolution_axis, np.array([1, 0, 0])):
       co_1 = z
       co_2 = ci_1
       co_3 = ci_2
        
    elif np.array_equal(revolution_axis, np.array([0, 1, 0])):
       co_1 = ci_1
       co_2 = z
       co_3 = ci_2

    elif np.array_equal(revolution_axis, np.array([0, 0, 1])):
       co_1 = ci_1
       co_2 = ci_2
       co_3 = z
    
    else:
       AssertionError

    transform_matrix_n = PT.get_node_from_predicates(zone, f'{name_in}/CoordinateTransform')
    if transform_matrix_n is None:
       fields = {'CoordinateX': co_1, 'CoordinateY': co_2, 'CoordinateZ' : co_3}
    else:
       fields = {'CoordinateXi': co_1, 'CoordinateEta': co_2, 'CoordinateZeta' : co_3, 'CoordinateTransform': transform_matrix_n[1]}

    PT.new_GridCoordinates(name_out, fields=fields, parent=zone)


def cartesian_to_cylinder(t, revolution_axis=(0, 0 ,1)):

  """Compute cylinder coordinates from any revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cartesian coordinates.

  Args:
    t (Tree) : Recover GridCoordinates from the zones in the tree.
               Tree can be a distributed or partitioned tree.
    revolution_axis (tuple, list, array) : Constant axis.
                                           By default it set on z-axis.             

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: scale_mesh@start
        :end-before: #scale_mesh@end
        :dedent: 2
  """

  if isinstance(revolution_axis, (tuple, list)):
     revolution_axis = np.array(revolution_axis)

  if len(np.where(revolution_axis==0)[0]) == 2:
    revolution_axis_unit = revolution_axis / np.linalg.norm(revolution_axis)
    cartesian_to_cylindric_from_unit_revolution_axis(t, revolution_axis=revolution_axis_unit)
  else:
    transform_matrix = create_transform_matrix(revolution_axis)
    change_basis(t, transform_matrix=transform_matrix)
    new_revolution_axis = np.dot(transform_matrix, revolution_axis)
    new_revolution_axis_unit = new_revolution_axis / np.linalg.norm(new_revolution_axis)
    cartesian_to_cylindric_from_unit_revolution_axis(t, revolution_axis=new_revolution_axis_unit, name_in='GridCoordinatesTransform')
       

def cylinder_to_cartesian(t, revolution_axis=(0, 0, 1)):

  """Compute cartesian coordinates from any revolution axis.

  Input zone(s) in the tree can be either structured or unstructured, but must have cylinder coordinates.

  Args:
    t (Tree) : Recover GridCoordinates from the zones in the tree.
               Tree can be a distributed or partitioned tree.
    revolution_axis (tuple, list, array) : Constant axis
                                           By default it set on z-axis.                

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: scale_mesh@start
        :end-before: #scale_mesh@end
        :dedent: 2
  """
  
  if isinstance(revolution_axis, (tuple, list)):
     revolution_axis = np.array(revolution_axis)

  if len(np.where(revolution_axis==0)[0]) == 2:
    revolution_axis_unit = revolution_axis / np.linalg.norm(revolution_axis)
    cylinder_to_cartesian_from_unit_revolution_axis(t, revolution_axis=revolution_axis_unit)
  else:
    transform_matrix = PT.get_nodes_from_predicates(t, 'CGNSBase_t/Zone_t/GridCoordinates_t/CoordinateTransform')[0][1]
    new_revolution_axis = np.dot(transform_matrix, revolution_axis)
    new_revolution_axis_unit = new_revolution_axis / np.linalg.norm(new_revolution_axis)
    transform_matrix = np.linalg.inv(transform_matrix)
    cylinder_to_cartesian_from_unit_revolution_axis(t, revolution_axis=new_revolution_axis_unit, name_out='GridCoordinatesTransform')
    change_basis(t, transform_matrix=transform_matrix, name_in='GridCoordinatesTransform', name_out='GridCoordinates')