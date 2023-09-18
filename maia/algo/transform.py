import numpy as np

import maia.pytree as PT
from maia.utils import py_utils, np_utils
from maia.algo.apply_function_to_nodes import zones_iterator

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

  where c, t are the rotation center and translation vector and R is the rotation matrix.
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
    # Transform coords
    for grid_co in PT.iter_children_from_label(zone, "GridCoordinates_t"):
      coords_n = [PT.get_child_from_name(grid_co, f"Coordinate{c}")  for c in ['X', 'Y', 'Z']]
      phy_dim = 2 if coords_n[2] is None else 3
      coords_n = coords_n[:phy_dim]
      coords = [PT.get_value(n) for n in coords_n]
    
      if phy_dim == 3:
        tr_coords = np_utils.transform_cart_vectors(*coords, translation, rotation_center, rotation_angle)
      else:
        tr_coords = np_utils.transform_cart_vectors_2d(*coords, translation, rotation_center, rotation_angle)
      for coord_n, tr_coord in zip(coords_n, tr_coords):
        PT.set_value(coord_n, tr_coord)

    # Transform fields
    if apply_to_fields:
      fields_nodes  = PT.get_children_from_label(zone, "FlowSolution_t")
      fields_nodes += PT.get_children_from_label(zone, "DiscreteData_t")
      fields_nodes += PT.get_children_from_label(zone, "ZoneSubRegion_t")
      for bc in PT.iter_children_from_predicates(zone, "ZoneBC_t/BC_t"):
        fields_nodes += PT.get_children_from_label(bc, "BCDataSet_t")
      for fields_node in fields_nodes:
        data_names = [PT.get_name(data) for data in PT.iter_nodes_from_label(fields_node, "DataArray_t")]
        cartesian_vectors_basenames = py_utils.find_cartesian_vector_names(data_names, phy_dim)
        for basename in cartesian_vectors_basenames:
          vectors_n = [PT.get_node_from_name_and_label(fields_node, f"{basename}{c}", 'DataArray_t')  for c in ['X', 'Y', 'Z'][:phy_dim]]
          vectors = [PT.get_value(n) for n in vectors_n]
          # Assume that vectors are position independant
          # Be careful, if coordinates vector needs to be transform, the translation is not applied !
          if phy_dim == 3:
            tr_vectors = np_utils.transform_cart_vectors(*vectors, rotation_center=rotation_center, rotation_angle=rotation_angle)
          else:
            tr_vectors = np_utils.transform_cart_vectors_2d(*vectors, rotation_center=rotation_center, rotation_angle=rotation_angle)
          for vector_n, tr_vector in zip(vectors_n, tr_vectors):
            PT.set_value(vector_n, tr_vector)


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
  for zone in zones_iterator(t):
    for grid_co in PT.get_children_from_label(zone, 'GridCoordinates_t'):
      for idir, dir in enumerate(['X', 'Y', 'Z']):
        node = PT.get_child_from_name(grid_co, f'Coordinate{dir}')
        if node is not None:
          node[1] *= scaling[idir]
