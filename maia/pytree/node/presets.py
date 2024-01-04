import warnings
from maia.pytree.typing import *

from maia.pytree.cgns_keywords import cgns_to_dtype

from maia.pytree.node import access as NA
from maia.pytree.node import new_node


def _check_parent_label(node, parent, allowed_list):
  if parent is not None and NA.get_label(parent) not in allowed_list:
    msg = f"Attaching node {NA.get_name(node)} ({NA.get_label(node)}) under a {NA.get_label(parent)} parent" \
          f" is not SIDS compliant. Admissible parent labels are {allowed_list}."
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


# Specialized
def new_CGNSTree(*, version:float=4.2):
  """ Create a CGNSTree_t node

  Args:
    version (float): Number used to fill the CGNSLibraryVersion data
  Example:
    >>> node = PT.new_CGNSTree()
    >>> PT.print_tree(node)
    CGNSTree CGNSTree_t 
    └───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
  """
  version = new_node('CGNSLibraryVersion', 'CGNSLibraryVersion_t', value=version)
  return new_node('CGNSTree', label='CGNSTree_t', children=[version])

def new_CGNSBase(name:str='Base', *, cell_dim:int=3, phy_dim:int=3, parent:CGNSTree=None):
  """ Create a CGNSBase_t node

  Link to corresponding SIDS section:
  `CGNSBase_t <https://cgns.github.io/CGNS_docs_current/sids/cgnsbase.html#CGNSBase>`_

  Args:
    name (str): Name of the created base
    cell_dim (one of 1,2,3): Cell dimension of the mesh
    phy_dim (one of 1,2,3): Physical dimension of the mesh
    parent (CGNSTree): Node to which created base should be attached
  Example:
    >>> node = PT.new_CGNSBase('Base', cell_dim=2)
    >>> PT.print_tree(node)
    Base CGNSBase_t I4 [2 3]
  """
  node = new_node(name, 'CGNSBase_t', value=[cell_dim, phy_dim], parent=parent)
  _check_parent_label(node, parent, ['CGNSTree_t'])
  return node

def new_Family(name:str='Family', *, family_bc:str=None, parent:CGNSTree=None):
  """ Create a Family_t node

  Link to corresponding SIDS section:
  `Family_t <https://cgns.github.io/CGNS_docs_current/sids/misc.html#Family>`_
  
  Args:
    name (str): Name of the created family
    family_bc (str):  If specified, create a FamilyBC taking this value under the Family node
    parent (CGNSTree): Node to which created family should be attached
  Example:
    >>> node = PT.new_Family('WALL', family_bc='BCWall')
    >>> PT.print_tree(node)
    WALL Family_t 
    └───FamilyBC FamilyBC_t "BCWall"
  """
  family = new_node(name, 'Family_t', None, [], parent=parent)
  _check_parent_label(family, parent, ['CGNSBase_t'])
  if family_bc is not None:
    new_FamilyBC(family_bc, family)
  return family

def new_FamilyName(family_name:str, as_additional='', parent:CGNSTree=None):
  """ Create a FamilyName_t or an AdditionalFamilyName_t node

  Args:
    family_name (str): Name of the family to which the FamilyName node refers
    as_additional (str) : If provided, node is created as an AdditionalFamilyName_t node
      named after this str
    parent (CGNSTree): Node to which created node should be attached
  Example:
    >>> node = PT.new_FamilyName('MyFamily')
    >>> PT.print_tree(node)
    FamilyName FamilyName_t "MyFamily"
    >>> node = PT.new_FamilyName('MyFamily')
    >>> PT.print_tree(node)
    AddFamName AdditionalFamilyName_t "MyFamily"
  """

  if as_additional:
    node = new_node(as_additional, 'AdditionalFamilyName_t', family_name, [], parent)
  else:
    node = new_node('FamilyName', 'FamilyName_t', family_name, [], parent)

  _check_parent_label(node, parent, ['Family_t', 'UserDefinedData_t', 'ZoneSubRegion_t', 'BC_t', 'Zone_t'])
  return node

def new_FamilyBC(family_bc:str, parent:CGNSTree=None):
  allowed_bc = """Null UserDefined BCAxisymmetricWedge BCDegenerateLine BCDegeneratePoint BCDirichlet BCExtrapolate
  BCFarfield BCGeneral BCInflow BCInflowSubsonic BCInflowSupersonic BCNeumann BCOutflow BCOutflowSubsonic 
  BCOutflowSupersonic BCSymmetryPlane BCSymmetryPolar BCTunnelInflow BCTunnelOutflow BCWall BCWallInviscid
  BCWallViscous BCWallViscousHeatFlux BCWallViscousIsothermal FamilySpecified""".split()
  assert family_bc in allowed_bc
  return new_node('FamilyBC', 'FamilyBC_t', family_bc, [], parent)

def new_Zone(name:str='Zone', *, type:str='Null', size:ArrayLike=None, family:str=None, parent:CGNSTree=None):
  """ Create a Zone_t node

  Note that the size array will not be reshaped and must consequently match the expected layout
  
    [[n_vtx, n_cell, n_bnd_vtx] for each IndexDimension]

  for example, [[11,10,0]] for an unstructured zone or [[11,10,0], [6,5,0]] for a 2D structured zone.

  Link to corresponding SIDS section:
  `Zone_t <https://cgns.github.io/CGNS_docs_current/sids/cgnsbase.html#Zone>`_

  Args:
    name (str): Name of the created zone
    type ({'Null', 'UserDefined', 'Structured' or 'Unstructured'}) : Type of the zone
    size (ArrayLike) : Size of the zone.
    family (str) : If specified, create a FamilyName refering to this family
    parent (CGNSTree): Node to which created Zone should be attached
  Example:
    >>> node = PT.new_Zone('Zone', type='Unstructured', 
    ...                    size=[[11,10,0]], family='Rotor')
    >>> PT.print_tree(node)
    Zone Zone_t I4 [[11 10  0]]
    ├───ZoneType ZoneType_t "Unstructured"
    └───FamilyName FamilyName_t "Rotor"
  """
  assert type in ['Null', 'UserDefined', 'Structured', 'Unstructured']
  zone = new_node(name, 'Zone_t', size, [], parent)
  _check_parent_label(zone, parent, ['CGNSBase_t'])
  new_node('ZoneType', 'ZoneType_t', type, [], zone)
  if family:
    new_FamilyName(family, parent=zone)
  return zone

def new_Elements(name:str='Elements', type:str='Null', *, erange:ArrayLike=None, econn:ArrayLike=None, parent:CGNSTree=None):
  """ Create an Element_t node

  This function is designed to create standard elements. 
  See :func:`new_NGonElements` or :func:`new_NFaceElements` to create polygonal
  elements.

  Link to corresponding SIDS section:
  `Elements_t <https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#Elements>`_

  Args:
    name (str): Name of the created element node
    type (str) : CGNSName of the element section, for example ``PYRA_5``
    erange (ArrayLike) : ElementRange array of the elements
    econn (ArrayLike) : ElementConnectivity array of the elements
    parent (CGNSTree): Node to which created elements should be attached
  Example:
    >>> node = PT.new_Elements('Edges', type='BAR_2', erange=[1,4],
    ...                        econn=[1,2, 2,3, 3,4, 4,1])
    >>> PT.print_tree(node)
    Edges Elements_t I4 [3 0]
    ├───ElementRange IndexRange_t I4 [1 4]
    └───ElementConnectivity DataArray_t I4 [1 2 2 3 3 4 4 1]
  """
  from maia.pytree.sids import elements_utils as EU
  if isinstance(type, str):
    _value = [EU.cgns_name_to_id(type), 0]
  elif isinstance(type, int):
    _value = [type, 0]
  else:
    _value = type # Try autoconversion in new_node
  elem = new_node(name, "Elements_t", _value, [], parent)
  _check_parent_label(elem, parent, ['Zone_t'])
  if erange is not None:
    new_node('ElementRange', 'IndexRange_t', erange, [], elem)
  if econn is not None:
    new_DataArray('ElementConnectivity', econn, parent=elem)
  return elem

def new_NGonElements(name:str = 'NGonElements',
                     *,
                     erange:ArrayLike = None,
                     eso:ArrayLike = None,
                     ec:ArrayLike = None,
                     pe:ArrayLike = None,
                     pepos:ArrayLike = None,
                     parent:CGNSTree = None):
  """ Create an Element_t node describing a NGON_n connectivity

  Link to corresponding SIDS section:
  `Elements_t <https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#Elements>`_

  Args:
    name (str): Name of the created element node
    erange (ArrayLike) : ElementRange array of the elements
    eso (ArrayLike) : ElementStartOffset array of the elements
    ec (ArrayLike) : ElementConnectivity array of the elements
    pe (ArrayLike) : ParentElements array of the elements
    pepos (ArrayLike) : ParentElementsPosition array of the elements
    parent (CGNSTree): Node to which created elements should be attached
  Example:
    >>> node = PT.new_NGonElements(erange=[1,4], eso=[0,3,6,9,12],
    ...                            ec=[1,3,2, 1,2,4, 2,3,4, 3,1,4])
    >>> PT.print_tree(node)
    NGonElements Elements_t I4 [22  0]
    ├───ElementRange IndexRange_t I4 [1 4]
    ├───ElementStartOffset DataArray_t I4 [ 0  3  6  9 12]
    └───ElementConnectivity DataArray_t I4 (12,)
  """
  elem = new_Elements(name, 22, erange=erange, parent=parent)
  names = ['ElementStartOffset', 'ElementConnectivity', 'ParentElements', 'ParentElementsPosition']
  for name, val in zip(names, [eso, ec, pe, pepos]):
    if val is not None:
      new_DataArray(name, val, parent=elem)
  return elem

def new_NFaceElements(name:str = 'NFaceElements',
                      *,
                      erange:ArrayLike = None,
                      eso:ArrayLike = None,
                      ec:ArrayLike = None,
                      parent:CGNSTree = None):
  """ Create an Element_t node describing a NFACE_n connectivity

  Link to corresponding SIDS section:
  `Elements_t <https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#Elements>`_

  Args:
    name (str): Name of the created element node
    erange (ArrayLike) : ElementRange array of the elements
    eso (ArrayLike) : ElementStartOffset array of the elements
    ec (ArrayLike) : ElementConnectivity array of the elements
    parent (CGNSTree): Node to which created elements should be attached
  Example:
    >>> node = PT.new_NFaceElements(erange=[5,5], eso=[0,4], ec=[1,2,3,4])
    >>> PT.print_tree(node)
    NFaceElements Elements_t I4 [23  0]
    ├───ElementRange IndexRange_t I4 [5 5]
    ├───ElementStartOffset DataArray_t I4 [0 4]
    └───ElementConnectivity DataArray_t I4 [1 2 3 4]
  """
  elem = new_Elements(name, 23, erange=erange, parent=parent)
  for name, val in zip(['ElementStartOffset', 'ElementConnectivity'], [eso, ec]):
    if val is not None:
      new_DataArray(name, val, parent=elem)
  return elem

def new_ZoneBC(parent:CGNSTree=None):
  """ Create a ZoneBC_t node

  Args:
    parent (CGNSTree): Node to which created ZBC should be attached
  """
  node = new_node('ZoneBC', 'ZoneBC_t', None, [], parent)
  _check_parent_label(node, parent, ['Zone_t'])
  return node

def new_BC(name:str = 'BC',
           type:str='Null',
           *,
           point_range:ArrayLike = None,
           point_list:ArrayLike = None,
           loc:str = None,
           family:str = None,
           parent:CGNSTree = None):
  """ Create a BC_t node

  The patch defining the BC must be provided using either ``point_range`` or
  ``point_list`` parameter : both can no be used simultaneously.

  Link to corresponding SIDS section:
  `BC_t <https://cgns.github.io/CGNS_docs_current/sids/bc.html#BC>`_

  Args:
    name (str): Name of the created bc node
    type (str) : Type of the boundary condition
    point_range (ArrayLike) : PointRange array defining the BC
    point_list (ArrayLike) : PointList array defining the BC
    loc (str) : If specified, create a GridLocation taking this value
    family (str) : If specified, create a FamilyName taking this value
    parent (CGNSTree): Node to which created bc should be attached
  Example:
    >>> node = PT.new_BC('BC', 'BCWall', point_list=[[1,5,10,15]],
    ...                  loc='FaceCenter', family='WALL')
    >>> PT.print_tree(node)
    BC BC_t "BCWall"
    ├───GridLocation GridLocation_t "FaceCenter"
    ├───FamilyName FamilyName_t "WALL"
    └───PointList IndexArray_t I4 [[ 1  5 10 15]]
  """
  allowed_bc = """Null UserDefined BCAxisymmetricWedge BCDegenerateLine BCDegeneratePoint BCDirichlet BCExtrapolate
  BCFarfield BCGeneral BCInflow BCInflowSubsonic BCInflowSupersonic BCNeumann BCOutflow BCOutflowSubsonic 
  BCOutflowSupersonic BCSymmetryPlane BCSymmetryPolar BCTunnelInflow BCTunnelOutflow BCWall BCWallInviscid
  BCWallViscous BCWallViscousHeatFlux BCWallViscousIsothermal FamilySpecified""".split()
  assert type in allowed_bc
  
  bc = new_node(name, 'BC_t', type, [], parent)
  _check_parent_label(bc, parent, ['ZoneBC_t'])
  if loc is not None:
    new_GridLocation(loc, bc)
  if family is not None:
    new_FamilyName(family, parent=bc)
  if point_range is not None:
    assert point_list is None
    new_IndexRange('PointRange', point_range, bc)
  if point_list is not None:
    assert point_range is None
    new_IndexArray('PointList', point_list, bc)
  return bc

def new_ZoneGridConnectivity(name:str='ZoneGridConnectivity', parent:CGNSTree=None):
  """ Create a ZoneGridConnectivity_t node

  Args:
    name (str): Name of the created ZoneGridConnectivity node
    parent (CGNSTree): Node to which created ZGC should be attached
  """
  node = new_node(name, 'ZoneGridConnectivity_t', None, [], parent)
  _check_parent_label(node, parent, ['Zone_t'])
  return node

def new_GridConnectivity(name:str = 'GC',
                         donor_name:str = None,
                         type:str = 'Null',
                         *,
                         loc:str = None, 
                         point_range:ArrayLike = None, 
                         point_range_donor:ArrayLike = None, 
                         point_list:ArrayLike = None, 
                         point_list_donor:ArrayLike = None, 
                         parent:CGNSTree = None):
  """ Create a GridConnectivity_t node

  The patch defining the GC must be provided using either ``point_range`` or
  ``point_list`` parameter : both can no be used simultaneously.
  The same applies for the opposite patch definition (using ``point_range_donor``
  or ``point_list_donor``).

  Link to corresponding SIDS section:
  `GridConnectivity_t <https://cgns.github.io/CGNS_docs_current/sids/cnct.html#GridConnectivity>`_

  Args:
    name (str): Name of the created gc node
    donor_name (str) : Name or path of the opposite zone
    type (one of 'Null', 'UserDefined', 'Overset', 'Abutting' or 'Abutting1to1') : Type of the gc node
    loc (str) : If specified, create a GridLocation taking this value
    point_range (ArrayLike) : PointRange array defining the current patch
    point_range_donor (ArrayLike) : PointRange array defining the opposite patch
    point_list (ArrayLike) : PointList array defining the current patch
    point_list_donor (ArrayLike) : PointList array defining the opposite patch
    parent (CGNSTree): Node to which created gc should be attached
  Example:
    >>> node = PT.new_GridConnectivity('GC', 'Zone', 'Abutting1to1',
    ...           point_list=[[1,4,7]], point_list_donor=[[3,6,9]])
    >>> PT.print_tree(node)
    GC GridConnectivity_t "Zone"
    ├───GridConnectivityType GridConnectivityType_t "Abutting1to1"
    ├───PointList IndexArray_t I4 [[1 4 7]]
    └───PointListDonor IndexArray_t I4 [[3 6 9]]
  """
  gc = new_node(name, 'GridConnectivity_t', donor_name, [], parent)
  _check_parent_label(gc, parent, ['ZoneGridConnectivity_t'])
  new_GridConnectivityType(type, parent=gc)
  if loc is not None:
    new_GridLocation(loc, gc)
  if point_range is not None:
    assert point_list is None
    new_IndexRange('PointRange', value=point_range, parent=gc)
  if point_list is not None:
    assert point_range is None
    new_IndexArray('PointList', value=point_list, parent=gc)
  if point_range_donor is not None:
    assert point_list_donor is None
    new_IndexRange('PointRangeDonor', value=point_range_donor, parent=gc)
  if point_list_donor is not None:
    assert point_range_donor is None
    new_IndexArray('PointListDonor', value=point_list_donor, parent=gc)
  return gc

def new_GridConnectivityType(type:str="Null", parent:CGNSTree=None):
  allowed_gc = "Null UserDefined Overset Abutting Abutting1to1".split()
  assert type in allowed_gc
  return new_node('GridConnectivityType', 'GridConnectivityType_t', type, [], parent)

def new_Periodic(rotation_angle:ArrayLike = [0., 0., 0.],
                 rotation_center:ArrayLike = [0., 0., 0.],
                 translation:ArrayLike = [0.,0.,0],
                 parent:CGNSTree = None):
  childs = [
      new_DataArray('RotationAngle', rotation_angle),
      new_DataArray('RotationCenter', rotation_center),
      new_DataArray('Translation', translation)
      ]
  return new_node('Periodic', 'Periodic_t', None, childs, parent)

def new_GridConnectivityProperty(periodic:Dict[str,ArrayLike]={}, parent:CGNSTree=None):
  """ Create a GridConnectivityProperty node

  The main interest of this function is to add periodic information to a GC_t node;
  this can be done with the ``periodic`` parameter which maps the keys
  'rotation_angle', 'rotation_center' and 'translation' to the corresponding arrays.

  Missing keys defaults to ``np.zeros(3)``, users should be careful if 
  when the physical dimension of the mesh if lower than 3.

  Link to corresponding SIDS section:
  `GridConnectivityProperty_t <https://cgns.github.io/CGNS_docs_current/sids/cnct.html#GridConnectivityProperty>`_

  Args:
    periodic (dict): Name of the created gc node
    parent (CGNSTree): Node to which created gc prop should be attached
  Example:
    >>> perio = {"translation" : [1.0, 0.0, 0.0]}
    >>> node = PT.new_GridConnectivityProperty(perio)
    >>> PT.print_tree(node)
    GridConnectivityProperty GridConnectivityProperty_t 
    └───Periodic Periodic_t 
        ├───RotationAngle DataArray_t R4 [0. 0. 0.]
        ├───RotationCenter DataArray_t R4 [0. 0. 0.]
        └───Translation DataArray_t R4 [1. 0. 0.]
  """
  gc_props = new_node('GridConnectivityProperty', 'GridConnectivityProperty_t', None, [], parent)
  _check_parent_label(gc_props, parent, ['GridConnectivity_t', 'GridConnectivity1to1_t'])
  if periodic:
    new_Periodic(**periodic, parent=gc_props)
  return gc_props

def new_GridConnectivity1to1(name:str = 'GC',
                            donor_name:str = None,
                            *,
                            point_range:ArrayLike = None, 
                            point_range_donor:ArrayLike = None,
                            transform:ArrayLike = None,
                            parent:CGNSTree = None):
  """ Create a GridConnectivity1to1_t node

  GridConnectivity1to1_t are reserved for structured zones. See
  :func:`new_GridConnectivity` to create general GridConnectivity_t nodes.

  Link to corresponding SIDS section:
  `GridConnectivity1to1_t <https://cgns.github.io/CGNS_docs_current/sids/cnct.html#GridConnectivity1to1>`_

  Args:
    name (str): Name of the created gc node
    donor_name (str) : Name or path of the opposite zone
    point_range (ArrayLike) : PointRange array defining the current patch
    point_range_donor (ArrayLike) : PointRange array defining the opposite patch
    transform (array of int) : short notation of the transformation matrix
    parent (CGNSTree): Node to which created gc should be attached
  Example:
    >>> node = PT.new_GridConnectivity1to1('GC', 'Zone', transform=[1,2,3],
    ...     point_range=[[1,1],[1,10]], point_range_donor=[[5,5],[10,10]])
    >>> PT.print_tree(node)
    GC GridConnectivity1to1_t "Zone"
    ├───Transform "int[IndexDimension]" I4 [1 2 3]
    ├───PointRange IndexRange_t I4 [[ 1  1] [ 1 10]]
    └───PointRangeDonor IndexRange_t I4 [[ 5  5] [10 10]]
  """
  gc = new_node(name, 'GridConnectivity1to1_t', donor_name, [], parent)
  _check_parent_label(gc, parent, ['ZoneGridConnectivity_t'])
  if transform is not None:
    new_node('Transform', '"int[IndexDimension]"', transform, [], parent=gc)
  if point_range is not None:
    new_IndexRange('PointRange',      value=point_range,       parent=gc)
  if point_range_donor is not None:
    new_IndexRange('PointRangeDonor', value=point_range_donor, parent=gc)
  return gc

def new_IndexArray(name:str='PointList', value:ArrayLike=None, parent:CGNSTree=None):
  """ Create an IndexArray_t node

  Note that the value array will not be reshaped and must consequently match the expected layout
  ``(IndexDimension, N)``.

  Link to corresponding SIDS section:
  `IndexArray_t <https://cgns.github.io/CGNS_docs_current/sids/build.html#IndexArray>`_

  Args:
    name (str): Name of the created index array node
    value (ArrayLike) : value of the index array
    parent (CGNSTree): Node to which created node should be attached
  Example:
    >>> node = PT.new_IndexArray(value=[[1,2,3]])
    >>> PT.print_tree(node)
    PointList IndexArray_t I4 [[1 2 3]]
  """
  node = new_node(name, 'IndexArray_t', value, [], parent)
  allowed_parents = "BC_t BCDataSet_t DiscreteData_t FlowSolution_t GridConnectivity_t \
                     OversetHoles_t UserDefinedData_t ZoneSubRegion_t".split()
  _check_parent_label(node, parent, allowed_parents)
  return node

def new_IndexRange(name:str='PointRange', value:ArrayLike=None, parent:CGNSTree=None):
  """ Create an IndexRange_t node

  Note that if needed, the value array will be reshaped to the expected layout
  ``(IndexDimension, 2)`` (see example below).

  Link to corresponding SIDS section:
  `IndexRange_t <https://cgns.github.io/CGNS_docs_current/sids/build.html#IndexRange>`_

  Args:
    name (str): Name of the created index range node
    value (ArrayLike) : value of the index range
    parent (CGNSTree): Node to which created node should be attached
  Example:
    >>> node = PT.new_IndexRange(value=[[1,10],[1,10]])
    >>> PT.print_tree(node)
    PointRange IndexRange_t I4 [[ 1 10] [ 1 10]]
    >>> node = PT.new_IndexRange(value=[1,10, 1,10, 1,1])
    >>> PT.print_tree(node)
    PointRange IndexRange_t I4 [[ 1 10] [ 1 10] [ 1  1]]
  """
  _value = NA._convert_value(value)
  if _value is not None and _value.ndim == 1:
    _value = _value.reshape((-1,2))
  allowed_parents = "BC_t BCDataSet_t DiscreteData_t FlowSolution_t GridConnectivity_t \
                     GridConnectivity1to1_t OversetHoles_t UserDefinedData_t ZoneSubRegion_t".split()
  node = new_node(name, 'IndexRange_t', _value, [], parent)
  _check_parent_label(node, parent, allowed_parents)
  return node

def new_PointList(name:str='PointList', value:ArrayLike=None, parent:CGNSTree=None):
  warnings.warn("This function is deprecated in favor of PT.new_IndexArray", DeprecationWarning, stacklevel=2)
  return new_IndexArray(name, value, parent)

def new_PointRange(name:str='PointRange', value:ArrayLike=None, parent:CGNSTree=None):
  warnings.warn("This function is deprecated in favor of PT.new_IndexRange", DeprecationWarning, stacklevel=2)
  return new_IndexRange(name, value, parent)

def new_GridLocation(loc:str, parent:CGNSTree=None):
  """ Create a GridLocation_t node

  Link to corresponding SIDS section:
  `GridLocation_t <https://cgns.github.io/CGNS_docs_current/sids/build.html#GridLocation>`_

  Args:
    loc (str): Value to set in the grid location node
    parent (CGNSTree): Node to which created node should be attached
  Example:
    >>> node = PT.new_GridLocation('FaceCenter')
    >>> PT.print_tree(node)
    GridLocation GridLocation_t "FaceCenter"
  """
  assert loc in ['Null', 'UserDefined', 'Vertex', 'EdgeCenter', 'CellCenter',
      'IFaceCenter', 'JFaceCenter', 'KFaceCenter', 'FaceCenter']
  allowed_parents = "ArbitraryGridMotion_t BCDataSet_t BC_t DiscreteData_t FlowSolution_t \
                     GridConnectivity_t OversetHoles_t UserDefinedData_t ZoneSubRegion_t".split()
  node = new_node('GridLocation', 'GridLocation_t', loc, parent=parent)
  _check_parent_label(node, parent, allowed_parents)
  return node

def new_BaseIterativeData(name:str='BaseIterativeData', *, time_values:ArrayLike=None, iter_values:ArrayLike=None, parent:CGNSTree=None):
  """ Create a BaseIterativeData_t node

  Link to corresponding SIDS section:
  `BaseIterativeData_t <https://cgns.github.io/CGNS_docs_current/sids/timedep.html#IterativeData>`_

  Args:
    name (str): Name of the created node
    time_values (ArrayLike) : if provided, create a TimeValues child array
    iter_values (ArrayLike) : if provided, create an IterationValues child array
    parent (CGNSTree): Node to which created node should be attached
  Example:
    >>> node = PT.new_BaseIterativeData(time_values=[0.0, 0.5, 1.0])
    >>> PT.print_tree(node)
    BaseIterativeData BaseIterativeData_t I4 [3]
    └───TimeValues DataArray_t R4 [0.  0.5 1. ]
  """
  node = new_node(name, 'BaseIterativeData_t', parent=parent)
  n_steps = 0
  if time_values is not None:
    tv = new_DataArray("TimeValues", time_values, parent=node)
    n_steps = tv[1].size
  if iter_values is not None:
    tv = new_DataArray("IterationValues", iter_values, parent=node)
    n_steps = tv[1].size
  NA.set_value(node, n_steps)
  _check_parent_label(node, parent, ['CGNSBase_t'])
  return node

def new_DataArray(name:str, value:ArrayLike, *, dtype:DTypeLike=None, parent:CGNSTree=None):
  """ Create a DataArray_t node

  The datatype of the DataArray can be enforced with the ``dtype`` parameter, which 
  must be a str value (eg ``I4``, ``R8``). If not provided, default conversion of 
  :func:`~maia.pytree.set_value` applies.

  Link to corresponding SIDS section:
  `DataArray_t <https://cgns.github.io/CGNS_docs_current/sids/data.html#DataArray>`_

  Args:
    name (str): Name of the created data array node
    value (ArrayLike) : value of the data array
    dtype (DTypeLike) : If used, cast ``value`` to the specified type
    parent (CGNSTree): Node to which created data array should be attached
  Example:
    >>> node = PT.new_DataArray('Data', [1,2,3])
    >>> PT.print_tree(node)
    Data DataArray_t I4 [1 2 3]
    >>> node = PT.new_DataArray('Data', [1,2,3], dtype='R8')
    >>> PT.print_tree(node)
    Data DataArray_t R8 [1. 2. 3.]
  """

  allowed_parent = "\
    BCData_t BaseIterativeData_t ChemicalKineticsModel_t ConvergenceHistory_t DiscreteData_t Elements_t \
    EMConductivityModel_t EMElectricFieldModel_t EMMagneticFieldModel_t FlowSolution_t GasModel_t GridCoordinates_t \
    Periodic_t ReferenceState_t RigidGridMotion_t ThermalConductivityModel_t ThermalRelaxationModel_t TurbulenceClosure_t \
    TurbulenceModel_t UserDefinedData_t ViscosityModel_t ZoneIterativeData_t ZoneSubRegion_t".split()

  _value = NA._convert_value(value)
  if dtype is not None:
    _dtype = cgns_to_dtype[dtype]
    _value = _value.astype(_dtype)
  node = new_node(name, 'DataArray_t', _value, [], parent)
  _check_parent_label(node, parent, allowed_parent)
  return node


def new_GridCoordinates(name:str='GridCoordinates', *, fields:Dict[str,ArrayLike]={}, parent:CGNSTree=None):
  """ Create a GridCoordinates_t node

  Link to corresponding SIDS section:
  `GridCoordinates_t <https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#GridCoordinates>`_

  Args:
    name (str): Name of the created gc node
    fields (dict) : fields to create under the container (see :ref:`fields setting <pt_presets_commun>`)
    parent (CGNSTree): Node to which created gc should be attached
  Example:
    >>> coords={'CoordinateX' : [1.,2.,3.], 'CoordinateY' : [1.,1.,1.]}
    >>> node = PT.new_GridCoordinates(fields=coords)
    >>> PT.print_tree(node)
    GridCoordinates GridCoordinates_t 
    ├───CoordinateX DataArray_t R4 [1. 2. 3.]
    └───CoordinateY DataArray_t R4 [1. 1. 1.]
  """
  gc = new_node(name, 'GridCoordinates_t', parent=parent)
  _check_parent_label(gc, parent, ['Zone_t'])
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=gc)
  return gc

def new_FlowSolution(name:str = 'FlowSolution',
                     *,
                     loc:str = None,
                     fields:Dict[str, ArrayLike] = {},
                     parent:CGNSTree = None):
  """ Create a FlowSolution_t node

  Link to corresponding SIDS section:
  `FlowSolution_t <https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#FlowSolution>`_

  Args:
    name (str): Name of the created flow solution node
    loc (str) : If specified, create a GridLocation taking this value
    fields (dict) : fields to create under the container (see :ref:`fields setting <pt_presets_commun>`)
    parent (CGNSTree): Node to which created flow solution should be attached
  Example:
    >>> node = PT.new_FlowSolution('FS', loc='CellCenter',
    ...                            fields={'Density' : np.ones(125)})
    >>> PT.print_tree(node)
    FS FlowSolution_t 
    ├───GridLocation GridLocation_t "CellCenter"
    └───Density DataArray_t R8 (125,)
  """
  sol = new_node(name, 'FlowSolution_t', parent=parent)
  _check_parent_label(sol, parent, ['Zone_t'])
  if loc is not None:
    new_GridLocation(loc, sol)
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=sol)
  return sol

def new_ZoneSubRegion(name:str = 'ZoneSubRegion',
                      *,
                      loc:str = None,
                      point_range:ArrayLike = None,
                      point_list:ArrayLike = None,
                      bc_name:str = None,
                      gc_name:str = None,
                      family:str = None,
                      fields:Dict[str, ArrayLike] = {},
                      parent:CGNSTree = None):
  """ Create a ZoneSubRegion_t node

  The patch defining the ZoneSubRegion must be provided using one of ``point_range``, 
  ``point_list``, ``bc_name`` or ``gc_name`` parameter : they can no be used simultaneously.
  Setting a GridLocation with ``loc`` parameter makes sens only if a patch is explicitly defined
  with ``point_range`` or ``point_list``.

  Link to corresponding SIDS section:
  `ZoneSubRegion_t <https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#ZoneSubRegion>`_

  Args:
    name (str): Name of the created zsr node
    loc (str) : If specified, create a GridLocation taking this value
    point_range (ArrayLike) : PointRange array defining the ZSR extent
    point_list (ArrayLike) : PointList array defining the ZSR extent
    bc_name (str) : Name of the BC_t node defining the ZSR extent
    gc_name (str) : Name of the GC_t node defining the ZSR extent
    family (str) : If specified, create a FamilyName refering to this family
    fields (dict) : fields to create under the container (see :ref:`fields setting <pt_presets_commun>`)
    parent (CGNSTree): Node to which created zsr should be attached
  Example:
    >>> node = PT.new_ZoneSubRegion('Extraction', bc_name = 'Bottom',
    ...                             fields={'Density' : np.ones(125)})
    >>> PT.print_tree(node)
    Extraction ZoneSubRegion_t 
    ├───BCRegionName Descriptor_t "Bottom"
    └───Density DataArray_t R8 (125,)
    >>> node = PT.new_ZoneSubRegion('Probe', loc='CellCenter',
    ...                             point_list=[[104]])
    >>> PT.print_tree(node)
    Probe1 ZoneSubRegion_t 
    ├───GridLocation GridLocation_t "CellCenter"
    └───PointList IndexArray_t I4 [[104]]
  """
  zsr = new_node(name, 'ZoneSubRegion_t', None, [], parent)
  _check_parent_label(zsr, parent, ['Zone_t'])
  if loc is not None:
    new_GridLocation(loc, zsr)
  if family is not None:
    new_FamilyName(family, parent=zsr)
  if point_range is not None:
    assert point_list is None and bc_name is None and gc_name is None
    new_IndexRange('PointRange', point_range, zsr)
  if point_list is not None:
    assert point_range is None and bc_name is None and gc_name is None
    new_IndexArray('PointList', point_list, zsr)
  if bc_name is not None:
    assert point_list is None and point_range is None and gc_name is None
    new_node('BCRegionName', 'Descriptor_t', bc_name, parent=zsr)
  if gc_name is not None:
    assert point_list is None and point_range is None and bc_name is None
    new_node('GridConnectivityRegionName', 'Descriptor_t', gc_name, parent=zsr)
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=zsr)
  return zsr
