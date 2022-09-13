from maia.pytree.cgns_keywords import cgns_to_dtype

from maia.pytree      import node as N
from maia.pytree.node import new_node

# Specialized
def new_CGNSTree(*, version=4.2):
  version = new_node('CGNSLibraryVersion', 'CGNSLibraryVersion_t', value=version)
  return new_node('CGNSTree', label='CGNSTree_t', children=[version])

def new_CGNSBase(name='Base', *, cell_dim=3, phy_dim=3, parent=None):
  return new_node(name, 'CGNSBase_t', value=[cell_dim, phy_dim], parent=parent)

def new_Zone(name='Zone', *, type='Null', size=None, family=None, parent=None):
  assert type in ['Null', 'UserDefined', 'Structured', 'Unstructured']
  zone = new_node(name, 'Zone_t', size, [], parent)
  zone_type = new_node('ZoneType', 'ZoneType_t', type, [], zone)
  if family:
    family = new_node('FamilyName', 'FamilyName_t', family, [], zone)
  return zone

def new_Elements(name='Elements', type='Null', *, erange=None, econn=None, parent=None):
  from maia.pytree.sids import elements_utils as EU
  if isinstance(type, str):
    _value = [EU.cgns_name_to_id(type), 0]
  elif isinstance(type, int):
    _value = [type, 0]
  else:
    _value = type # Try autoconversion in new_node
  elem = new_node(name, "Elements_t", _value, [], parent)
  if erange is not None:
    new_node('ElementRange', 'IndexRange_t', erange, [], elem)
  if econn is not None:
    new_DataArray('ElementConnectivity', econn, parent=elem)
  return elem

def new_NGonElements(name='NGonElements', *, erange=None, eso=None, ec=None, pe=None, pepos=None, parent=None):
  elem = new_Elements(name, 22, erange=erange, parent=parent)
  names = ['ElementStartOffset', 'ElementConnectivity', 'ParentElements', 'ParentElementsPosition']
  for name, val in zip(names, [eso, ec, pe, pepos]):
    if val is not None:
      new_DataArray(name, val, parent=elem)
  return elem

def new_NFaceElements(name='NGonElements', *, erange=None, eso=None, ec=None, parent=None):
  elem = new_Elements(name, 23, erange=erange, parent=parent)
  for name, val in zip(['ElementStartOffset', 'ElementConnectivity'], [eso, ec]):
    if val is not None:
      new_DataArray(name, val, parent=elem)
  return elem


def new_BC(name='BC', type='Null', *, point_range=None, point_list=None, loc=None, family=None, parent=None):
  allowed_bc = """Null UserDefined BCAxisymmetricWedge BCDegenerateLine BCDegeneratePoint BCDirichlet BCExtrapolate
  BCFarfield BCGeneral BCInflow BCInflowSubsonic BCInflowSupersonic BCNeumann BCOutflow BCOutflowSubsonic 
  BCOutflowSupersonic BCSymmetryPlane BCSymmetryPolar BCTunnelInflow BCTunnelOutflow BCWall BCWallInviscid
  BCWallViscous BCWallViscousHeatFlux BCWallViscousIsothermal FamilySpecified""".split()
  assert type in allowed_bc
  
  bc = new_node(name, 'BC_t', type, [], parent)
  if loc is not None:
    new_GridLocation(loc, bc)
  if family is not None:
    new_node('FamilyName', 'FamilyName_t', family, [], bc)
  if point_range is not None:
    assert point_list is None
    new_PointRange('PointRange', point_range, bc)
  if point_list is not None:
    assert point_range is None
    new_PointList('PointList', point_list, bc)
  return bc

def new_PointList(name='PointList', value=None, parent=None):
  return new_node(name, 'IndexArray_t', value, [], parent)

def new_PointRange(name='PointRange', value=None, parent=None):
  _value = N.convert_value(value)
  if _value is not None and _value.ndim == 1:
    _value = _value.reshape((-1,2))
  return new_node(name, 'IndexRange_t', _value, [], parent)

def new_GridLocation(loc, parent=None):
  assert loc in ['Null', 'UserDefined', 'Vertex', 'EdgeCenter', 'CellCenter',
      'IFaceCenter', 'JFaceCenter', 'KFaceCenter', 'FaceCenter']
  return new_node('GridLocation', 'GridLocation_t', loc, parent=parent)

def new_DataArray(name, value, *, dtype=None, parent=None):
  _value = N.convert_value(value)
  if dtype is not None:
    _dtype = cgns_to_dtype[dtype]
    _value = _value.astype(_dtype)
  return new_node(name, 'DataArray_t', _value, [], parent)

def new_FlowSolution(name='FlowSolution', *, loc=None, fields={}, parent=None):
  sol = new_node(name, 'FlowSolution_t', parent=parent)
  if loc is not None:
    new_GridLocation(loc, sol)
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=sol)
  return sol
