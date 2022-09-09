from maia.pytree import walk
from maia.pytree import nodes_attr as NA

UNSET = Ellipsis

def new_node(name='Node', label='UserDefined_t', value=None, children=[], parent=None):
  """ Create a new node """
  node = ['Node', None, [], 'UserDefined_t']
  # Use update method to enable checks through the set_ functions
  update_node(node, name, label, value, children)
  if parent is not None:
    NA.add_child(parent, node)
  return node

def update_node(node, name=UNSET, label=UNSET, value=UNSET, children=UNSET):
  if name is not UNSET:
    NA.set_name(node, name)
  if label is not UNSET:
    NA.set_label(node, label)
  if value is not UNSET:
    NA.set_value(node, value)
  if children is not UNSET:
    NA.set_children(node, children)

# def create_child(parent, name, label='UserDefined_t', value=None, children=[]):
  # walk.rm_children_from_name(parent, name)
  # return new_node(name, label, value, children, parent)

def new_child(parent, name, label='UserDefined_t', value=None, children=[]):
  return new_node(name, label, value, children, parent)

def update_child(parent, name, label=UNSET, value=UNSET, children=UNSET):
  node = walk.get_child_from_name(parent, name)
  if node is None:
    node = new_node(name, parent=parent)
  update_node(node, ..., label, value, children)
  return node

def shallow_copy(t):
  out = [NA.get_name(t), NA.get_value(t, raw=True), [], NA.get_label(t)]
  for child in NA.get_children(t):
    out[2].append(shallow_copy(child))
  return out

def deep_copy(t):
  out = [NA.get_name(t), None, [], NA.get_label(t)]
  _val = NA.get_value(t, raw=True)
  if _val is not None:
    out[1] = _val.copy()
  for child in NA.get_children(t):
    out[2].append(deep_copy(child))
  return out



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
    new_node('ElementRange', 'ElementRange_t', erange, [], elem)
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
  _value = NA.convert_value(value)
  if _value is not None and _value.ndim == 1:
    _value = _value.reshape((-1,2))
  return new_node(name, 'IndexRange_t', _value, [], parent)

def new_GridLocation(loc, parent=None):
  assert loc in ['Null', 'UserDefined', 'Vertex', 'EdgeCenter', 'CellCenter',
      'IFaceCenter', 'JFaceCenter', 'KFaceCenter', 'FaceCenter']
  return new_node('GridLocation', 'GridLocation_t', loc, parent=parent)

def new_DataArray(name, value, *, dtype=None, parent=None):
  _value = NA.convert_value(value)
  if dtype is not None:
    print('Not yet implemented') #TODO : DataConversion
  return new_node(name, 'DataArray_t', _value, [], parent)

def new_FlowSolution(name='FlowSolution', *, loc=None, fields={}, parent=None):
  sol = new_node(name, 'FlowSolution_t', parent=parent)
  if loc is not None:
    new_GridLocation(loc, sol)
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=sol)
  return sol
