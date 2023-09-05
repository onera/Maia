from maia.pytree.cgns_keywords import cgns_to_dtype

from maia.pytree.node import access as NA
from maia.pytree.node import new_node

# Specialized
def new_CGNSTree(*, version=4.2):
  version = new_node('CGNSLibraryVersion', 'CGNSLibraryVersion_t', value=version)
  return new_node('CGNSTree', label='CGNSTree_t', children=[version])

def new_CGNSBase(name='Base', *, cell_dim=3, phy_dim=3, parent=None):
  return new_node(name, 'CGNSBase_t', value=[cell_dim, phy_dim], parent=parent)

def new_Family(name='Family', *, family_bc=None, parent=None):
  family = new_node(name, 'Family_t', None, [], parent=parent)
  if family_bc is not None:
    new_FamilyBC(family_bc, family)
  return family

def new_FamilyName(family_name, parent=None):
  return new_node('FamilyName', 'FamilyName_t', family_name, [], parent)

def new_FamilyBC(family_bc, parent=None):
  allowed_bc = """Null UserDefined BCAxisymmetricWedge BCDegenerateLine BCDegeneratePoint BCDirichlet BCExtrapolate
  BCFarfield BCGeneral BCInflow BCInflowSubsonic BCInflowSupersonic BCNeumann BCOutflow BCOutflowSubsonic 
  BCOutflowSupersonic BCSymmetryPlane BCSymmetryPolar BCTunnelInflow BCTunnelOutflow BCWall BCWallInviscid
  BCWallViscous BCWallViscousHeatFlux BCWallViscousIsothermal FamilySpecified""".split()
  assert family_bc in allowed_bc
  return new_node('FamilyBC', 'FamilyBC_t', family_bc, [], parent)

def new_Zone(name='Zone', *, type='Null', size=None, family=None, parent=None):
  assert type in ['Null', 'UserDefined', 'Structured', 'Unstructured']
  zone = new_node(name, 'Zone_t', size, [], parent)
  zone_type = new_node('ZoneType', 'ZoneType_t', type, [], zone)
  if family:
    new_FamilyName(family, zone)
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

def new_NFaceElements(name='NFaceElements', *, erange=None, eso=None, ec=None, parent=None):
  elem = new_Elements(name, 23, erange=erange, parent=parent)
  for name, val in zip(['ElementStartOffset', 'ElementConnectivity'], [eso, ec]):
    if val is not None:
      new_DataArray(name, val, parent=elem)
  return elem

def new_ZoneBC(parent=None):
  return new_node('ZoneBC', 'ZoneBC_t', None, [], parent)

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
    new_FamilyName(family, bc)
  if point_range is not None:
    assert point_list is None
    new_PointRange('PointRange', point_range, bc)
  if point_list is not None:
    assert point_range is None
    new_PointList('PointList', point_list, bc)
  return bc

def new_ZoneGridConnectivity(name='ZoneGridConnectivity', parent=None):
  return new_node(name, 'ZoneGridConnectivity_t', None, [], parent)

def new_GridConnectivity(name='GC', donor_name=None, type='Null', *, loc=None, \
    point_range=None, point_range_donor=None, point_list=None, point_list_donor=None, parent=None):
  gc = new_node(name, 'GridConnectivity_t', donor_name, [], parent)
  new_GridConnectivityType(type, parent=gc)
  if loc is not None:
    new_GridLocation(loc, gc)
  if point_range is not None:
    assert point_list is None
    new_PointRange('PointRange', value=point_range, parent=gc)
  if point_list is not None:
    assert point_range is None
    new_PointList('PointList', value=point_list, parent=gc)
  if point_range_donor is not None:
    assert point_list_donor is None
    new_PointRange('PointRangeDonor', value=point_range_donor, parent=gc)
  if point_list_donor is not None:
    assert point_range_donor is None
    new_PointList('PointListDonor', value=point_list_donor, parent=gc)
  return gc

def new_GridConnectivityType(type="Null", parent=None):
  allowed_gc = "Null UserDefined Overset Abutting Abutting1to1".split()
  assert type in allowed_gc
  return new_node('GridConnectivityType', 'GridConnectivityType_t', type, [], parent)

def new_Periodic(rotation_angle=[0., 0., 0.], rotation_center=[0., 0., 0.], translation=[0.,0.,0], parent=None):
  childs = [
      new_DataArray('RotationAngle', rotation_angle),
      new_DataArray('RotationCenter', rotation_center),
      new_DataArray('Translation', translation)
      ]
  return new_node('Periodic', 'Periodic_t', None, childs, parent)

def new_GridConnectivityProperty(periodic={}, parent=None):
  gc_props = new_node('GridConnectivityProperty', 'GridConnectivityProperty_t', None, [], parent)
  if periodic:
    new_Periodic(**periodic, parent=gc_props)
  return gc_props

def new_GridConnectivity1to1(name='GC', donor_name=None, *, point_range=None, \
    point_range_donor=None, transform=None, parent=None):
  gc = new_node(name, 'GridConnectivity1to1_t', donor_name, [], parent)
  if transform is not None:
    new_node('Transform', '"int[IndexDimension]"', transform, [], parent=gc)
  if point_range is not None:
    new_PointRange('PointRange',      value=point_range,       parent=gc)
  if point_range_donor is not None:
    new_PointRange('PointRangeDonor', value=point_range_donor, parent=gc)
  return gc

def new_PointList(name='PointList', value=None, parent=None):
  return new_node(name, 'IndexArray_t', value, [], parent)

def new_PointRange(name='PointRange', value=None, parent=None):
  _value = NA._convert_value(value)
  if _value is not None and _value.ndim == 1:
    _value = _value.reshape((-1,2))
  return new_node(name, 'IndexRange_t', _value, [], parent)

def new_GridLocation(loc, parent=None):
  assert loc in ['Null', 'UserDefined', 'Vertex', 'EdgeCenter', 'CellCenter',
      'IFaceCenter', 'JFaceCenter', 'KFaceCenter', 'FaceCenter']
  return new_node('GridLocation', 'GridLocation_t', loc, parent=parent)

def new_DataArray(name, value, *, dtype=None, parent=None):
  _value = NA._convert_value(value)
  if dtype is not None:
    _dtype = cgns_to_dtype[dtype]
    _value = _value.astype(_dtype)
  return new_node(name, 'DataArray_t', _value, [], parent)

def new_GridCoordinates(name='GridCoordinates', *, fields={}, parent=None):
  gc = new_node(name, 'GridCoordinates_t', parent=parent)
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=gc)
  return gc

def new_FlowSolution(name='FlowSolution', *, loc=None, fields={}, parent=None):
  sol = new_node(name, 'FlowSolution_t', parent=parent)
  if loc is not None:
    new_GridLocation(loc, sol)
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=sol)
  return sol

def new_ZoneSubRegion(name='ZoneSubRegion', *, loc=None, point_range=None, point_list=None, bc_name=None, gc_name=None, \
    family=None, fields={}, parent=None):
  zsr = new_node(name, 'ZoneSubRegion_t', None, [], parent)
  if loc is not None:
    new_GridLocation(loc, zsr)
  if family is not None:
    new_FamilyName(family, zsr)
  if point_range is not None:
    assert point_list is None and bc_name is None and gc_name is None
    new_PointRange('PointRange', point_range, zsr)
  if point_list is not None:
    assert point_range is None and bc_name is None and gc_name is None
    new_PointList('PointList', point_list, zsr)
  if bc_name is not None:
    assert point_list is None and point_range is None and gc_name is None
    new_node('BCRegionName', 'Descriptor_t', bc_name, parent=zsr)
  if gc_name is not None:
    assert point_list is None and point_range is None and bc_name is None
    new_node('GridConnectivityRegionName', 'Descriptor_t', gc_name, parent=zsr)
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=zsr)
  return zsr
