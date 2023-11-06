from maia.pytree.typing import *

from maia.pytree.cgns_keywords import cgns_to_dtype

from maia.pytree.node import access as NA
from maia.pytree.node import new_node

# Specialized
def new_CGNSTree(*, version:float=4.2):
  version = new_node('CGNSLibraryVersion', 'CGNSLibraryVersion_t', value=version)
  return new_node('CGNSTree', label='CGNSTree_t', children=[version])

def new_CGNSBase(name:str='Base', *, cell_dim:int=3, phy_dim:int=3, parent:CGNSTree=None):
  return new_node(name, 'CGNSBase_t', value=[cell_dim, phy_dim], parent=parent)

def new_Family(name:str='Family', *, family_bc:str=None, parent:CGNSTree=None):
  family = new_node(name, 'Family_t', None, [], parent=parent)
  if family_bc is not None:
    new_FamilyBC(family_bc, family)
  return family

def new_FamilyName(family_name:str, parent:CGNSTree=None):
  return new_node('FamilyName', 'FamilyName_t', family_name, [], parent)

def new_FamilyBC(family_bc:str, parent:CGNSTree=None):
  allowed_bc = """Null UserDefined BCAxisymmetricWedge BCDegenerateLine BCDegeneratePoint BCDirichlet BCExtrapolate
  BCFarfield BCGeneral BCInflow BCInflowSubsonic BCInflowSupersonic BCNeumann BCOutflow BCOutflowSubsonic 
  BCOutflowSupersonic BCSymmetryPlane BCSymmetryPolar BCTunnelInflow BCTunnelOutflow BCWall BCWallInviscid
  BCWallViscous BCWallViscousHeatFlux BCWallViscousIsothermal FamilySpecified""".split()
  assert family_bc in allowed_bc
  return new_node('FamilyBC', 'FamilyBC_t', family_bc, [], parent)

def new_Zone(name:str='Zone', *, type:str='Null', size:ArrayLike=None, family:str=None, parent:CGNSTree=None):
  assert type in ['Null', 'UserDefined', 'Structured', 'Unstructured']
  zone = new_node(name, 'Zone_t', size, [], parent)
  zone_type = new_node('ZoneType', 'ZoneType_t', type, [], zone)
  if family:
    new_FamilyName(family, zone)
  return zone

def new_Elements(name:str='Elements', type:str='Null', *, erange:ArrayLike=None, econn:ArrayLike=None, parent:CGNSTree=None):
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

def new_NGonElements(name:str = 'NGonElements',
                     *,
                     erange:ArrayLike = None,
                     eso:ArrayLike = None,
                     ec:ArrayLike = None,
                     pe:ArrayLike = None,
                     pepos:ArrayLike = None,
                     parent:CGNSTree = None):
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
  elem = new_Elements(name, 23, erange=erange, parent=parent)
  for name, val in zip(['ElementStartOffset', 'ElementConnectivity'], [eso, ec]):
    if val is not None:
      new_DataArray(name, val, parent=elem)
  return elem

def new_ZoneBC(parent:CGNSTree=None):
  return new_node('ZoneBC', 'ZoneBC_t', None, [], parent)

def new_BC(name:str = 'BC',
           type:str='Null',
           *,
           point_range:ArrayLike = None,
           point_list:ArrayLike = None,
           loc:str = None,
           family:str = None,
           parent:CGNSTree = None):
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

def new_ZoneGridConnectivity(name:str='ZoneGridConnectivity', parent:CGNSTree=None):
  return new_node(name, 'ZoneGridConnectivity_t', None, [], parent)

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
  gc_props = new_node('GridConnectivityProperty', 'GridConnectivityProperty_t', None, [], parent)
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
  gc = new_node(name, 'GridConnectivity1to1_t', donor_name, [], parent)
  if transform is not None:
    new_node('Transform', '"int[IndexDimension]"', transform, [], parent=gc)
  if point_range is not None:
    new_PointRange('PointRange',      value=point_range,       parent=gc)
  if point_range_donor is not None:
    new_PointRange('PointRangeDonor', value=point_range_donor, parent=gc)
  return gc

def new_PointList(name:str='PointList', value:ArrayLike=None, parent:CGNSTree=None):
  return new_node(name, 'IndexArray_t', value, [], parent)

def new_PointRange(name:str='PointRange', value:ArrayLike=None, parent:CGNSTree=None):
  _value = NA._convert_value(value)
  if _value is not None and _value.ndim == 1:
    _value = _value.reshape((-1,2))
  return new_node(name, 'IndexRange_t', _value, [], parent)

def new_GridLocation(loc:str, parent:CGNSTree=None):
  assert loc in ['Null', 'UserDefined', 'Vertex', 'EdgeCenter', 'CellCenter',
      'IFaceCenter', 'JFaceCenter', 'KFaceCenter', 'FaceCenter']
  return new_node('GridLocation', 'GridLocation_t', loc, parent=parent)

def new_DataArray(name:str, value:ArrayLike, *, dtype:DTypeLike=None, parent:CGNSTree=None):
  _value = NA._convert_value(value)
  if dtype is not None:
    _dtype = cgns_to_dtype[dtype]
    _value = _value.astype(_dtype)
  return new_node(name, 'DataArray_t', _value, [], parent)

def new_GridCoordinates(name:str='GridCoordinates', *, fields:Dict[str,ArrayLike]={}, parent:CGNSTree=None):
  gc = new_node(name, 'GridCoordinates_t', parent=parent)
  for field_name, field_val in fields.items():
    new_DataArray(field_name, field_val, parent=gc)
  return gc

def new_FlowSolution(name:str = 'FlowSolution',
                     *,
                     loc:str = None,
                     fields:Dict[str, ArrayLike] = {},
                     parent:CGNSTree = None):
  sol = new_node(name, 'FlowSolution_t', parent=parent)
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
