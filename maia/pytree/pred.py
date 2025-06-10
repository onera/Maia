import fnmatch
import functools
import numpy as np

from maia.pytree.typing import *

import maia.pytree.cgns_keywords as CGK
from   maia.pytree      import node as N
from   maia.pytree      import walk as W
from   maia.pytree      import sids as S

class UnaryPredicate:
  def __init__(self, func):
    self.func = func

  def __call__(self, X:CGNSTree) -> bool:
    return self.func(X)
  
  def __and__(self, other:"UnaryPredicate") -> "UnaryPredicate":
    return UnaryPredicate(lambda X : self(X) and other(X))

  def __or__(self, other:"UnaryPredicate") -> "UnaryPredicate":
    return UnaryPredicate(lambda X : self(X) or other(X))
  
  def __invert__(self) -> "UnaryPredicate":
    return UnaryPredicate(lambda X : not self(X))

_py_any = any
_py_all = all
def any(preds:Iterable[UnaryPredicate]) -> UnaryPredicate:
  return UnaryPredicate(lambda X : _py_any(pred(X) for pred in preds))
def all(preds:Iterable[UnaryPredicate]) -> UnaryPredicate:
  return UnaryPredicate(lambda X : _py_all(pred(X) for pred in preds))
def predicate_generator(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    return UnaryPredicate(lambda X: func(X, *args, **kwargs))
  return wrapper
def name_is(name: str) -> UnaryPredicate:
  """ Name of the node is exactly equal to the provided ``name`` """
  return UnaryPredicate(lambda n : n[0] == name)
def name_in(name_l) -> UnaryPredicate:
  """ Name of the node belongs to the provided ``name_l`` list """
  return UnaryPredicate(lambda n : n[0] in name_l)
def name_matches(name: str) -> UnaryPredicate:
  """ Name of the node matches the provided ``name``, for which wildcard ``*`` is accepted """
  return UnaryPredicate(lambda n : fnmatch.fnmatch(n[0], name))

def __value_is(n:CGNSTree, value) -> bool:
  if n[1] is None:
    return value is None
  elif value is None: #value is None and node[1] is not None
    return False
  else:
    _value = N.access._convert_value(value)
    assert _value is not None
    return np.array_equal(n[1], _value)
def value_is(value) -> UnaryPredicate:
  """ Value of the node is equal to the provided ``value`` """
  return UnaryPredicate(lambda n : __value_is(n, value))
def value_in(value_l) -> UnaryPredicate:
  """ Value of the node belongs to the provided ``value_l`` list """
  return any([value_is(val) for val in value_l])

def label_is(label) -> UnaryPredicate:
  """ Label of the node is exactly equal to the provided ``label`` """
  _label = label.name if isinstance(label, CGK.Label) else label
  return UnaryPredicate(lambda n : n[3] == _label)

def label_matches(label) -> UnaryPredicate:
  """ Label of the node matches the provided ``label``, for which wildcard ``*`` is accepted """
  if isinstance(label, CGK.Label):
    return UnaryPredicate(lambda n : n[3] == label.name)
  else:
    return UnaryPredicate(lambda n : fnmatch.fnmatch(n[3], label))

def label_in(label_l) -> UnaryPredicate:
  """ Label of the node belongs to the provided ``label_l`` list """
  _label_l = [label.name if isinstance(label, CGK.Label) else label for label in label_l]
  return UnaryPredicate(lambda n : n[3] in _label_l)

def has_child(child_name) -> UnaryPredicate:
  """ Node has a child whose name is exactly ``child_name`` """
  return UnaryPredicate(lambda n : W.get_child_from_predicate(n, name_is(child_name)) is not None)

def has_child_of_label(child_label) -> UnaryPredicate:
  """ Node has a child whose label is exactly ``child_label`` """
  return UnaryPredicate(lambda n : W.get_child_from_predicate(n, label_is(child_label)) is not None)

def has_location(loc:str) -> UnaryPredicate:
  """ Node allows a GridLocation child, and its value is ``loc`` [1]_ """
  _ALLOW_LOC = label_in(["FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t",
    "BC_t", "BCDataSet_t", "GridConnectivity_t", "GridConnectivity1to1_t", "OversetHoles_t",
    "ArbitraryGridMotion_t", "UserDefinedData_t"])
  def get_loc(node:CGNSTree) -> str:
    gl = W.get_child_from_label(node, 'GridLocation_t')
    return N.get_str_value(gl) if gl is not None else 'Vertex'

  return _ALLOW_LOC & UnaryPredicate(lambda n: get_loc(n) == loc)

def __belongs_to_family(n:CGNSTree, target_family:str, allow_additional=False):
  family_name_n = W.get_child_from_label(n, 'FamilyName_t')
  if family_name_n:
    fam_val = N.get_str_value(family_name_n)
    if fnmatch.fnmatch(fam_val, target_family):
      return True
  if allow_additional:
    for additional_family_n in W.iter_children_from_label(n, 'AdditionalFamilyName_t'):
      fam_val = N.get_str_value(additional_family_n)
      if fnmatch.fnmatch(fam_val, target_family):
        return True
  return False

def belongs_to_family(family:str, allow_additional=False) -> UnaryPredicate:
  """ Node has an (Additional)FamilyName_t [2]_ child whose value is ``family``
  (wildcard accepted) """
  return UnaryPredicate(lambda n : __belongs_to_family(n, family, allow_additional))

def is_bc_of_loc(loc):
  """ Label of node is BC_t and its GridLocation value is ``loc`` """
  predicate = lambda n: N.get_label(n)=='BC_t' and S.Subset.GridLocation(n)==loc
  return UnaryPredicate(predicate)

def is_elmt_of_type(cgns_name):
  """ Label of node is Elements_t and its CGNSName is ``cgns_name`` """
  predicate = lambda n: N.get_label(n)=='Elements_t' and S.Element.CGNSName(n)==cgns_name
  return UnaryPredicate(predicate)

def is_gc_with(match:Optional[bool]=None, perio:Optional[bool]=None):
  """ Label of node is GridConnectivity(1to1)_t and join is or not
  Abutting1to1 (resp periodic) depending of the value of ``match`` (resp ``perio``) [3]_ """
  pred = label_in(['GridConnectivity_t', 'GridConnectivity1to1_t'])
  if match is not None:
    pred = pred & UnaryPredicate(lambda n : S.GridConnectivity.is1to1(n) == match)
  if perio is not None:
    pred = pred & UnaryPredicate(lambda n : S.GridConnectivity.isperiodic(n) == perio)
  return pred

def _is_zone_of_celldim(cell_dim):
  return label_is('Zone_t') & UnaryPredicate(lambda z: S.Zone.CellDimension(z) == cell_dim)

HAS_POINTLIST = has_child('PointList') #: Node has a child named ``PointList``
IS_NGON_ELT = is_elmt_of_type('NGON_n') #: Node is an Elements_t node of type ``NGON_n``

#: Node is a Zone_t with structured connectivity
IS_S_ZONE = label_is('Zone_t') & UnaryPredicate(lambda z : S.Zone.Type(z) == 'Structured')
#: Node is a Zone_t with unstructured connectivity
IS_U_ZONE = label_is('Zone_t') & UnaryPredicate(lambda z : S.Zone.Type(z) == 'Unstructured')

#: Node is a Zone_t described by polyedric 2D elements
IS_POLY2D_ZONE = has_child_of_label('Elements_t') & _is_zone_of_celldim(2) \
  & UnaryPredicate(lambda z : _py_all(S.Element.CGNSName(e) in ['BAR_2', 'NGON_n'] \
                                  for e in W.get_children_from_label(z, 'Elements_t')))
 #: Node is a Zone_t described by polyedric 3D elements
IS_POLY3D_ZONE = has_child_of_label('Elements_t') & _is_zone_of_celldim(3) \
  & UnaryPredicate(lambda z: S.Zone.has_ngon_elements(z))
