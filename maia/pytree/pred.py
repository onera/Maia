import fnmatch
import functools
import numpy as np

from maia.pytree.typing import *

import maia.pytree.cgns_keywords as CGK
from   maia.pytree      import node as N
from   maia.pytree      import walk as W
from   maia.pytree      import sids as S

def _escape_str(s):
  s = s.replace(']', '¤')
  s = s.replace('[', '[[]').replace('¤', '[]]')
  s = s.replace('?', '[?]')
  return s

def _fnmatch(s, target):
  return fnmatch.fnmatch(s, _escape_str(target)) if '*' in target else s == target

class NodePredicate:
  def __init__(self, func):
    self.func = func

  def __call__(self, X:CGNSTree) -> bool:
    return self.func(X)
  
  def __and__(self, other:"NodePredicate") -> "NodePredicate":
    return NodePredicate(lambda X : self(X) and other(X))

  def __or__(self, other:"NodePredicate") -> "NodePredicate":
    return NodePredicate(lambda X : self(X) or other(X))
  
  def __invert__(self) -> "NodePredicate":
    return NodePredicate(lambda X : not self(X))

_py_any = any
_py_all = all
def any(preds:Iterable[NodePredicate]) -> NodePredicate:
  return NodePredicate(lambda X : _py_any(pred(X) for pred in preds)) #type:ignore #(mypy confused because we erase builtin func)
def all(preds:Iterable[NodePredicate]) -> NodePredicate:
  return NodePredicate(lambda X : _py_all(pred(X) for pred in preds)) #type:ignore #(mypy confused because we erase builtin func)

def predicate_generator(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    return NodePredicate(lambda X: func(X, *args, **kwargs))
  return wrapper

def name_is(name: str) -> NodePredicate:
  """ Name of the node is exactly equal to the provided ``name`` """
  return NodePredicate(lambda n : n[0] == name)
def name_in(name_l:Iterable[str]) -> NodePredicate:
  """ Name of the node belongs to the provided ``name_l`` list """
  if isinstance(name_l, str):
    raise ValueError("Invalid type of argument `name_l` (expected list, got str). Use ``name_is`` instead")
  return NodePredicate(lambda n : n[0] in name_l)
def name_matches(name: str) -> NodePredicate:
  """ Name of the node matches the provided ``name``, for which wildcard ``*`` is accepted """
  name = _escape_str(name)
  return NodePredicate(lambda n : fnmatch.fnmatch(n[0], name))

def __value_is(n:CGNSTree, value) -> bool:
  if n[1] is None:
    return value is None
  elif value is None: #value is None and node[1] is not None
    return False
  else:
    _value = N.access._convert_value(value)
    assert _value is not None
    return np.array_equal(n[1], _value)
def value_is(value) -> NodePredicate:
  """ Value of the node is equal to the provided ``value`` """
  return NodePredicate(lambda n : __value_is(n, value))
def value_in(value_l) -> NodePredicate:
  """ Value of the node belongs to the provided ``value_l`` list """
  return any([value_is(val) for val in value_l])

def label_is(label) -> NodePredicate:
  """ Label of the node is exactly equal to the provided ``label`` """
  _label = label.name if isinstance(label, CGK.Label) else label
  return NodePredicate(lambda n : n[3] == _label)

def label_matches(label:str) -> NodePredicate:
  """ Label of the node matches the provided ``label``, for which wildcard ``*`` is accepted """
  label = _escape_str(label)
  return NodePredicate(lambda n : fnmatch.fnmatch(n[3], label))

def label_in(label_l:Iterable) -> NodePredicate:
  """ Label of the node belongs to the provided ``label_l`` list """
  if isinstance(label_l, (str, CGK.Label)):
    raise ValueError("Invalid type of argument `label_l` (expected list). Use ``label_is`` instead")
  _label_l = [label.name if isinstance(label, CGK.Label) else label for label in label_l]
  return NodePredicate(lambda n : n[3] in _label_l)

def has_child_of_name(child_name:str) -> NodePredicate:
  """ Node has a child whose name is exactly ``child_name`` """
  return NodePredicate(lambda n : W.get_child_from_predicate(n, name_is(child_name)) is not None)

def has_child_of_label(child_label) -> NodePredicate:
  """ Node has a child whose label is exactly ``child_label`` """
  return NodePredicate(lambda n : W.get_child_from_predicate(n, label_is(child_label)) is not None)

def has_location(loc:str) -> NodePredicate:
  """ Node allows a GridLocation child, and its value matches ``loc`` [1]_ """
  _ALLOW_LOC = label_in(["FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t",
    "BC_t", "BCDataSet_t", "GridConnectivity_t", "GridConnectivity1to1_t", "OversetHoles_t",
    "ArbitraryGridMotion_t", "UserDefinedData_t"])
  def get_loc(node:CGNSTree) -> str:
    gl = W.get_child_from_label(node, 'GridLocation_t')
    return N.get_str_value(gl) if gl is not None else 'Vertex'

  return _ALLOW_LOC & NodePredicate(lambda n: fnmatch.fnmatch(get_loc(n), loc))

def __belongs_to_family(n:CGNSTree, target_family:str, allow_additional=True):
  family_name_n = W.get_child_from_label(n, 'FamilyName_t')
  if family_name_n:
    fam_val = N.get_str_value(family_name_n)
    if _fnmatch(fam_val, target_family):
      return True
  if allow_additional:
    for additional_family_n in W.iter_children_from_label(n, 'AdditionalFamilyName_t'):
      fam_val = N.get_str_value(additional_family_n)
      if _fnmatch(fam_val, target_family):
        return True
  return False

def belongs_to_family(family:str, allow_additional=True) -> NodePredicate:
  """ Node has an (Additional)FamilyName_t [2]_ child whose value is ``family``
  (wildcard accepted) """
  return NodePredicate(lambda n : __belongs_to_family(n, family, allow_additional))

def is_bc_of_location(loc:str):
  """ Label of node is BC_t and its GridLocation value is ``loc`` """
  predicate = lambda n: N.get_label(n)=='BC_t' and S.Subset.GridLocation(n)==loc
  return NodePredicate(predicate)

def is_element_of_type(type:str):
  """ Label of node is Elements_t and its str type (*eg* ``QUAD_4``) is ``type`` """
  predicate = lambda n: N.get_label(n)=='Elements_t' and S.Element.Type(n)==type
  return NodePredicate(predicate)

def is_zone_of_kind(kind:Optional[str]=None, cell_dim:Optional[int]=None):
  """ Node is a Zone_t and its connectivity is described by ``kind``
  (one of ``S``, ``U``, ``Poly``, ``Std``, ``Particle``) and
  ``cell_dim`` (one of ``1``, ``2``, ``3``) [3]_
  """
  def _celldim_is(cell_dim:int):
    return NodePredicate(lambda z: (S.Zone.CellDimension(z) if N.get_label(z) == 'Zone_t' else 0) == cell_dim)

  if kind is None:
    pred = label_in({'Zone_t', 'ParticleZone_t'})
  elif kind == 'Particle':
    pred = label_is('ParticleZone_t')
  else:
    pred = label_is('Zone_t')
    if kind == 'S':
      pred &= NodePredicate(lambda z: S.Zone.Type(z) == 'Structured')
    else: # U, Poly, Std : zone need to be Unstructured
      pred &= NodePredicate(lambda z: S.Zone.Type(z) == 'Unstructured')
    if kind != 'U': # Poly or std
      elts_ok = NodePredicate(lambda z: _py_all(S.Element.Type(e) in
        ['BAR_2', 'NGON_n', 'NFACE_n'] for e in W.get_children_from_label(z, 'Elements_t')))
      is_poly = elts_ok & ~_celldim_is(1)
      if kind == 'Poly':
        pred &= is_poly
      elif kind == 'Std':
        pred &= ~is_poly

  if cell_dim is not None:
    pred &= _celldim_is(cell_dim)

  return pred

def is_gc_of_kind(is_1to1:Optional[bool]=None, is_perio:Optional[bool]=None):
  """ Label of node is GridConnectivity(1to1)_t and join is or not
  Abutting1to1 (resp periodic) depending of the boolean value of ``is_1to1`` (resp ``is_perio``) [4]_ """
  pred = label_in(['GridConnectivity_t', 'GridConnectivity1to1_t'])
  if is_1to1 is not None:
    pred &= NodePredicate(lambda n : S.GridConnectivity.is1to1(n) == is_1to1)
  if is_perio is not None:
    pred &= NodePredicate(lambda n : S.GridConnectivity.isperiodic(n) == is_perio)
  return pred


#: A predicate returning True, useful for example as a default value
ALWAYS_TRUE = NodePredicate(lambda X: True)
#: Node has a PointList or PointRange child
IS_SUBSET = has_child_of_name('PointList') | has_child_of_name('PointRange')
#: Label of node is either GridConnectivity_t or GridConnectivity1to1_t
IS_GC = label_in(['GridConnectivity_t', 'GridConnectivity1to1_t'])
#: Node is a Zone_t described by polyedric 2D elements
IS_POLY2D_ZONE = is_zone_of_kind('Poly', 2)
#: Node is a Zone_t described by polyedric 3D elements
IS_POLY3D_ZONE = is_zone_of_kind('Poly', 3)
