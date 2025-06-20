import pytest
import numpy as np

from maia.pytree.cgns_keywords import Label as CGL

from maia.pytree.walk import auto_pred as AP

def partial_funcs_equal(f1, f2):
  return all([getattr(f1, attr) == getattr(f2, attr) for attr in ['func']])

def test_auto_predicate():
  nface = ['NFace', np.array([23, 0], np.int32), [], 'Elements_t']

  assert AP.auto_predicate('NFace')(nface)
  assert AP.auto_predicate('Elements_t')(nface)
  assert AP.auto_predicate(CGL.Elements_t)(nface)
  assert not AP.auto_predicate('Element_t')(nface)
  assert AP.auto_predicate(np.array([23,0]))(nface)
  assert AP.auto_predicate(lambda n : True)(nface)
  assert AP.auto_predicate(lambda n : len(n[0]) == 5)(nface)
  assert not AP.auto_predicate(lambda n : False)(nface)

  with pytest.raises(TypeError):
    AP.auto_predicate(123)

"""
def test_auto_predicates():
  auto_predicates = AP.auto_predicates(['Base', 'Zone'])
  assert partial_funcs_equal(auto_predicates[0], AP.auto_predicate('Base'))
  assert partial_funcs_equal(auto_predicates[1], AP.auto_predicate('Zone'))
  auto_predicates = auto_predicates('ZoneBC_t/BC_t')
  assert partial_funcs_equal(auto_predicates[0], AP.auto_predicate('ZoneBC_t'))
  assert partial_funcs_equal(auto_predicates[1], AP.auto_predicate('BC_t'))

  with pytest.raises(TypeError):
    AP.auto_predicates(123)

"""
