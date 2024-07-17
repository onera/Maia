import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT

from maia.algo import geometry_utils as GU

def test_get_or_create_container():
  zone = PT.new_Zone(type='Unstructured')
  fields = {'SolA' : np.ones(10)}

  cont = GU.get_or_create_container(zone, 'MyContainer', 'CellCenter', fields)
  assert PT.get_name(cont) == 'MyContainer' and PT.get_label(cont) == 'DiscreteData_t'
  assert PT.Subset.GridLocation(cont) == 'CellCenter'
  assert PT.get_child_from_name_and_label(cont, 'SolA', 'DataArray_t')[1].size == 10

  # Container should not be erased
  cont2 = GU.get_or_create_container(zone, 'MyContainer', 'CellCenter')
  assert PT.is_same_tree(cont, cont2)

  fields = {'SolA' : np.ones(15), 'SolB' : np.ones(15)}
  cont3 = GU.get_or_create_container(zone, 'MyContainer', 'CellCenter', fields)
  assert PT.get_child_from_name_and_label(cont3, 'SolA', 'DataArray_t')[1].size == 15
  assert PT.get_child_from_name_and_label(cont3, 'SolB', 'DataArray_t')[1].size == 15

  with pytest.raises(RuntimeError):
    cont4 = GU.get_or_create_container(zone, 'MyContainer', 'FaceCenter')

