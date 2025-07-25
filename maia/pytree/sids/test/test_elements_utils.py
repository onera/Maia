import pytest

from maia.pytree.sids import elements_utils as EU

def test_id_to_name():
  assert EU.id_to_name(5)  == "TRI_3"
  assert EU.id_to_name(38) == "HEXA_56"
  with pytest.raises(AssertionError):
    EU.id_to_name(1000)

def test_name_to_id():
  assert EU.name_to_id('Null') == 0
  assert EU.name_to_id('HEXA_64') == 39
  assert EU.name_to_id('HEXA_64') == 39
  with pytest.raises(ValueError):
    EU.name_to_id('NOTINLIST')

def test_id_to_dim():
  assert EU.id_to_dim(5)  == 2
  assert EU.id_to_dim(38) == 3
  with pytest.raises(AssertionError):
    EU.id_to_dim(1000)

def test_name_to_dim():
  assert EU.name_to_dim('TETRA_4') == 3

def test_id_to_nvtx():
  assert EU.id_to_nvtx(5)  == 3
  assert EU.id_to_nvtx(38) == 56
  with pytest.raises(AssertionError):
    EU.id_to_nvtx(1000)

def test_name_to_nvtx():
  assert EU.name_to_nvtx('PENTA_66') == 66
