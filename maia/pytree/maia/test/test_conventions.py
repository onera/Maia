import pytest

from maia.pytree.maia import conventions as conv

def test_add_part_suffix():
  assert conv.add_part_suffix('zone', 9, 3) == 'zone.P9.N3'
  assert conv.add_part_suffix('zone.with.dot', 212, 88) == 'zone.with.dot.P212.N88'

def test_get_part_prefix():
  assert conv.get_part_prefix('zone.P5.N98') == 'zone'
  assert conv.get_part_prefix('zone.P1.N0') == 'zone'
  with pytest.raises(AssertionError):
    conv.get_part_prefix('zone')

def test_get_part_suffix():
  assert conv.get_part_suffix('zone.P5.N98') == (5,98)
  assert conv.get_part_suffix('zone.P1.N0') == (1,0)
  with pytest.raises(AssertionError):
    conv.get_part_suffix('zone')

def test_add_split_suffix():
  assert conv.add_split_suffix('match', 2) == 'match.2'

def test_get_split_prefix():
  assert conv.get_split_prefix('myjoin.5') == 'myjoin'
  assert conv.get_split_prefix('myjoin.5.9') == 'myjoin.5'
  with pytest.raises(AssertionError):
    conv.get_split_prefix('myjoin')

def test_name_intra_gc():
  assert conv.name_intra_gc(4,8,0,12) == 'JN.P4.N8.LT.P0.N12'

def test_is_intra_gc():
  assert conv.is_intra_gc('JN.P219.N0192.LT.P0.N11') == True
  assert conv.is_intra_gc('myjoin.5') == False
