import pytest
import maia.pytree        as PT
import maia.pytree.maia   as MT


def test_get_Distribution():
  zone = PT.new_Zone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = MT.new_Distribution(distri_arrays, zone)
  assert MT.get_Distribution(zone) is distri
  assert (MT.Zone.cell_distribution(zone) == [0,15,30]).all()
  assert (MT.Zone.vtx_distribution(zone) == [100,1000,1000]).all()

def test_get_GlobalNumbering():
  zone = PT.new_Zone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = MT.new_GlobalNumbering(gnum_arrays, zone)
  assert MT.get_GlobalNumbering(zone) is gnum_node
  assert (PT.get_value(MT.get_GlobalNumbering(zone, 'Cell')) == [4,21,1,2,8,12]).all()
  assert  PT.get_value(MT.get_GlobalNumbering(zone, 'Vertex')) == None

