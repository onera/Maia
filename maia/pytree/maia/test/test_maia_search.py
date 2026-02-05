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

def test_get_partitioned_zones():
  pt = """
  BaseA CGNSBase_t:
    Zone1.P0.N1 Zone_t:
    Zone1.P0.N2 Zone_t:
    Zone2.With.dot.P0.N0 Zone_t:
  BaseB CGNSBase_t:
    Zone3.P0.N0 Zone_t:
  """
  part_tree = PT.yaml.to_cgns_tree(pt)
  get_names = lambda nodes : [PT.get_name(n) for n in nodes]
  assert get_names(MT.get_partitioned_zones(part_tree, 'BaseA/Zone1')) == ['Zone1.P0.N1', 'Zone1.P0.N2']
  assert get_names(MT.get_partitioned_zones(part_tree, 'BaseA/Zone2.With.dot')) == ['Zone2.With.dot.P0.N0']
  assert get_names(MT.get_partitioned_zones(part_tree, 'BaseA/Zone3')) == []
  assert get_names(MT.get_partitioned_zones(part_tree, 'BaseB/Zone3')) == ['Zone3.P0.N0']
  base = PT.find_child_from_name(part_tree, 'BaseA')
  assert get_names(MT.get_partitioned_zones(base, 'Zone1')) == ['Zone1.P0.N1', 'Zone1.P0.N2']
  assert get_names(MT.get_partitioned_zones(base, 'Zone2.With.dot')) == ['Zone2.With.dot.P0.N0']
  assert get_names(MT.get_partitioned_zones(base, 'Zone3')) == []

