import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import agglomeration as AGL

def test_path_to_level():
  assert AGL.path_to_level('Base.LV13/Zone/ZoneBC') == 13
  assert AGL.path_to_level('Base.42.LV245.LV42') == 42
  with pytest.raises(IndexError):
    AGL.path_to_level('Zone/FlowSolution')

def test_update_path_level():
  assert AGL.update_path_level('Base.LV13/Zone/ZoneBC/Xmin', 42) == 'Base.LV42/Zone/ZoneBC/Xmin'
  assert AGL.update_path_level('Base.42.LV245.LV42', 7) == 'Base.42.LV245.LV7'

def test_n_level():
  tree = PT.yaml.to_cgns_tree("""
  Base.LV0 CGNSBase_t [3, 3]:
  Base.LV1 CGNSBase_t [3, 3]:
  Base.LV2 CGNSBase_t [3, 3]:
  """)
  assert AGL.n_level(tree) == 2

  with pytest.raises(IndexError):
    tree = PT.yaml.to_cgns_tree("""
    Base CGNSBase_t [3, 3]:
    """)
    AGL.n_level(tree)

def test_single_level_tree():
  tree = PT.yaml.to_cgns_tree("""
  Base.LV0 CGNSBase_t [3, 3]:
    LargeZone Zone_t:
    SmallZone Zone_t:
  Base.LV1 CGNSBase_t [3, 3]:
    LargeZone Zone_t:
    SmallZone Zone_t:
  Base.LV2 CGNSBase_t [3, 3]:
    LargeZone Zone_t:
    SmallZone Zone_t:
  """)

  tree1 = AGL.single_level_tree(tree, 1)

  expected = PT.yaml.to_cgns_tree("""
  Base.LV1 CGNSBase_t [3, 3]:
    LargeZone Zone_t:
    SmallZone Zone_t:
  """)
  assert PT.is_same_tree(tree1, expected)

  tree18 = AGL.single_level_tree(tree, 18)
  assert len(PT.get_all_CGNSBase_t(tree18)) == 0


def test_suffix_bases():
  tree = PT.yaml.to_cgns_tree("""
  SomeBase CGNSBase_t [3, 3]:
    Zone Zone_t:
      ZoneGridConnectivity ZoneGridConnectivity_t:
        jnintra GridConnectivity_t "Zone":
  SomeOtherBase CGNSBase_t [3, 3]:
    OtherZone Zone_t:
      ZoneGridConnectivity ZoneGridConnectivity_t:
        jn GridConnectivity_t "SomeBase/Zone":
  """)
  AGL._suffix_bases(tree, '.LV8')

  assert [PT.get_name(b) for b in PT.get_all_CGNSBase_t(tree)] == \
    ['SomeBase.LV8', 'SomeOtherBase.LV8']
  assert PT.get_value(PT.find_node_from_name(tree, 'jn')) == "SomeBase.LV8/Zone"
  assert PT.get_value(PT.find_node_from_name(tree, 'jnintra')) == "Zone"

def test_slab_half_size():
  assert AGL._slab_half_size([[0, 64], [16, 32], [0, 1]]) == 256
  assert AGL._slab_half_size([[1, 64], [16, 32], [0, 1]]) == 256 - 8
  assert AGL._slab_half_size([[2, 64], [16, 32], [0, 1]]) == 256 - 8
  assert AGL._slab_half_size([[0, 64], [17, 32], [0, 1]]) == 256 - 32

@pytest_parallel.mark.parallel(2)
def test_multigrid_s_2D(comm):
  n_lvl = 3
  coeff = 2**n_lvl
  n1 = 8
  n2 = 4
  tree = maia.factory.generate_dist_block([coeff*n1+1,coeff*n2+1], "S", comm)

  # Split a bc in two for test
  zbc = PT.find_node_from_name(tree, 'ZoneBC')
  bc1 = PT.find_child_from_name(zbc, 'Xmax')
  bc2 = PT.deep_copy(bc1)
  pr1 = PT.get_np_value(PT.find_child_from_name(bc1, 'PointRange'))
  pr2 = PT.get_np_value(PT.find_child_from_name(bc2, 'PointRange'))
  PT.set_name(bc1, 'Xmax_1')
  PT.set_name(bc2, 'Xmax_2')
  pr1[1,1] = coeff + 1
  pr2[1,0] = coeff + 1
  others = PT.get_children_from_predicate(zbc, ~PT.pred.name_matches('Xmax*'))
  PT.set_children(zbc, others + [bc1, bc2])


  AGL.agglomerate_cells(tree, n_lvl, comm)

  # Check computed coarse index (on thin grid): we compute actual cell idx
  # on coarse mesh, and move it on thin mesh by interpolation to compare.
  for i in range(n_lvl):
    thin = PT.shallow_copy(tree)
    PT.keep_children_from_name(thin, f'Base.LV{i}')
    coarse = PT.shallow_copy(tree)
    PT.keep_children_from_name(coarse, f'Base.LV{i+1}')
              
    from maia.utils import s_numbering
    thin_zone = PT.get_all_Zone_t(thin)[0]
    coarse_zone = PT.get_all_Zone_t(coarse)[0]
    coarse_distri = MT.distribution_value(coarse_zone, 'Cell')
    fi, fj = s_numbering.index_to_ij(np.arange(coarse_distri[0]+1, coarse_distri[1]+1),
                                     PT.Zone.CellSize(coarse_zone))
    PT.new_FlowSolution('CoarseId',
                        loc='CellCenter',
                        fields={'I': fi, 'J': fj},
                        parent=coarse_zone)

    maia.algo.interpolate(coarse, thin, comm, ['CoarseId'], 'CellCenter')

    expected_i = PT.get_np_value(PT.find_node_from_path(thin, f'Base.LV{i}/zone/CoarseId/I'))
    computed_i = PT.get_np_value(PT.find_node_from_path(thin, f'Base.LV{i}/zone/MultiGridCellInfo/ICoarseIdx'))
    assert np.array_equal(computed_i, expected_i)

    expected_j = PT.get_np_value(PT.find_node_from_path(thin, f'Base.LV{i}/zone/CoarseId/J'))
    computed_j = PT.get_np_value(PT.find_node_from_path(thin, f'Base.LV{i}/zone/MultiGridCellInfo/JCoarseIdx'))
    assert np.array_equal(computed_j, expected_j)

    assert PT.Zone.n_cell(coarse_zone) == PT.Zone.n_cell(thin_zone) // 4

  # Check splited BC sizes (small subset is 25% total size)
  for base in PT.iter_all_CGNSBase_t(tree):
    cur_lvl = int(PT.get_name(base)[-1])
    bc1 = PT.find_node_from_name(base, 'Xmax_1')
    bc2 = PT.find_node_from_name(base, 'Xmax_2')
    assert 3*(PT.Subset.n_elem(bc1)-1) == (PT.Subset.n_elem(bc2) - 1)
    assert (PT.Subset.n_elem(bc1) + PT.Subset.n_elem(bc2) - 1) == 2**(n_lvl-cur_lvl) * n2 + 1

@pytest_parallel.mark.parallel(2)
def test_multigrid_s(comm):
  
  nb_vtx_per_dir = 5 # 5 = 4n+1 avec n=1
  nb_lvl = 2
  
  dt = maia.factory.generate_dist_block(nb_vtx_per_dir, "S", comm)
  AGL.agglomerate_cells(dt, nb_lvl, comm)
  
  assert(len(PT.get_all_CGNSBase_t(dt)) == nb_lvl+1)
  
  for t in range(nb_lvl+1):
    zone = PT.find_node_from_path(dt, f'Base.LV{t}/zone')
    assert PT.Zone.VertexSize(zone) == (2**(2-t)+1, 2**(2-t)+1, 2**(2-t)+1)
    assert PT.Zone.CellSize(zone) == (2**(2-t), 2**(2-t), 2**(2-t))
    cx,cy,cz = PT.Zone.coordinates(zone)
    if t == 0:
      assert np.all(np.isin(np.unique(cx), [0, 0.25, 0.5, 0.75, 1.]))
      assert np.all(np.isin(np.unique(cy), [0, 0.25, 0.5, 0.75, 1.]))
      assert np.all(np.isin(np.unique(cz), [0, 0.25, 0.5, 0.75, 1.]))
    elif t == 1:
      assert np.all(np.isin(np.unique(cx), [0, 0.5, 1.]))
      assert np.all(np.isin(np.unique(cy), [0, 0.5, 1.]))
      assert np.all(np.isin(np.unique(cz), [0, 0.5, 1.]))
    elif t == 2:
      assert np.all(np.isin(np.unique(cx), [0, 1.]))
      assert np.all(np.isin(np.unique(cy), [0, 1.]))
      assert np.all(np.isin(np.unique(cz), [0, 1.]))
  
  icoarse_ids_lvl0 = PT.get_value(PT.get_node_from_path(dt, 'Base.LV0/zone/MultiGridCellInfo/ICoarseIdx'))
  icoarse_ids_lvl0_ref = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
  assert np.all(icoarse_ids_lvl0 == icoarse_ids_lvl0_ref)
  
  jcoarse_ids_lvl0 = PT.get_value(PT.get_node_from_path(dt, 'Base.LV0/zone/MultiGridCellInfo/JCoarseIdx'))
  jcoarse_ids_lvl0_ref = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
  assert np.all(jcoarse_ids_lvl0 == jcoarse_ids_lvl0_ref)
  
  kcoarse_ids_lvl0 = PT.get_value(PT.get_node_from_path(dt, 'Base.LV0/zone/MultiGridCellInfo/KCoarseIdx'))
  kcoarse_ids_lvl0_ref = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  assert np.all(kcoarse_ids_lvl0 == kcoarse_ids_lvl0_ref+comm.rank)
  
  icoarse_ids_lvl1 = PT.get_value(PT.get_node_from_path(dt, 'Base.LV1/zone/MultiGridCellInfo/ICoarseIdx'))
  assert np.all(icoarse_ids_lvl1==1)
  jcoarse_ids_lvl1 = PT.get_value(PT.get_node_from_path(dt, 'Base.LV1/zone/MultiGridCellInfo/JCoarseIdx'))
  assert np.all(jcoarse_ids_lvl1==1)
  kcoarse_ids_lvl1 = PT.get_value(PT.get_node_from_path(dt, 'Base.LV1/zone/MultiGridCellInfo/KCoarseIdx'))
  assert np.all(kcoarse_ids_lvl1==1)
  
  assert PT.get_node_from_path(dt, 'Base.LV2/zone/MultiGridCellInfo') == None
