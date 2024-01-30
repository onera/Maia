import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT

from maia.pytree.yaml import parse_yaml_cgns
from maia.utils import par_utils

from maia.algo.dist import remove_element as RME

def test_remove_element():
  zone = PT.new_Zone()

  quad_ec = [1,2,6,5,2,3,7,6,3,4,8,7,5,6,10,9,6,7,11,10,7,8,12,11]
  quad = PT.new_Elements('Quad', 'QUAD_4', econn=quad_ec, erange=[1,6], parent=zone)
  bar  = PT.new_Elements('Bar', 'BAR_2', econn=[10,9,11,10,12,11,5,1,9,5], erange=[7,11], parent=zone)

  ngon_ec = [2,1,3,2,1,5,4,3,2,6,3,7,6,5,8,4,6,7,5,9,8,7,10,6,7,11,9,10,12,8,10,11,11,12]
  ngon_pe = np.array([[29,0],[30,0],[29,0],[31,0],[29,30],[30,31],[29,32],[31,0],[33,30],\
                      [32,0],[31,34],[33,32],[33,34],[32,0],[34,0],[33,0],[34,0]])
  expected_pe = np.copy(ngon_pe)
  expected_pe[np.where(ngon_pe > 0)] -= 5
  ngon = PT.new_NGonElements('NGon', parent=zone,
      erange=[12,28], eso=np.arange(0, 35, 2), ec=ngon_ec, pe=ngon_pe)

  nface_ec = [7,5,1,3,6,2,5,9,11,4,6,8,7,10,12,14,13,12,9,16,11,13,15,17]
  nface = PT.new_NFaceElements('NFace', erange=[29,34], eso=np.arange(0,24+1,4), ec=nface_ec, parent=zone)

  zbc = PT.new_ZoneBC(zone)
  bc = PT.new_BC(point_list = [[25,27,28]], loc='EdgeCenter', parent=zbc)

  RME.remove_element(zone, bar)
  assert (PT.Element.Range(quad)  == [1,6]).all()
  assert (PT.Element.Range(ngon)  == [12-5,28-5]).all()
  assert (PT.Element.Range(nface) == [29-5,34-5]).all()
  assert (PT.get_child_from_name(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (PT.get_node_from_name(zone, 'PointList')[1] == [[20,22,23]]).all()
  assert PT.get_node_from_name(zone, 'Bar') is None

@pytest_parallel.mark.parallel(1)
def test_remove_ngons(comm):
  #Generated from G.cartNGon((0,0,0), (1,1,0), (3,4,1))
  # TODO handwritten ngon (4_cubes?)

  ec = [1,4, 2,5, 3,6, 4,7, 5,8, 6,9, 7,10, 8,11, 9,12, 1,2, 2,3, 4,5, 5,6, 7,8, 8,9, 10,11, 11,12]
  pe = np.array([[18,0], [18,19], [19,0],  [20,0],  [20,21], [21,0],  [22,0], [22,23], [23,0],
                 [18,0], [19,0],  [18,20], [19,21], [20,22], [21,23], [22,0], [23,0]])
  ngon = PT.new_NGonElements('NGon', erange=[1,17], eso=np.arange(0, 35, 2), ec=ec, pe=pe)
  distri = PT.new_child(ngon, ':CGNS#Distribution', 'UserDefinedData_t')
  PT.new_DataArray('Element', [0, 17, 17], parent=distri)
  PT.new_DataArray('ElementConnectivity', [0, 34, 34], parent=distri)

  RME.remove_ngons(ngon, [1,15], comm)

  expected_ec = [1,4,    3,6, 4,7, 5,8, 6,9, 7,10, 8,11, 9,12, 1,2, 2,3, 4,5, 5,6, 7,8, 8,9,      11,12]
  expected_pe = np.array([[16,0],         [17,0],  [18,0],  [18,19], [19,0], [20,0], [20,21], [21,0],
                          [16,0], [17,0], [16,18], [17,19], [18,20], [19,21],        [21,0]])
  assert (PT.get_node_from_name(ngon, 'ElementRange')[1] == [1, 15]).all()
  assert (PT.get_node_from_name(ngon, 'ElementConnectivity')[1] == expected_ec).all()
  assert (PT.get_node_from_name(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (PT.get_node_from_name(ngon, 'ElementStartOffset')[1] == np.arange(0,31,2)).all()
  assert (PT.get_node_from_path(ngon, ':CGNS#Distribution/Element')[1] == [0,15,15]).all()
  assert (PT.get_node_from_path(ngon, ':CGNS#Distribution/ElementConnectivity')[1] == [0,30,30]).all()

@pytest_parallel.mark.parallel(2)
def test_remove_ngons_2p(comm):

  #Generated from G.cartNGon((0,0,0), (1,1,0), (3,4,1))

  if comm.Get_rank() == 0:
    ec = [1,4,2,5,3,6,4,7,5,8]
    pe = np.array([[1,0], [1,2], [2,0], [3,0], [3,4]])
    eso = np.arange(0,2*5+1,2)
    distri_e  = [0, 5, 17]
    distri_ec = [0, 10, 34]
    to_remove = [2-1]

    expected_distri_e = [0, 4, 15]
    expected_ec = [1,4,    3,6,4,7,5,8]
    expected_pe = np.array([[1,0],        [2,0], [3,0], [3,4]])
    expected_eso = np.arange(0, 2*4+1, 2)
  elif comm.Get_rank() == 1:
    ec = [6,9,7,10,8,11,9,12,1,2,2,3,4,5,5,6,7,8,8,9,10,11,11,12]
    pe = np.array([[4,0], [5,0], [5,6], [6,0], [1,0], [2,0], [1,3], [2,4], [3,5], [4,6], [5,0], [6,0]])
    eso = np.arange(10, 2*17+1,2)
    distri_e  = [5, 17, 17]
    distri_ec = [10, 34, 34]
    to_remove = [16-5-1]

    expected_distri_e = [4, 15, 15]
    expected_ec = [6,9,7,10,8,11,9,12,1,2,2,3,4,5,5,6,7,8,8,9,      11,12]
    expected_pe = np.array([[4,0], [5,0], [5,6], [6,0], [1,0], [2,0], [1,3], [2,4], [3,5], [4,6],        [6,0]])
    expected_eso = np.arange(8, 2*15+1, 2)

  ngon = PT.new_NGonElements('NGon', erange=[7,24], eso=eso, ec=ec, pe=pe)
  distri = PT.new_child(ngon, ':CGNS#Distribution', 'UserDefinedData_t')
  PT.new_DataArray('Element', distri_e, parent=distri)
  PT.new_DataArray('ElementConnectivity', distri_ec, parent=distri)

  RME.remove_ngons(ngon, to_remove, comm)

  assert (PT.get_node_from_name(ngon, 'ElementRange')[1] == [7, 24-2]).all()
  assert (PT.get_node_from_name(ngon, 'ElementConnectivity')[1] == expected_ec).all()
  assert (PT.get_node_from_name(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (PT.get_node_from_name(ngon, 'ElementStartOffset')[1] == expected_eso).all()
  assert (PT.get_node_from_path(ngon, ':CGNS#Distribution/Element')[1] == expected_distri_e).all()

@pytest.mark.parametrize('elt_name',['BAR_2','TRI_3','TETRA_4'])
@pytest_parallel.mark.parallel([2])
def test_remove_elts_from_pl(elt_name, comm):

  dist_tree = maia.factory.dcube_generator.dcube_nodal_generate(3, 1., [0.,0.,0.], 'TETRA_4', comm, get_ridges=True)
  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  
  # > Define lineic BC
  ridge_pl_f = np.arange(89, 113, dtype=np.int32)
  ridge_distri = par_utils.uniform_distribution(ridge_pl_f.size, comm)
  ridge_pl = ridge_pl_f[ridge_distri[0] : ridge_distri[1]]

  zone_bc_n = PT.get_node_from_label(dist_tree, 'ZoneBC_t')
  bc_n = PT.new_BC('ridges', point_list=ridge_pl.reshape((1,-1), order='F'), loc='EdgeCenter', parent=zone_bc_n)
  PT.maia.newDistribution({'Index':ridge_distri}, parent=bc_n)

  # > Define elements to remove
  is_asked_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)==elt_name
  elt_n = PT.get_child_from_predicate(dist_zone, is_asked_elt)
  if elt_name=='TETRA_4':
    elt_pl_f = np.array([1,13,2,25,14,37], dtype=np.int32)+1
  elif elt_name=='TRI_3':
    elt_pl_f = np.arange(41, 51, dtype=np.int32)
  elif elt_name=='BAR_2':
    elt_pl_f = np.arange(89, 113, dtype=np.int32)
  start, end, _ = par_utils.uniform_distribution(elt_pl_f.size, comm)
  elt_pl = elt_pl_f[start:end]

  RME.remove_elts_from_pl(dist_zone, elt_n, elt_pl, comm)

  # > Checking result
  n_tet = {'TETRA_4':34, 'TRI_3':40, 'BAR_2':40}
  n_tri = {'TETRA_4':48, 'TRI_3':38, 'BAR_2':48}
  n_bar = {'TETRA_4':24, 'TRI_3':24, 'BAR_2': 0}

  is_tet_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TETRA_4'
  is_tri_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='TRI_3'
  is_bar_elt = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.CGNSName(n)=='BAR_2'
  is_tri_bc  = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='FaceCenter'
  is_bar_bc  = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=='EdgeCenter'

  elt_n  = PT.get_child_from_predicate(dist_zone, is_tet_elt)
  elt_distrib = PT.maia.getDistribution(elt_n, 'Element')[1]
  n_elt  = elt_distrib[1]-elt_distrib[0]
  elt_er = PT.get_child_from_name(elt_n,'ElementRange')[1]
  assert np.array_equal(elt_er, np.array([1, n_tet[elt_name]]))
  assert PT.get_child_from_name(elt_n,'ElementConnectivity')[1].size==n_elt*4
  assert elt_distrib[2]==n_tet[elt_name]

  cell_distri = par_utils.dn_to_distribution(n_elt, comm)
  assert np.array_equal(PT.maia.getDistribution(dist_zone, 'Cell')[1], cell_distri)

  elt_n  = PT.get_child_from_predicate(dist_zone, is_tri_elt)
  elt_er = PT.get_child_from_name(elt_n,'ElementRange')[1]
  elt_distrib = PT.maia.getDistribution(elt_n, 'Element')[1]
  n_elt  = elt_distrib[1]-elt_distrib[0]
  expected_er = np.array([n_tet[elt_name]+1, n_tet[elt_name]+n_tri[elt_name]])
  assert np.array_equal(elt_er, expected_er)
  assert PT.get_child_from_name(elt_n,'ElementConnectivity')[1].size==n_elt*3
  assert elt_distrib[2]==n_tri[elt_name]
  for bc_n in PT.get_nodes_from_predicate(dist_zone, is_tri_bc):
    pl = PT.Subset.getPatch(bc_n)[1]
    assert elt_er[0]<=np.min(pl) and np.max(pl)<=elt_er[1]
  if elt_name=='TRI_3':
    assert PT.get_node_from_name_and_label(dist_zone, 'Zmin', 'BC_t') is None
    zmax_bc_n = PT.get_node_from_name_and_label(dist_zone, 'Zmax', 'BC_t')
    assert PT.maia.getDistribution(zmax_bc_n, 'Index')[1][2]==6

  if elt_name=='BAR_2':
    assert PT.get_child_from_name(dist_zone, 'BAR_2') is None
    assert len(PT.get_nodes_from_predicate(dist_zone, is_bar_bc))==0  
  else: 
    elt_n  = PT.get_child_from_predicate(dist_zone, is_bar_elt)
    elt_er = PT.get_child_from_name(elt_n,'ElementRange')[1]
    elt_distrib = PT.maia.getDistribution(elt_n, 'Element')[1]
    n_elt  = elt_distrib[1]-elt_distrib[0]
    expected_er = np.array([n_tet[elt_name]+n_tri[elt_name]+1,
                            n_tet[elt_name]+n_tri[elt_name]+n_bar[elt_name]])
    assert np.array_equal(elt_er, expected_er)
    assert PT.get_child_from_name(elt_n,'ElementConnectivity')[1].size==n_elt*2
    assert elt_distrib[2]==n_bar[elt_name]


@pytest_parallel.mark.parallel(2)
def test_remove_elts_from_pl_conflict_bc(comm):
  rank = comm.Get_rank()
  cell_dn = ['[0,1,1]'      ,'[1,1,1]'   ][rank]
  tri1_ec = ['[1,2,3,3,2,1]','[4,5,6]'   ][rank]
  tri2_ec = ['[7,8,9,9,8,7]','[10,11,12]'][rank]
  tri_dn  = ['[0,2,3]'      ,'[2,3,3]'   ][rank]
  bc_pl   = ['[2,3]'        ,'[6]'       ][rank]
  bc_dn   = ['[0,2,3]'      ,'[2,3,3]'   ][rank]

  dist_zone = parse_yaml_cgns.to_node(f"""
    Zone Zone_t:
      ZoneType ZoneType_t 'Unstructured':
      :CGNS#Distribution UserDefinedData_t:
        Cell DataArray_t {cell_dn}:
      TRI_1 Elements_t I4 [5, 0]:
        ElementRange IndexRange_t I4 [1, 3]:
        ElementConnectivity DataArray_t I4 {tri1_ec}:
        :CGNS#Distribution UserDefinedData_t:
          Element DataArray_t {tri_dn}:
      TRI_2 Elements_t I4 [5, 0]:
        ElementRange IndexRange_t I4 [4, 6]:
        ElementConnectivity DataArray_t I4 {tri2_ec}:
        :CGNS#Distribution UserDefinedData_t:
          Element DataArray_t {tri_dn}:
      TETRA Elements_t I4 [10, 0]:
        ElementRange IndexRange_t I4 [7, 7]:
      ZoneBC ZoneBC_t:
        BC BC_t 'Null':
          GridLocation GridLocation_t 'FaceCenter':
          PointList IndexArray_t I4 [{bc_pl}]:
          :CGNS#Distribution UserDefinedData_t:
            Index DataArray_t {bc_dn}:
    """)

  elt_n = PT.get_child_from_name(dist_zone, 'TRI_1')
  elt_pl = np.array([rank+1], dtype=np.int32)
  RME.remove_elts_from_pl(dist_zone, elt_n, elt_pl, comm)

  expected_elt_er = { 'TRI_1': np.array([1,1]),
                      'TRI_2': np.array([2,4]),
                      }
  expected_elt_dn = { 'TRI_1': np.array([[0,0,1],[0,1,1]][rank]),
                      'TRI_2': np.array([[0,2,3],[2,3,3]][rank]),
                      }
  for elt_name, elt_er in expected_elt_er.items():
    elt_n = PT.get_child_from_name(dist_zone, elt_name)
    assert np.array_equal(PT.Element.Range(elt_n), elt_er)
    assert np.array_equal(PT.maia.getDistribution(elt_n, 'Element')[1], expected_elt_dn[elt_name])
  
  expected_bc_pl = np.array([[[1]],[[4]]][rank])
  expected_bc_dn = np.array([[0,1,2],[1,2,2]][rank])
  bc_n = PT.get_node_from_label(dist_zone, 'BC_t')
  assert np.array_equal(PT.Subset.getPatch(bc_n)[1], expected_bc_pl)
  assert np.array_equal(PT.maia.getDistribution(bc_n, 'Index')[1], expected_bc_dn)

  expected_cell_dn = np.array([[0,1,1],[1,1,1]][rank])
  assert np.array_equal(PT.maia.getDistribution(dist_zone, 'Cell')[1], expected_cell_dn)
