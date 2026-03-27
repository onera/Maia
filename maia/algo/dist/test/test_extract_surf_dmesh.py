import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia              import npy_pdm_gnum_dtype as pdm_dtype
from maia.factory      import dcube_generator
from maia.factory      import full_to_dist as F2D
from maia.utils        import par_utils

from maia.algo.dist   import extract_surf_dmesh as EXC

@pytest_parallel.mark.parallel(2)
def test_extract_single_zone(comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]

  #Simplify mesh keeping only 2 BCs
  zone_bc = PT.get_child_from_label(zone, "ZoneBC_t")
  zone_bc[2] = [zone_bc[2][0], zone_bc[2][3]]

  surf_zone = EXC.extract_surf_from_bc_single(zone, PT.pred.label_is('BC_t'), comm)

  assert PT.Zone.n_cell(surf_zone) == 2*2*2 #2BC, 2*2 faces
  
  dtype = 'I4' if pdm_dtype == np.int32 else 'I8'
  yt = f"""
  NGonElements Elements_t [22,0]:
    ElementRange IndexRange_t {dtype} [1,8]:
    ElementStartOffset DataArray_t {dtype} [0,4,8,12,16,20,24,28,32]:
    ElementConnectivity DataArray_t:
      {dtype} : [1,4,5,2,2,5,6,3,4,7,8,5,5,8,9,6,3,6,11,10,6,9,12,11,10,11,14,13,11,12,15,14]
  """
  expected_ngon_full = PT.yaml.to_node(yt)
  expected_ngon = F2D.distribute_element_node(expected_ngon_full, comm)
  expected_parent = [[1,2,3,4], [21,22,23,24]][comm.rank]

  ngon = PT.Zone.NGonNode(surf_zone)
  assert PT.is_same_tree(ngon, expected_ngon)
  assert (PT.get_np_value(PT.find_node_from_path(surf_zone, 'DiscreteData/Parent')) == expected_parent).all()
    

@pytest_parallel.mark.parallel(3)
def test_extract_tree(comm):
  tree = dcube_generator.dcube_generate(4,1.,[0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]

  surf_zone = EXC.extract_surf_from_bc_single(zone, PT.pred.label_is('BC_t'), comm)
  surf_tree = EXC.extract_surf_from_bc(tree, PT.pred.label_is('BC_t'), comm)

  assert len(PT.get_all_CGNSBase_t(surf_tree)) == 1
  assert len(PT.get_all_Zone_t(surf_tree)) == 1
  assert PT.is_same_tree(PT.get_all_Zone_t(surf_tree)[0], surf_zone)

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize('elt_kind',  ['NFACE_n', 'TETRA_4'])
def test_extract_surf(elt_kind, comm):
  vol = maia.factory.generate_dist_sphere(5, elt_kind, comm)
  surf = EXC.extract_surf_from_bc(vol, PT.pred.name_is('Skin'), comm)

  base = PT.get_all_CGNSBase_t(surf)[0]
  zone = PT.get_all_Zone_t(surf)[0]
  assert PT.Base.CellDimension(base) == PT.Zone.CellDimension(zone) == 2
  assert PT.Zone.n_cell(zone) == 500 and comm.allreduce(MT.Zone.dn_cell(zone)) == 500
  assert PT.Zone.n_vtx(zone) == 252 and comm.allreduce(MT.Zone.dn_vtx(zone)) == 252

  ng = PT.Zone.NGonNode(zone)
  assert (PT.Element.Range(ng) == [1, 500]).all()

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize('elt_kind',  ['NGON_n', 'TRI_3'])
def test_extract_edge(elt_kind, comm):
  surf = maia.factory.generate_dist_sphere(5, 'NGON_n', comm)
  
  vtx_list = np.array([55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 168, 174, 180, 186, 192, 194, 196, 200, 202, 206, 208, 212, 214, 218, 220])
  zone = PT.get_all_Zone_t(surf)[0]
  bar_n = MT.Zone.EdgeNode(zone)
  edge_vtx = PT.get_np_value(PT.get_child_from_name(bar_n, 'ElementConnectivity'))
  
  isin = np.isin(edge_vtx[0::2], vtx_list) & np.isin(edge_vtx[1::2], vtx_list)
  pl = np.flatnonzero(isin) + MT.Element.distribution(bar_n)[0] + PT.Element.Range(bar_n)[0]
  pl = pl.astype(pdm_dtype)

  zbc = PT.new_ZoneBC(parent=zone)
  bc = PT.new_BC('Tropic', loc='EdgeCenter', point_list=pl.reshape((1,-1), order='F'), parent=zbc)
  MT.new_Distribution({'Index' : par_utils.dn_to_distribution(pl.size, comm)}, parent=bc)

  if elt_kind == 'TRI_3':
    maia.algo.dist.convert_ngon_to_elements(surf, comm)

  edge = EXC.extract_surf_from_bc(surf, PT.pred.ALWAYS_TRUE, comm)

  base = PT.get_all_CGNSBase_t(edge)[0]
  zone = PT.get_all_Zone_t(edge)[0]
  assert PT.Base.CellDimension(base) == PT.Zone.CellDimension(zone) == 1
  assert PT.Zone.n_cell(zone) == 25 and comm.allreduce(MT.Zone.dn_cell(zone)) == 25
  assert PT.Zone.n_vtx(zone) == 25 and comm.allreduce(MT.Zone.dn_vtx(zone)) == 25

  bar = MT.Zone.EdgeNode(zone)
  assert (PT.Element.Range(bar) == [1, 25]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('dim',  [2,3])
def test_struct(dim, comm):
  n_vtx = dim * [11]
  tree = maia.factory.generate_dist_block(n_vtx, 'S', comm)
  surf = EXC.extract_surf_from_bc(tree, PT.pred.name_in(['Xmax', 'Ymax']), comm)

  zone = PT.get_all_Zone_t(surf)[0]
  assert PT.Zone.Type(zone) == 'Unstructured'
  assert PT.Zone.CellDimension(zone) == dim-1
  assert PT.Zone.n_cell(zone) == 2*10**(dim-1)
  assert PT.Zone.n_vtx(zone) == 2*11**(dim-1) - 11**(dim-2)
  assert all(PT.Element.Dimension(e) == dim-1 for e in PT.get_children_from_label(zone, 'Elements_t'))

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('elt_kind',  ['NFACE_n', 'TETRA_4', 'S'])
def test_empty(elt_kind, comm):
  tree = maia.factory.generate_dist_block(11, elt_kind, comm)
  surf = EXC.extract_surf_from_bc(tree, PT.pred.name_is('WRONG'), comm)
  zone = PT.get_node_from_label(surf, 'Zone_t')
  assert PT.Zone.n_cell(zone) == PT.Zone.n_vtx(zone) == 0