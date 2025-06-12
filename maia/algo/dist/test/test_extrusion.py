import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import par_utils

from maia.algo.dist import extrusion as EXT

get_elt_ec = lambda n : PT.find_child_from_name(n, 'ElementConnectivity')[1]

IS_BAR = PT.pred.is_elmt_of_type('BAR_2')
IS_EDGE_SUBSET = PT.pred.label_in(['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) & PT.pred.has_location('*EdgeCenter')
IS_FACE_SUBSET = PT.pred.label_in(['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) & PT.pred.has_location('*FaceCenter')

@pytest_parallel.mark.parallel([1,2])
def test_nodes_duplication(comm):

  # Prepare test
  extrusion_vector = [1., 2., 3.]
  dist_tree = maia.factory.generate_dist_block(10, 'TRI_3', comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]
  coords_2d = PT.Zone.coordinates(zone)
  distrib_vtx_2d = MT.distribution_value(zone, 'Vertex')
  dn_vtx_2d = distrib_vtx_2d[1] - distrib_vtx_2d[0]

  # Run test
  EXT._nodes_duplication(zone, extrusion_vector, comm)

  # Verification
  assert MT.distribution_value(zone, 'Vertex')[2] == 2*distrib_vtx_2d[2]
  coords = PT.Zone.coordinates(zone)
  if comm.size==1:
    for i in range(3):
      assert (coords[i][:dn_vtx_2d] == coords_2d[i]).all()
      assert (coords[i][dn_vtx_2d:] == coords_2d[i]+extrusion_vector[i]).all()
  elif comm.size==2:
    for i in range(3):
        if comm.rank == 0: # Since size == 2, rank 0 has all old coords. In particular, first half is its data
          assert (coords[i][0:dn_vtx_2d] == coords_2d[i]).all()
        elif comm.rank == 1: # Since size == 2, rank 1 has all new data. In particular, second half is its data
          assert (coords[i][dn_vtx_2d:] == coords_2d[i]+extrusion_vector[i]).all()

@pytest_parallel.mark.parallel(3)
def test_determine_mesh_orientation(comm):

  # Prepare test
  extrusion_vector = np.array([1., 2., 3.])
  dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]

  # Run test and verification
  assert     EXT._determine_mesh_orientation(zone,  extrusion_vector, comm)
  assert not EXT._determine_mesh_orientation(zone, -extrusion_vector, comm)

def test_reorder_ngon_ec():

  # Prepare test
  ngon_n = PT.new_NGonElements(erange=[1, 1], eso=[7, 11, 14], ec=[1,2,3,4, 5,6,7])
  MT.new_Distribution({'ElementConnectivity': [7, 10, 25]}, parent=ngon_n)

  # Run test
  EXT._reorder_ngon_ec(ngon_n)

  # Verification
  assert (get_elt_ec(ngon_n) == [4, 3, 2, 1, 7, 6, 5]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("align", [True, False])
def test_ngon_duplication(align, comm):

  # Prepare test
  dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]
  n_cell_2d = PT.Zone.n_cell(zone)
  n_vtx_2d  = PT.Zone.n_vtx(zone)
  ngon_name_ini = PT.get_name(PT.Zone.NGonNode(zone))
  ngon_ec_ini = PT.get_child_from_name(PT.Zone.NGonNode(zone), "ElementConnectivity")[1].copy()

  # Run test
  EXT._ngon_duplication(zone, comm, align)

  # Verification
  assert len(PT.get_children_from_predicate(zone, PT.pred.is_elmt_of_type('NGON_n'))) == 2
  old_ngon = PT.get_child_from_name(zone, ngon_name_ini)
  new_ngon = PT.get_child_from_name(zone, f'{ngon_name_ini}_bis')
  
  assert (PT.Element.Range(new_ngon) == PT.Element.Range(old_ngon)+n_cell_2d).all()
  for array in ['ElementStartOffset', 'ParentElements']:
    assert (PT.get_child_from_name(new_ngon, array)[1] == PT.get_child_from_name(old_ngon, array)[1]).all()
  if align:
    assert (get_elt_ec(new_ngon) == ngon_ec_ini+n_vtx_2d).all()
  else:
    assert (get_elt_ec(old_ngon) == ngon_ec_ini).all()

@pytest.mark.parametrize("align", [True, False])
def test_extrude_bar_to_ngon(align):

  # Prepare test
  n_vtx  = 25
  n_cell = 37
  bar = PT.new_Elements('EdgeElements', type='BAR_2', erange=[1,17], econn=[1,2, 2,3, 3,4, 4,1, 1,3], 
                        pe=[[6, 6, 7, 7, 6], [0, 0, 0, 0, 7]])
  MT.new_Distribution({'Element': [5, 10, 17]}, parent=bar)

  # Run test
  EXT._extrude_bar_to_ngon(bar, n_vtx, n_cell, align)

  # Verification
  assert (bar[1] == [22, 0]).all()
  assert (PT.Element.Range(bar) == [1, 17]).all()
  
  if align:
    assert (get_elt_ec(bar) == [2,1,1+n_vtx,2+n_vtx, 3,2,2+n_vtx,3+n_vtx,
                                4,3,3+n_vtx,4+n_vtx, 1,4,4+n_vtx,1+n_vtx,
                                3,1,1+n_vtx,3+n_vtx]).all()
  else:
    assert (get_elt_ec(bar) == [1,2,2+n_vtx,1+n_vtx, 2,3,3+n_vtx,2+n_vtx,
                                3,4,4+n_vtx,3+n_vtx, 4,1,1+n_vtx,4+n_vtx,
                                1,3,3+n_vtx,1+n_vtx]).all()
  bar_eso = PT.get_child_from_name(bar, 'ElementStartOffset')
  assert (bar_eso[1] == [20, 24, 28, 32, 36, 40]).all()

@pytest_parallel.mark.parallel(1)
def test_merge_ngons(comm):

  # Prepare test
  dist_tree = maia.factory.generate_dist_block(11, 'HEXA_8', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]
  PT.rm_node_from_path(zone, 'NFaceElements')
  old_ngon1 = PT.Zone.NGonNode(zone)
  old_ngon2 = PT.deep_copy(old_ngon1)
  PT.set_name(old_ngon2, f'{old_ngon1[0]}_bis')
  old_ngon1_er = PT.Element.Range(old_ngon1)
  old_ngon2_er = PT.Element.Range(old_ngon2)
  old_ngon2_er += old_ngon1_er[1]
  PT.add_child(zone, old_ngon2)

  # Run test
  EXT._merge_ngons(zone, comm)

  # Verification
  new_ngon = PT.Zone.NGonNode(zone) # Would fail if number of NGON != 1
  assert PT.get_name(new_ngon) == 'NGonElements'
  assert (PT.Element.Range(new_ngon) == [old_ngon1_er[0], old_ngon2_er[1]]).all()

  old_eso_len = PT.Element.Size(old_ngon1)+1
  old_ngon1_eso = PT.get_child_from_name(old_ngon1, 'ElementStartOffset')[1]
  old_ngon2_eso = PT.get_child_from_name(old_ngon2, 'ElementStartOffset')[1]
  new_ngon_eso  = PT.get_child_from_name(new_ngon,  'ElementStartOffset')[1]
  assert (new_ngon_eso[0:old_eso_len]    == old_ngon1_eso).all()
  assert (new_ngon_eso[old_eso_len-2:-1] == old_ngon2_eso+old_ngon1_eso[-2]).all()

  old_ec_len = old_ngon1_eso[-1]
  new_ngon_ec  = PT.get_child_from_name(new_ngon,  'ElementConnectivity')[1]
  assert (new_ngon_ec[0:old_ec_len]  == get_elt_ec(old_ngon1)).all()
  assert (new_ngon_ec[old_ec_len:]   == get_elt_ec(old_ngon2)).all()

@pytest.mark.parametrize("align", [True, False])
def test_extrude_tri_to_prism_and_tris(align):

  # Prepare test
  n_vtx  = 25
  n_cell = 20
  num    = 1
  er_max = 12
  tri = PT.new_Elements('TRI', type='TRI_3', erange=[1,17], econn=[1,2,3, 1,3,4])
  MT.new_Distribution({'Element' : [0, 17, 17]}, tri)
  old_tri = PT.deep_copy(tri)

  # Run test
  new_tri1, new_tri2 = EXT._extrude_tri_to_prism_and_tris(tri, num, n_vtx, n_cell, er_max, align)

  # Verification
  penta = tri # Old tri is now penta
  assert PT.get_name(penta) == 'PENTA_6.1'   and PT.Element.CGNSName(penta) == 'PENTA_6'
  assert PT.get_name(new_tri1) == 'TRI_3.1a' and PT.Element.CGNSName(new_tri1) == 'TRI_3'
  assert PT.get_name(new_tri2) == 'TRI_3.1b' and PT.Element.CGNSName(new_tri2) == 'TRI_3'

  assert (get_elt_ec(penta) == [1,2,3,26,27,28, 1,3,4,26,28,29]).all()
  assert (PT.Element.Range(penta) == PT.Element.Range(old_tri)).all()

  if align:
    assert (get_elt_ec(new_tri2) == get_elt_ec(old_tri)+n_vtx).all()
  else:
    assert (get_elt_ec(new_tri1) == get_elt_ec(old_tri)).all()
  assert (PT.Element.Range(new_tri1) == [er_max+1,    er_max+17]).all()
  assert (PT.Element.Range(new_tri2) == [er_max+1+20, er_max+17+20]).all()

@pytest.mark.parametrize("align", [True, False])
def test_extrude_quad_to_hexa_and_quads(align):

  # Prepare test
  n_vtx  = 25
  n_cell = 17
  num    = 1
  er_max = 12
  quad = PT.new_Elements('QUAD', type='QUAD_4', erange=[1,17], econn=[1,2,5,4, 2,3,6,5])
  MT.new_Distribution({'Element' : [0,17,17]}, quad)
  old_quad = PT.deep_copy(quad)

  # Run test
  new_quad1, new_quad2 = EXT._extrude_quad_to_hexa_and_quads(quad, num, n_vtx, n_cell, er_max, align)

  # Verification
  hexa = quad # Old quad is now hexa
  assert PT.get_name(hexa) == 'HEXA_8.1'       and PT.Element.CGNSName(hexa) == 'HEXA_8'
  assert PT.get_name(new_quad1) == 'QUAD_4.1a' and PT.Element.CGNSName(new_quad1) == 'QUAD_4'
  assert PT.get_name(new_quad2) == 'QUAD_4.1b' and PT.Element.CGNSName(new_quad2) == 'QUAD_4'

  assert (get_elt_ec(hexa) == [1,2,5,4,26,27,30,29, 2,3,6,5,27,28,31,30]).all()
  assert (PT.Element.Range(hexa) == PT.Element.Range(old_quad)).all()

  if align:
    assert (get_elt_ec(new_quad2) == get_elt_ec(old_quad)+n_vtx).all()
  else:
    assert (get_elt_ec(new_quad1) == get_elt_ec(old_quad)).all()
  assert (PT.Element.Range(new_quad1) == [er_max+1,    er_max+17]).all()
  assert (PT.Element.Range(new_quad2) == [er_max+1+17, er_max+2*17]).all()

@pytest.mark.parametrize("align", [True, False])
def test_extrude_bar_to_quad(align):

  # Prepare test
  n_vtx  = 25
  num    = 1
  bar = PT.new_Elements('EdgeElements', type='BAR_2', erange=[1,17], econn=[1,2, 2,3])
  old_bar = PT.deep_copy(bar)

  # Run test
  EXT._extrude_bar_to_quad(bar, num, n_vtx, align)

  # Verification
  quad = bar # Old bar is now quad
  assert PT.get_name(quad) == 'QUAD_4.1' and PT.Element.CGNSName(quad) == 'QUAD_4'

  assert (PT.Element.Range(quad) == PT.Element.Range(old_bar)).all()

  if align:
    assert (get_elt_ec(quad) == [1,2,27,26, 2,3,28,27]).all()
  else:
    assert (get_elt_ec(quad) == [2,1,26,27, 3,2,27,28]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("pl", [False, True])
@pytest.mark.parametrize("data", [False, True])
def test_pl_and_data_vtx_duplication(pl, data, comm):

  # Prepare test
  distrib_idx = np.array([0,2,3]) if comm.rank == 0 else np.array([2,3,3])
  slice_me = lambda t: t[distrib_idx[0]:distrib_idx[1]]
  pl = slice_me(np.array([5,7,13])).reshape((1,-1),order='F') if pl else None
  data = {'data' : slice_me(np.array([5,7,13.]))}             if data else {}
  n_vtx_2d = 25

  # Run test
  new_distrib_idx, new_pl, dist_data = EXT._pl_and_data_vtx_duplication(pl, distrib_idx, n_vtx_2d, data, comm)

  # Verification
  assert (pl is None) == (new_pl is None)
  assert data.keys() == dist_data.keys()

  expt_distri = [0,3,6]    if comm.rank == 0 else [3,6,6]
  expt_pl     = [[5,7,13]] if comm.rank == 0 else [[30,32,38]]
  assert (new_distrib_idx == expt_distri).all()
  if pl is not None:
    assert (new_pl == expt_pl).all()
  if data:
    assert (dist_data['data'] == [5.,7.,13.]).all()

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("coords_dim", [2, 3])
@pytest.mark.parametrize("ksubset_as", ['GC', 'BC'])
def test_extrusion_2d_cart_ngon(coords_dim, ksubset_as, comm):

  # Prepare 2D case
  dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  
  # Add some cases w/o NG or PE
  if coords_dim == 2 and ksubset_as == 'GC':
    PT.rm_nodes_from_name(dist_tree, 'ParentElements')
  if coords_dim == 3 and ksubset_as == 'BC':
    PT.rm_nodes_from_name(dist_tree, 'NGonElements')

  if coords_dim == 2:
    PT.rm_nodes_from_name(dist_tree, 'CoordinateZ')
    base = PT.get_child_from_name(dist_tree, 'Base')
    PT.set_value(base, [2,2])

  base = PT.get_all_CGNSBase_t(dist_tree)[0]
  zone = PT.get_all_Zone_t(dist_tree)[0]

  n_cell = PT.Zone.n_cell(zone)
  n_edges = sum(PT.Element.Size(e) for e in PT.get_children_from_predicate(zone, IS_BAR))

  # Run test
  EXT.extrude(dist_tree, [0., 0., 1.], comm, ksubset_as=ksubset_as)

  # Verification
  assert np.all(PT.get_value(base) == [3, 3])
  assert PT.Zone.n_cell(zone) == n_cell
  assert PT.Zone.n_face(zone) == n_edges+2*n_cell

  assert len(PT.get_nodes_from_predicate(zone, IS_BAR)) == 0

  assert PT.get_child_from_name(PT.Zone.NGonNode(zone), 'ParentElements') is not None
  assert len(PT.get_nodes_from_predicate(zone, IS_EDGE_SUBSET)) == 0

  if ksubset_as == 'GC':
    assert len(PT.get_nodes_from_predicate(zone, IS_FACE_SUBSET)) == 4 # 4 initial BC
    assert len(PT.get_nodes_from_label(zone, 'GridConnectivity_t')) == 2
    assert np.all(np.abs(PT.get_node_from_name(zone, 'Translation')[1]) == [0., 0., 1.])
  elif ksubset_as == 'BC':
    assert len(PT.get_nodes_from_predicate(zone, IS_FACE_SUBSET)) == 4 + 2
    assert len(PT.get_children_from_label(base, 'Family_t')) == 2


@pytest_parallel.mark.parallel([1])
@pytest.mark.parametrize("dupl_vtx_data", [True, False])
def test_extrusion_2d_cart_ngon_loc(dupl_vtx_data, comm):

  # Prepare test
  dist_tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)

  zone = PT.get_all_Zone_t(dist_tree)[0]

  n_cell = PT.Zone.n_cell(zone)
  n_vtx  = PT.Zone.n_vtx(zone)
  n_edges = sum(PT.Element.Size(e) for e in PT.get_children_from_predicate(zone, IS_BAR))

  # > Container CellCenter
  id_cc = np.arange(n_cell) + n_edges + 1
  PT.new_DiscreteData('DD_woPL#CellCenter', loc='CellCenter', fields={'Id': id_cc}, parent=zone)
  fs_wpl_cc = PT.new_FlowSolution('FS_wPL#CellCenter', loc='CellCenter', fields={'Id': id_cc[-2:]}, parent=zone)
  PT.new_IndexArray('PointList', value=[id_cc[-2:]], parent=fs_wpl_cc)
  MT.new_Distribution({'Index': [0,2,2]}, parent=fs_wpl_cc)

  # > Container EdgeCenter (with PL only because no pl is not allowed)
  id_ec = np.array([4,6,11,13], zone[1].dtype) # Internal edges
  zsr_wpl_ec = PT.new_ZoneSubRegion('ZSR_wPL#EdgeCenter', loc='EdgeCenter', fields={'Id': id_ec}, parent=zone)
  PT.new_IndexArray('PointList', value=[id_ec], parent=zsr_wpl_ec)
  MT.new_Distribution({'Index': [0,4,4]}, parent=zsr_wpl_ec)

  # > Container Vertex
  id_vtx = np.arange(n_vtx, dtype=pdm_dtype) + 1
  PT.new_DiscreteData('DD_woPL#Vertex', loc='Vertex', fields={'Id': id_vtx}, parent=zone)
  fs_wpl_vtx = PT.new_FlowSolution('FS_wPL#Vertex', loc='Vertex', fields={'Id': id_vtx[0:3]}, parent=zone)
  PT.new_IndexArray('PointList', value=[id_vtx[0:3]], parent=fs_wpl_vtx)
  MT.new_Distribution({'Index': [0,3,3]}, parent=fs_wpl_vtx)
  PT.new_ZoneSubRegion('ZSR_related#Vertex', gc_name='Ymin', fields={'Id': id_vtx[-3:]}, parent=zone)

  # BC setup : to test several cases, we define : 
  #  - Ymin and Ymax --> JN (Vertex)
  #  - Xmin --> BC EdgeCenter + BCDS Vertex, EdgeCenter and CellCenter
  #  - Xmax --> BC CellCenter
  xmin, xmax, ymin, ymax = [PT.get_node_from_name(zone, name) for name in ['Xmin', 'Xmax', 'Ymin', 'Ymax']]

  # > Xmin : BC Edge + BCDS with PointList
  bcds_wpl_cc = PT.new_BCDataSet(name='BCDS_wpl#CellCenter', loc='CellCenter', point_list=[[1+16,5+16]], parent=xmin)
  PT.new_BCData('NeumannData', fields={'Id': [1,5]}, parent=bcds_wpl_cc)
  MT.new_Distribution({'Index': np.array([0,2,2],dtype=pdm_dtype)}, parent=bcds_wpl_cc)
  bcds_wpl_ec = PT.new_BCDataSet(name='BCDS_wpl#EdgeCenter', loc='EdgeCenter', point_list=[[2,10]], parent=xmin)
  PT.new_BCData('NeumannData', fields={'Id': [2,10]}, parent=bcds_wpl_ec)
  MT.new_Distribution({'Index': np.array([0,2,2],dtype=pdm_dtype)}, parent=bcds_wpl_ec)
  bcds_wpl_vtx = PT.new_BCDataSet(name='BCDS_wpl#Vertex', loc='Vertex', point_list=[[1,2,3]], parent=xmin)
  PT.new_BCData('NeumannData', fields={'Id': [1,2,3]}, parent=bcds_wpl_vtx)
  MT.new_Distribution({'Index': np.array([0,3,3],dtype=pdm_dtype)}, parent=bcds_wpl_vtx)

  # > Xmax : BC CellCenter
  PT.update_child(xmax, 'GridLocation', 'GridLocation_t', 'CellCenter')
  PT.update_child(xmax, 'PointList', value=np.array([[4+16,8+16]], zone[1].dtype))

  # > Ymin and Ymax : GC Vertex (more interesting than Edge)
  # Remark: GC is not well defined but enough for test (missing perio)
  for bc in [ymin, ymax]:
    PT.update_child(bc, 'GridLocation', 'GridLocation_t', 'Vertex')
    PT.update_node(bc, value=PT.get_name(zone), label='GridConnectivity_t')
    MT.new_Distribution({'Index' : [0,3,3]}, bc)
  PT.update_child(ymin, 'PointList', value=np.array([[1,2,3]], zone[1].dtype))
  PT.update_child(ymax, 'PointList', value=np.array([[7,8,9]], zone[1].dtype))
  PT.new_IndexArray('PointListDonor', PT.get_child_from_name(ymax, 'PointList')[1].copy(), ymin)
  PT.new_IndexArray('PointListDonor', PT.get_child_from_name(ymin, 'PointList')[1].copy(), ymax)
  PT.new_Descriptor('GridConnectivityRegionName', 'Ymax', parent=ymin)
  PT.new_Descriptor('GridConnectivityRegionName', 'Ymin', parent=ymax)
  PT.rm_nodes_from_predicate(zone, PT.pred.name_in(['Ymin', 'Ymax']))
  PT.new_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t', children=[ymin, ymax])

  # Run test
  EXT.extrude(dist_tree, [0., 0., 1.], comm, dupl_vtx_data=dupl_vtx_data)

  # Verification
  # CellCenter containers
  container = PT.get_node_from_name(zone, f'DD_woPL#CellCenter')
  assert PT.Subset.GridLocation(container) == 'CellCenter'
  assert PT.get_child_from_name(container, 'PointList') is None
  assert (PT.get_child_from_name(container, 'Id')[1] == [17,18,19,20,21,22,23,24]).all() # Data is unchanged
  container = PT.get_node_from_name(zone, f'FS_wPL#CellCenter')
  assert PT.Subset.GridLocation(container) == 'CellCenter'
  assert (PT.get_child_from_name(container, 'Id')[1] == [23,24]).all() # Data is unchanged
  assert (PT.get_child_from_name(container, 'PointList')[1] == [39,40]).all()

  # Vertex containers
  tile_value = 2 if dupl_vtx_data else 1
  container = PT.get_node_from_name(zone, f'DD_woPL#Vertex')
  assert PT.Subset.GridLocation(container) == 'Vertex'
  assert (PT.get_child_from_name(container, 'Id')[1] == np.tile(np.arange(9)+1, tile_value)).all()
  if dupl_vtx_data:
    assert PT.get_child_from_name(container, 'PointList') is None
  else:
    assert (PT.get_child_from_name(container, 'PointList')[1][0] == np.arange(9)+1).all()
  container = PT.get_node_from_name(zone, f'FS_wPL#Vertex')
  assert PT.Subset.GridLocation(container) == 'Vertex'
  expected_pl = [1,2,3,10,11,12] if dupl_vtx_data else [1,2,3]
  assert (PT.get_child_from_name(container, 'Id')[1] == np.tile([1,2,3], tile_value)).all()
  assert (PT.get_child_from_name(container, 'PointList')[1][0] == expected_pl).all()
  container = PT.get_node_from_name(zone, f'ZSR_related#Vertex')
  assert (PT.get_child_from_name(container, 'GridConnectivityRegionName') is not None) == dupl_vtx_data
  assert (PT.get_child_from_name(container, 'Id')[1] == np.tile([7,8,9], tile_value)).all()
  if dupl_vtx_data: # Ref node will be duplicated also -> no PL
    assert PT.get_child_from_name(container, 'PointList') is None
  else:
    assert (PT.get_child_from_name(container, 'PointList')[1][0] == [1,2,3]).all()

  # Face containers
  container = PT.get_node_from_name(zone, f'ZSR_wPL#EdgeCenter')
  assert PT.Subset.GridLocation(container) == 'FaceCenter'
  assert (PT.get_child_from_name(container, 'Id')[1] == [4,6,11,13]).all()
  assert (PT.get_child_from_name(container, 'PointList')[1][0] == [4,6,11,13]).all()

  # Xmin
  assert PT.Subset.GridLocation(xmin) == 'FaceCenter'
  assert (PT.get_node_from_name(xmin, 'PointList')[1][0] == [2,10]).all()

  container = PT.get_node_from_name(zone, 'BCDS_wpl#CellCenter')
  assert PT.Subset.GridLocation(container) == 'CellCenter'
  assert (PT.get_node_from_name(container, 'Id')[1] == [1,5]).all()
  assert (PT.get_child_from_name(container, 'PointList')[1][0] == [33,37]).all()
  container = PT.get_node_from_name(zone, 'BCDS_wpl#Vertex')
  assert PT.Subset.GridLocation(container) == 'Vertex'
  assert (PT.get_node_from_name(container, 'Id')[1] == np.tile([1,2,3], tile_value)).all()
  if dupl_vtx_data:
    assert (PT.get_child_from_name(container, 'PointList')[1][0] == [1,2,3,10,11,12]).all()
  else:
    assert (PT.get_child_from_name(container, 'PointList')[1][0] == [1,2,3]).all()
  container = PT.get_node_from_name(zone, 'BCDS_wpl#EdgeCenter')
  assert PT.Subset.GridLocation(container) == 'FaceCenter'
  assert (PT.get_node_from_name(container, 'Id')[1] == [2,10]).all()
  assert (PT.get_child_from_name(container, 'PointList')[1][0] == [2,10]).all()

  # Xmax
  assert PT.Subset.GridLocation(xmax) == 'CellCenter'
  assert (PT.get_node_from_name(xmax, 'PointList')[1][0] == [36,40]).all()

  # Ymin:
  assert PT.Subset.GridLocation(ymin) == 'Vertex'
  assert (PT.get_node_from_name(ymin, 'PointList'     )[1][0] == [1,2,3, 10,11,12]).all()
  assert (PT.get_node_from_name(ymin, 'PointListDonor')[1][0] == [7,8,9,  16,17,18]).all()
  # Ymax:
  assert PT.Subset.GridLocation(ymax) == 'Vertex'
  assert (PT.get_node_from_name(ymax, 'PointList'     )[1][0] == [7,8,9,  16,17,18]).all()
  assert (PT.get_node_from_name(ymax, 'PointListDonor')[1][0] == [1,2,3, 10,11,12]).all()


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("element_type", ['TRI_3', 'QUAD_4'])
def test_extrusion_2d_cart_elem(element_type, comm):

  # Prepare 2D case
  dist_tree = maia.factory.generate_dist_block(11, element_type, comm)

  base = PT.get_all_CGNSBase_t(dist_tree)[0]
  zone = PT.get_all_Zone_t(dist_tree)[0]

  # Run test
  EXT.extrude(dist_tree, [0., 0., 2.], comm)

  # Verification
  assert np.all(PT.get_value(base) == [3, 3])

  assert len(PT.get_nodes_from_predicate(zone, IS_BAR)) == 0

  allowed_elts = ['TRI_3', 'QUAD_4', 'PENTA_6'] if element_type == 'TRI_3' else ['QUAD_4', 'HEXA_8']
  assert all(PT.Element.CGNSName(e) in allowed_elts for e in PT.get_children_from_label(zone, 'Elements_t'))

  assert [len(e) for e in PT.Zone.get_ordered_elements_per_dim(zone)] == [0, 0, 3, 1]

  assert len(PT.get_nodes_from_predicate(zone, IS_EDGE_SUBSET)) == 0
  assert len(PT.get_nodes_from_predicate(zone, IS_FACE_SUBSET)) == 4
  assert len(PT.get_nodes_from_predicate(zone, 'GridConnectivity_t')) == 2
  assert np.all(np.abs(PT.get_node_from_name(zone, 'Translation')[1]) == [0., 0., 2.])

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("subset_as", ['BC', 'GC'])
def test_extrusion_2d_S(subset_as, comm):

  # Prepare 2D case : to test two different i,j, orientation, wanted setup is:

  #    +----+----+----+----++----+----+
  #    |    |    |    |    ||    |    |
  #    +----+----+----+----+^i---+----+  
  #    |    |    |    |    ||  j |    |
  #    +----+----+----+----++-->-+----+
  #    |    |    |    |    |
  #   j^----+----+----+----+
  #    |  i |    |    |    |
  #    +-->-+----+----+----+ 
  treeA = maia.factory.generate_dist_block((9,5), 'S', comm, origin=[0., 0.], length=[2., 1])
  treeB = maia.factory.generate_dist_block((3,6), 'S', comm, origin=[0., 0.], length=[.5, 1])
  maia.algo.scale_mesh(treeB, [-1., 1])
  maia.algo.transform_affine(treeB, rotation_center=[0,.0], rotation_angle=-0.5*np.pi, translation=[2., 0.5])
  zoneA = PT.get_node_from_label(treeA, 'Zone_t')
  zoneB = PT.get_node_from_label(treeB, 'Zone_t')
  PT.set_name(zoneA, 'Large')
  PT.set_name(zoneB, 'Small')
  # Change a  BC to EdgeCenter in zone A
  ymin = PT.get_node_from_name(zoneA, 'Ymin')
  PT.update_child(ymin, 'GridLocation', value='JEdgeCenter')
  PT.update_child(ymin, 'PointRange', value=[[1,8],[1,1]])
  MT.new_Distribution({'Index' : par_utils.uniform_distribution(8, comm)}, ymin)
  # Add a BCDS in zoneB
  ymin = PT.get_node_from_name(zoneB, 'Ymin')
  bcds = PT.new_BCDataSet(loc='Vertex', parent=ymin)
  distri = MT.distribution_value(ymin, 'Index')
  PT.new_BCData('DirichletData', fields={'field': np.arange(distri[2])[distri[0]:distri[1]]}, parent=bcds)

  # Create JN A -> B
  xmax = PT.get_node_from_name(zoneA, 'Xmax')
  PT.get_child_from_name(xmax, 'PointRange')[1][1,1] = 3  
  MT.new_Distribution({'Index' : par_utils.uniform_distribution(3, comm)}, xmax)
  zgc = PT.new_ZoneGridConnectivity(parent=zoneA)
  gc = PT.new_GridConnectivity1to1('matchLeft', 'Small', point_range=[[9,9],[3,5]], point_range_donor=[[1,3], [1,1]], transform=[2,1], parent=zgc)
  MT.new_Distribution({'Index' : par_utils.uniform_distribution(3, comm)}, gc)
  # Create JN B -> A
  zgc = PT.new_ZoneGridConnectivity(parent=zoneB)
  gc = PT.new_GridConnectivity1to1('matchRight', 'Large', point_range=[[1,3],[1,1]], point_range_donor=[[9,9], [3,5]], transform=[2,1], parent=zgc)
  MT.new_Distribution({'Index' : par_utils.uniform_distribution(3, comm)}, gc)
  PT.rm_nodes_from_name(zoneB, 'Ymax')

  tree = PT.union(treeA, treeB)
  base = PT.get_all_CGNSBase_t(tree)[0]

  # Run test
  dupl_vtx = (subset_as == 'BC')
  EXT.extrude(tree, [0., 0., 2.], comm, subset_as, dupl_vtx)

  # If volumes are > 0, i,j,k is direct
  for zone in PT.get_all_Zone_t(tree):
    assert maia.algo.geometry._compute_elements_measure(zone, 3, comm).min() > 0

  # Verification
  assert np.all(PT.get_value(base) == [3, 3])

  assert len(PT.get_nodes_from_predicate(tree, IS_EDGE_SUBSET)) == 0
  assert len(PT.get_nodes_from_predicate(tree, IS_FACE_SUBSET)) == 1

  gcs = PT.get_nodes_from_label(tree, 'GridConnectivity1to1_t')
  assert len(gcs) == 2 + 4*(subset_as=='GC')
  for gc in gcs:
    if PT.get_name(gc) in ['InitialSurface', 'ExtrudedSurface']:
      assert (PT.get_node_from_name(gc, 'Transform')[1] == [1,2,3]).all()
    else:
      assert (PT.get_node_from_name(gc, 'Transform')[1] == [2,1,-3]).all()
  
  for bc in PT.get_nodes_from_label(tree, 'BC_t'):
    if PT.Subset.GridLocation(bc) == 'Vertex' and PT.get_name(bc) not in ['InitialSurface', 'ExtrudedSurface']:
      assert (PT.get_child_from_name(bc, 'PointRange')[1][2,:] == [1,2]).all()

  bcds = PT.get_node_from_path(tree, 'Base/Small/ZoneBC/Ymin/BCDataSet')
  assert comm.allreduce(PT.get_node_from_name(bcds, 'field')[1].size) == 3*(dupl_vtx+1)
  assert (PT.get_child_from_name(bcds, 'PointRange') is None) == dupl_vtx
  