from   packaging.version import Version
import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.pred as PTp
import maia.pytree.maia as MT

from maia.utils    import test_utils as TU

from maia.algo.dist import interpolation_cons as ITP

from maia.algo.test.test_interpolation_impl import minimal_tri, union, integrated_val

@pytest_parallel.mark.parallel(2)
def test_vtx2cell(comm):
  
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      field DataArray_t R8 [4, 5, 7, 10, 8, 3]:
  """)
  tree = maia.factory.full_to_dist_tree(ftree, comm)
  cell_distri = MT.Zone.cell_distribution(PT.find_node_from_label(tree, 'Zone_t'))

  vtx_field = PT.get_np_value(PT.find_node_from_name(tree, 'field'))
  dual_vol = PT.get_np_value(PT.find_node_from_name(tree, 'DualVol24')) / 24
  it = ITP.VertexToCell(PT.get_all_Zone_t(tree), comm)
  cell_field = it._exchange_fields({'field' : [vtx_field]}, True)
  
  expected_cell_val = np.array([12, 15, 20, 21, 15]) / 3
  assert len(cell_field) == 1 and len(vals := cell_field['field']) == 1
  assert np.allclose(vals[0], expected_cell_val[cell_distri[0]:cell_distri[1]])

  # From integrated
  vtx_field *= dual_vol
  cell_vol = PT.get_np_value(PT.find_node_from_name(tree, 'Measure'))
  cell_field = it._exchange_fields({'field': [vtx_field]}, False)['field'][0]
  assert np.allclose(cell_field/cell_vol, expected_cell_val[cell_distri[0]:cell_distri[1]])

  # Check conservativity (on integrated var)
  vtx_sum = comm.allreduce(vtx_field.sum())
  cell_sum = comm.allreduce(cell_field.sum())
  assert abs(vtx_sum - cell_sum) < 1E-12

@pytest_parallel.mark.parallel(2)
def test_cell2vtx(comm):
  
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      field DataArray_t R8 [2, 6, 3, 8, 5]:
  """)
  tree = maia.factory.full_to_dist_tree(ftree, comm)
  vtx_distri = MT.Zone.vtx_distribution(PT.find_node_from_label(tree, 'Zone_t'))

  cell_field = PT.get_np_value(PT.find_node_from_name(tree, 'field'))

  it = ITP.CellToVertex(PT.get_all_Zone_t(tree), comm)
  vtx_field = it._exchange_fields({'field' : [cell_field]}, False)

  expected_vtx_val = np.array([7, 8, 9, 11, 13, 24]) / 3
  assert len(vtx_field) == 1 and len(vals := vtx_field['field']) == 1
  assert np.allclose(vals[0], expected_vtx_val[vtx_distri[0]:vtx_distri[1]])

  # From conservative
  vol = PT.get_np_value(PT.find_node_from_name(tree, 'Measure'))
  dual_vol = PT.get_np_value(PT.find_node_from_name(tree, 'DualVol24')) / 24
  cell_field_cons = cell_field / vol
  vtx_field_cons = it._exchange_fields({'field' : [cell_field_cons]}, True)['field'][0]
  assert np.allclose(vtx_field_cons*dual_vol, expected_vtx_val[vtx_distri[0]:vtx_distri[1]])
  
  # Check conservativity (on integrated var)
  _vtx_field = vtx_field['field'][0]
  vtx_sum = comm.allreduce(_vtx_field.sum())
  cell_sum = comm.allreduce(cell_field.sum())
  assert abs(vtx_sum - cell_sum) < 1E-12


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('offset', ['none', 'overlap', 'outside'])
def test_cell_cell_interpolation(offset, comm):
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      field DataArray_t R8 [2, 6, 3, 8, 5]:
      wrongfield DataArray_t R8 [0, 0, 0, 0, 0]:
  """)
  src = maia.factory.full_to_dist_tree(ftree, comm)
  tgt = maia.factory.generate_dist_block(3, 'TRI_3', comm)

  # Make some cells larger on tgt mesh
  cx = PT.get_np_value(PT.get_node_from_name(tgt, 'CoordinateX'))
  if offset == 'overlap':
    cx[cx == 1] += 0.1
  elif offset == 'outside':
    cx += 0.6

  interpolator = ITP.ConservativeDistInterpolator(PT.get_all_Zone_t(src), PT.get_all_Zone_t(tgt), comm)
  interpolator._exchange_fields('Sol', 'CellCenter', PTp.name_is('field'), is_conservative=False)

  for zone in PT.get_all_Zone_t(tgt):
    fs = PT.get_node_from_name(zone, 'Sol')
    assert PT.get_child_from_name(fs, 'field') is not None
    assert PT.get_child_from_name(fs, 'wrongfield') is None
    assert fs is not None and PT.Container.GridLocation(fs) == 'CellCenter'

  if offset == 'none': # Mass should be conserved
    src_sum = comm.allreduce(PT.find_node_from_name(src, 'field')[1].sum())
    tgt_sum = comm.allreduce(PT.find_node_from_name(tgt, 'field')[1].sum())
    assert abs(src_sum - tgt_sum) < 1E-12

  if offset == 'none':
    expected = np.array([3.75, 3.75, 1.5, 4., 2., 2.5, 3.25, 3.25])
  elif offset == 'overlap':
    expected = np.array([3.75, 3.75, 2.22857143, 4.8, 2., 2.5, 3.95844156, 3.81818182])
  else:
    expected = np.array([2.33333333, 4., 4., 4., 3.5, 3.25, 4., 4.])

  cell_distri = MT.Zone.cell_distribution(PT.get_all_Zone_t(tgt)[0])
  dsol = PT.get_np_value(PT.find_node_from_name(tgt, 'field'))
  assert np.allclose(expected[cell_distri[0]:cell_distri[1]], dsol)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('dim', [2, 3])
def test_poly_and_s_meshes(dim, comm):
  if dim == 2:
    src = maia.factory.generate_dist_block(11, 'TRI_3', comm)
    maia.algo.dist.convert_elements_to_ngon(src, comm)
    tgt = maia.factory.generate_dist_block([8, 14], 'S', comm, origin=(0.,0.))
  else:
    src = maia.factory.generate_dist_block(11, 'S', comm)
    tgt = maia.factory.generate_dist_block(7, 'Poly', comm)

  for zone in PT.get_all_Zone_t(src):
    cell_distri = MT.Zone.cell_distribution(zone)
    PT.new_FlowSolution(loc='CellCenter', fields={'gnum' : np.arange(cell_distri[0], cell_distri[1])}, parent=zone)

  interpolator = ITP.ConservativeDistInterpolator(PT.get_all_Zone_t(src), PT.get_all_Zone_t(tgt), comm)
  interpolator._exchange_fields('FlowSolution', 'CellCenter', PTp.ALWAYS_TRUE, False)

  src_sum = comm.allreduce(PT.get_node_from_name(src, 'gnum')[1].sum())
  tgt_sum = comm.allreduce(PT.get_node_from_name(tgt, 'gnum')[1].sum())
  assert abs(src_sum - tgt_sum) /  src_sum < 1E-3 # TODO restore 1E-12 when PDM / optim is OK

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('in_loc', ['CellCenter', 'Vertex'])
@pytest.mark.parametrize('out_loc', ['CellCenter', 'Vertex'])
def test_vertex_fields(in_loc, out_loc, comm):
  
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    CellCenterSol DiscreteData_t:
      GridLocation GridLocation_t "CellCenter":
      field DataArray_t R8 [2, 6, 3, 8, 5]:
    VertexSol DiscreteData_t:
      field DataArray_t R8 [4, 5, 7, 10, 8, 3]:
  """)
  src = maia.factory.full_to_dist_tree(ftree, comm)
  tgt = maia.factory.generate_dist_block(3, 'TRI_3', comm)

  # Here we just check that output is produced at good location
  # (results already checked in other tests)

  interpolator = ITP.ConservativeDistInterpolator(PT.get_all_Zone_t(src), PT.get_all_Zone_t(tgt), comm)
  interpolator._exchange_fields(in_loc+'Sol', out_loc, PTp.ALWAYS_TRUE, is_conservative=False)

  for zone in PT.get_all_Zone_t(tgt):
    fs = PT.get_node_from_name(zone, in_loc+'Sol')
    assert PT.Container.GridLocation(fs) == out_loc and PT.get_label(fs) == 'DiscreteData_t'

@pytest.mark.skipif(TU.PDM_VERSION < Version('2.8'), reason="Require PDM fixes on PtP")
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('elt_kind', ["Poly", "Standard"])
def test_multidom(dim, elt_kind, comm):
  
  if dim == 2:
    src = maia.factory.generate_dist_block(11, 'TRI_3', comm)

    tgt1 = maia.factory.generate_dist_block(11, 'QUAD_4', comm, length=(.5, 1))
    tgt2 = maia.factory.generate_dist_block(11, 'QUAD_4', comm, origin=(.5,0,0), length=(.5, 1))
    tgt = union(tgt1, tgt2)
    if elt_kind == 'Poly':
      maia.algo.dist.convert_elements_to_ngon(tgt, comm)
    
  else:

    src = maia.io.file_to_dist_tree(TU.mesh_dir / 'S_twoblocks.yaml', comm)
    tgt = PT.deep_copy(src)
    maia.algo.dist.convert_s_to_u(tgt, elt_kind, comm)

  for idom,zone in enumerate(PT.get_all_Zone_t(src)):
    cell_distri = MT.Zone.cell_distribution(zone)
    PT.new_FlowSolution(loc='CellCenter', fields={'gnum' : 1000*(idom) + np.arange(cell_distri[0], cell_distri[1])}, parent=zone)

  maia.algo.interpolate(src, tgt, comm, ['FlowSolution'], 'CellCenter', strategy='Intersection')

  src_sum = integrated_val(src, 'gnum', comm)
  tgt_sum = integrated_val(tgt, 'gnum', comm)
  assert abs(src_sum - tgt_sum) /  src_sum < 1E-12

@pytest.mark.skipif(TU.PDM_VERSION < Version('2.8'), reason="Require PDM fixes on PtP")
def test_multidom_gnum_offsets(comm):
  src1 = maia.factory.generate_dist_block(6, 'TRI_3', comm, length=.5)
  src2 = maia.factory.generate_dist_block(6, 'TRI_3', comm, origin=(.5,0,0), length=.5)
  src3 = maia.factory.generate_dist_block(6, 'TRI_3', comm, origin=(.0,.5,0), length=.5)
  src4 = maia.factory.generate_dist_block(6, 'TRI_3', comm, origin=(.5,.5,0), length=.5)
  src = union(src1, src2, src3, src4)

  tgt1 = maia.factory.generate_dist_block(11, 'QUAD_4', comm, length=(.5, 1))
  tgt2 = maia.factory.generate_dist_block(11, 'QUAD_4', comm, origin=(.5,0,0), length=(1, 1))
  tgt = union(tgt1, tgt2)

  for idom,zone in enumerate(PT.get_all_Zone_t(src)):
    cell_distri = MT.Zone.cell_distribution(zone)
    PT.new_FlowSolution(loc='CellCenter', fields={'gnum' : 1000*(idom) + np.arange(cell_distri[0], cell_distri[1])}, parent=zone)

  itp = maia.algo.create_interpolator(src, tgt, comm, 'CellCenter', 'CellCenter', strategy='Intersection')
  itp._exchange_fields('FlowSolution', 'CellCenter', PTp.ALWAYS_TRUE, False)
  tgt_sum = comm.allreduce(sum([PT.get_np_value(n).sum() for n in PT.get_nodes_from_name(tgt, 'gnum')]))
  assert abs(tgt_sum - 507800) < 1E-3

@pytest.mark.skipif(TU.PDM_VERSION < Version('2.8'), reason="Require PDM fixes on PtP")
@pytest_parallel.mark.parallel(2)
def test_multidom_vtx(comm):
  src = union(maia.factory.generate_dist_block(5, 'TRI_3', comm, length=(.5, 1), origin=(0,0,0)),
              maia.factory.generate_dist_block(5, 'TRI_3', comm, length=(.5, 1), origin=(.5,0,0)))
  tgt = union(maia.factory.generate_dist_block(9, 'TRI_3', comm, length=(1, 1./3), origin=(0,0)),
              maia.factory.generate_dist_block(9, 'TRI_3', comm, length=(1, 1./3), origin=(0,1./3)),
              maia.factory.generate_dist_block(9, 'TRI_3', comm, length=(1, 1./3), origin=(0,2./3)))

  for idom, zone in enumerate(PT.get_all_Zone_t(src)):
    cell_distri = MT.Zone.cell_distribution(zone)
    PT.new_FlowSolution(loc='CellCenter', fields={'gnum' : 100*(idom) + np.arange(cell_distri[0], cell_distri[1])}, parent=zone)

  itp = maia.algo.create_interpolator(src, tgt, comm, 'CellCenter', 'CellCenter', strategy='Intersection')
  itp.exchange_fields('FlowSolution')

  src_sum = integrated_val(src, 'gnum', comm)
  tgt_sum = integrated_val(tgt, 'gnum', comm)
  assert abs(src_sum - tgt_sum) /  src_sum < 1E-3 # TODO restore 1E-12 when PDM / optim is OK

