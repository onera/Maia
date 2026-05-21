from   packaging.version import Version
import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP
from maia.utils    import test_utils as TU

from maia.algo.part import interpolation_cons as ITP

minimal_tri = """
  zone Zone_t [[6, 5, 0]]:
    ZoneType ZoneType_t "Unstructured":
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [0, 0, 0, 1, 1, 0.5]:
      CoordinateY DataArray_t R8 [1, 0.5, 0, 0, 1, 0.5]:
      CoordinateZ DataArray_t R8 [0, 0, 0, 0, 0, 0]:
    TRI Elements_t [5, 0]:
      ElementRange IndexRange_t [1, 5]:
      ElementConnectivity DataArray_t [1,2,6, 2,3,6, 3,4,6, 4,5,6, 5,1,6]:
    Geometry_0d DiscreteData_t:
      DualVol24 DataArray_t R8 [3, 2, 3, 4, 4, 8]:
    Geometry_2d DiscreteData_t:
      GridLocation GridLocation_t "CellCenter":
      Measure DataArray_t R8 [0.125, 0.125, 0.25, 0.25, 0.25]:
"""

def union(*trees):
  for i,tree in enumerate(trees):
    PT.set_name(PT.get_node_from_label(tree, 'Zone_t'), f'Zone_{i}')
  return PT.union(*trees)

@pytest_parallel.mark.parallel(2)
def test_vtx2cell(comm):
  
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      field DataArray_t R8 [4, 5, 7, 10, 8, 3]:
  """)
  tree = maia.factory.full_to_dist_tree(ftree, comm)
  cell_gnum = [np.array([1,5]), np.array([2,3,4])][comm.rank]
  ptree = TU.portable_partitioning(tree, [cell_gnum], comm, data_transfer='FIELDS')

  vtx_field = PT.get_np_value(PT.find_node_from_name(ptree, 'field'))
  dual_vol = PT.get_np_value(PT.find_node_from_name(ptree, 'DualVol24')) / 24
  it = ITP.VertexToCell([PT.get_all_Zone_t(ptree)], comm)
  cell_field = it._exchange_fields({'field' : [vtx_field]}, True)
  
  expected_cell_val = np.array([12, 15, 20, 21, 15]) / 3
  assert len(cell_field) == 1 and len(vals := cell_field['field']) == 1
  assert np.allclose(vals[0], expected_cell_val[cell_gnum-1])

  # From integrated
  vtx_field *= dual_vol
  cell_vol = PT.get_np_value(PT.find_node_from_name(ptree, 'Measure'))
  cell_field = it._exchange_fields({'field': [vtx_field]}, False)['field'][0]
  assert np.allclose(cell_field/cell_vol, expected_cell_val[cell_gnum-1])

  # Check conservativity (on integrated var)
  vtx_gnum = MT.Zone.vtx_globalnumbering(PT.get_all_Zone_t(ptree)[0])
  vtx_sum = comm.allreduce(EP.part_to_block(vtx_field, np.array([0,3,6]), vtx_gnum-1, comm).sum())
  cell_sum = comm.allreduce((cell_field).sum())
  assert abs(vtx_sum - cell_sum) < 1E-12

@pytest_parallel.mark.parallel(2)
def test_cell2vtx(comm):
  
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      field DataArray_t R8 [2, 6, 3, 8, 5]:
  """)
  tree = maia.factory.full_to_dist_tree(ftree, comm)
  cell_gnum = [np.array([1,4]), np.array([2,3,5])][comm.rank]
  ptree = TU.portable_partitioning(tree, [cell_gnum], comm, data_transfer='FIELDS')

  vtx_gnum = MT.Zone.vtx_globalnumbering(PT.get_all_Zone_t(ptree)[0])
  cell_field = PT.get_np_value(PT.find_node_from_name(ptree, 'field'))

  it = ITP.CellToVertex([PT.get_all_Zone_t(ptree)], comm)
  vtx_field = it._exchange_fields({'field' : [cell_field]}, False)

  expected_vtx_val = np.array([7, 8, 9, 11, 13, 24]) / 3
  assert len(vtx_field) == 1 and len(vals := vtx_field['field']) == 1
  assert np.allclose(vals[0], expected_vtx_val[vtx_gnum-1])

  # From conservative
  vol = PT.get_np_value(PT.find_node_from_name(ptree, 'Measure'))
  dual_vol = PT.get_np_value(PT.find_node_from_name(ptree, 'DualVol24')) / 24
  cell_field_cons = cell_field / vol
  vtx_field_cons = it._exchange_fields({'field' : [cell_field_cons]}, True)['field'][0]
  assert np.allclose(vtx_field_cons*dual_vol, expected_vtx_val[vtx_gnum-1])
  
  # Check conservativity (on integrated var)
  _vtx_field = vtx_field['field'][0]
  vtx_sum = comm.allreduce(EP.part_to_block(_vtx_field, np.array([0,3,6]), vtx_gnum-1, comm).sum())
  cell_sum = comm.allreduce((cell_field).sum())
  assert abs(vtx_sum - cell_sum) < 1E-12


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('offset', ['none', 'overlap', 'outside'])
def test_cell_cell_interpolation(offset, comm):
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      field DataArray_t R8 [2, 6, 3, 8, 5]:
  """)
  src = maia.factory.full_to_dist_tree(ftree, comm)
  tgt = maia.factory.generate_dist_block(3, 'TRI_3', comm)

  # Make some cells larger on tgt mesh
  cx = PT.get_np_value(PT.get_node_from_name(tgt, 'CoordinateX'))
  if offset == 'overlap':
    cx[cx == 1] += 0.1
  elif offset == 'outside':
    cx += 0.6

  # Try fancy partitioning in some cases
  if offset == 'none':
    cell_gnum = [np.array([1,5])] if comm.rank == 0 else [np.array([2,3]),np.array([4])]
    tgt_split_w = {'Base/zone' : [1.]} if comm.rank == 0 else {}
  elif offset == 'overlap':
    cell_gnum = [] if comm.rank == 0 else [np.array([1,2,3,4,5])]
    # tgt_split_w = {'Base/zone' : [.25,.25]} if comm.rank == 0 else {'Base/zone' : [.5]} # Warning bug in Cython / PTP. Uncomment if fixed
    tgt_split_w = {'Base/zone' : [1.]} if comm.rank == 0 else {}
  else:
    cell_gnum = [np.array([1,2,3,4,5])] if comm.rank == 0 else []
    tgt_split_w = {'Base/zone' : [1.]} if comm.rank == 0 else {}
  
  psrc = TU.portable_partitioning(src, cell_gnum, comm, data_transfer='FIELDS')
  ptgt = maia.factory.partition_dist_tree(tgt, comm, zone_to_parts=tgt_split_w)

  interpolator = ITP.ConservativePartInterpolator([PT.get_all_Zone_t(psrc)], [PT.get_all_Zone_t(ptgt)], comm)
  interpolator.exchange_fields('Sol', 'CellCenter', is_conservative=False)

  for zone in PT.get_all_Zone_t(ptgt):
    fs = PT.get_node_from_name(zone, 'Sol')
    assert fs is not None and PT.Container.GridLocation(fs) == 'CellCenter'

  maia.transfer.part_tree_to_dist_tree_all(tgt, ptgt, comm)

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
@pytest.mark.parametrize('dim', [2,3])
def test_poly_and_s_meshes(dim, comm):
  if dim == 2:
    src = maia.factory.generate_dist_block(11, 'TRI_3', comm)
    maia.algo.dist.convert_elements_to_ngon(src, comm)
    tgt = maia.factory.generate_dist_block([8, 14], 'S', comm, origin=(0.,0.))
  else:
    src = maia.factory.generate_dist_block(11, 'S', comm)
    tgt = maia.factory.generate_dist_block(7, 'Poly', comm)

  #  NB : preserve_orientation = True seems required for NG meshes
  psrc = maia.factory.partition_dist_tree(src, comm)
  ptgt = maia.factory.partition_dist_tree(tgt, comm)

  for zone in PT.get_all_Zone_t(psrc):
    PT.new_FlowSolution(loc='CellCenter', fields={'gnum' : MT.Zone.cell_globalnumbering(zone)}, parent=zone)

  interpolator = ITP.ConservativePartInterpolator([PT.get_all_Zone_t(psrc)], [PT.get_all_Zone_t(ptgt)], comm)
  interpolator.exchange_fields('FlowSolution', 'CellCenter', False)

  maia.transfer.part_tree_to_dist_tree_all(tgt, ptgt, comm)
  maia.transfer.part_tree_to_dist_tree_all(src, psrc, comm)
  src_sum = comm.allreduce(sum([PT.get_np_value(n).sum() for n in PT.get_nodes_from_name(psrc, 'gnum')]))
  tgt_sum = comm.allreduce(sum([PT.get_np_value(n).sum() for n in PT.get_nodes_from_name(ptgt, 'gnum')]))
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

  psrc = maia.factory.partition_dist_tree(src, comm, data_transfer='FIELDS')
  ptgt = maia.factory.partition_dist_tree(tgt, comm)

  # Here we just check that output is produced at good location
  # (results already checked in other tests)

  interpolator = ITP.ConservativePartInterpolator([PT.get_all_Zone_t(psrc)], [PT.get_all_Zone_t(ptgt)], comm)
  interpolator.exchange_fields(in_loc+'Sol', out_loc, is_conservative=False)

  for zone in PT.get_all_Zone_t(ptgt):
    fs = PT.get_node_from_name(zone, in_loc+'Sol')
    assert PT.Container.GridLocation(fs) == out_loc and PT.get_label(fs) == 'DiscreteData_t'

def test_from_api(comm):
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      field DataArray_t R8 [2, 6, 3, 8, 5]:
  """)
  src = maia.factory.full_to_dist_tree(ftree, comm)
  tgt = maia.factory.generate_dist_block(3, 'TRI_3', comm)

  psrc = maia.factory.partition_dist_tree(src, comm, data_transfer='FIELDS')
  ptgt = maia.factory.partition_dist_tree(tgt, comm)
  maia.algo.interpolate(psrc, ptgt, comm, ['Sol'], 'Vertex', strategy='Intersection', is_conservative=False)
  assert PT.get_node_from_name(ptgt, 'Sol') is not None

@pytest.mark.skipif(TU.PDM_VERSION < Version('2.8.dev'), reason="Require PDM fixes on PtP")
@pytest.mark.parametrize('dim', [2,3])
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

  psrc = maia.factory.partition_dist_tree(src, comm, data_transfer='FIELDS')
  ptgt = maia.factory.partition_dist_tree(tgt, comm)

  for idom,zone in enumerate(PT.get_all_Zone_t(psrc)):
    PT.new_FlowSolution(loc='CellCenter', fields={'gnum' : 1000*(idom) + MT.Zone.cell_globalnumbering(zone)}, parent=zone)

  maia.algo.interpolate(psrc, ptgt, comm, ['FlowSolution'], 'CellCenter', strategy='Intersection', is_conservative=False)

  src_sum = comm.allreduce(sum([PT.get_np_value(n).sum() for n in PT.get_nodes_from_name(psrc, 'gnum')]))
  tgt_sum = comm.allreduce(sum([PT.get_np_value(n).sum() for n in PT.get_nodes_from_name(ptgt, 'gnum')]))
  assert abs(src_sum - tgt_sum) /  src_sum < 1E-12

@pytest.mark.skipif(TU.PDM_VERSION < Version('2.8.dev'), reason="Require PDM fixes on PtP")
@pytest_parallel.mark.parallel(2)
def test_multidom_vtx(comm):
  src = union(maia.factory.generate_dist_block(5, 'TRI_3', comm, length=(.5, 1), origin=(0,0,0)),
              maia.factory.generate_dist_block(5, 'TRI_3', comm, length=(.5, 1), origin=(.5,0,0)))
  tgt = union(maia.factory.generate_dist_block(9, 'TRI_3', comm, length=(1, 1./3), origin=(0,0)),
              maia.factory.generate_dist_block(9, 'TRI_3', comm, length=(1, 1./3), origin=(0,1./3)),
              maia.factory.generate_dist_block(9, 'TRI_3', comm, length=(1, 1./3), origin=(0,2./3)))

  psrc = maia.factory.partition_dist_tree(src, comm)
  ptgt = maia.factory.partition_dist_tree(tgt, comm)

  for zone in PT.get_all_Zone_t(psrc):
    idom = int(MT.conv.get_part_prefix(PT.get_name(zone))[-1])
    PT.new_FlowSolution(loc='CellCenter', fields={'gnum' : 100*(idom) + MT.Zone.cell_globalnumbering(zone)}, parent=zone)

  maia.algo.interpolate(psrc, ptgt, comm, ['FlowSolution'], 'CellCenter', strategy='Intersection', is_conservative=False)

  src_sum = comm.allreduce(sum([PT.get_np_value(n).sum() for n in PT.get_nodes_from_name(psrc, 'gnum')]))
  tgt_sum = comm.allreduce(sum([PT.get_np_value(n).sum() for n in PT.get_nodes_from_name(ptgt, 'gnum')]))
  assert abs(src_sum - tgt_sum) /  src_sum < 1E-3 # TODO restore 1E-12 when PDM / optim is OK

