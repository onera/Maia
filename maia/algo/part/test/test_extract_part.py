from mpi4py import MPI
import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT

from maia.algo.part import extract_part as EP

def sample_part_tree(comm):
  dist_tree = maia.factory.generate_dist_block(3, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  return part_tree

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("dim", [0,2,3])
def test_extract_part_simple(dim, comm):
  part_tree = sample_part_tree(comm)

  pl = np.array([[1,2]], np.int32)
  if dim == 3: 
    pl += PT.Zone.n_face(PT.get_all_Zone_t(part_tree)[0])

  ex_zone, ptp_data = EP.extract_part_one_domain_u(PT.get_all_Zone_t(part_tree), \
      [pl], dim, comm)
  assert len(ex_zone)==1
  ex_zone = ex_zone[0]
  if dim == 0:
    assert PT.Zone.n_vtx(ex_zone) == 2
    assert PT.Zone.n_cell(ex_zone) == 0
  elif dim == 2:
    assert PT.Zone.n_vtx(ex_zone) == 6
    assert PT.Zone.n_cell(ex_zone) == 2
  elif dim == 3:
    assert PT.Zone.n_vtx(ex_zone) == 12
    assert PT.Zone.n_cell(ex_zone) == 2
    assert ptp_data['part_to_part']["CellCenter"] is not None

  assert ptp_data['part_to_part']["Vertex"] is not None

@pytest_parallel.mark.parallel(2)
def test_extract_part_obj(comm):
  dist_tree = maia.factory.generate_dist_block(3, "Poly", comm)
  zone_to_parts = maia.factory.partitioning.compute_regular_weights(\
      dist_tree, comm, 2*(comm.Get_rank() == 1)) #2 parts on proc 1, 0 on proc 0
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts)

  if comm.Get_rank() == 0:
    pl = []
  else:
    pl = [np.array([[1,2]], np.int32), np.array([[1,2]], np.int32)]

  extractor = EP.Extractor(part_tree, [pl], "FaceCenter", comm)
  extracted_tree = extractor.get_extract_part_tree()

  assert len(PT.get_all_Zone_t(extracted_tree)) == 1
  assert (PT.get_all_CGNSBase_t(extracted_tree)[0][1] == [2,3]).all()
  
@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('partial', [False, True])
def test_exch_field(partial, comm):
  part_tree = sample_part_tree(comm)

  # Add field
  for zone in PT.get_all_Zone_t(part_tree):
    n_vtx = PT.Zone.n_vtx(zone)
    gnum = PT.maia.getGlobalNumbering(zone, 'Vertex')[1]
    if partial: #Take one over two
      pl = np.arange(1,n_vtx+1)[::2].astype(np.int32).reshape((1,-1))
      PT.new_ZoneSubRegion('FlowSol', loc="Vertex", point_list=pl, fields={'gnum': gnum[::2]}, parent=zone)
    else:
      PT.new_FlowSolution('FlowSol', loc="Vertex", fields={'gnum': gnum}, parent=zone)

  extractor = EP.Extractor(part_tree, [[np.array([[1,2,3]], np.int32)]], "Vertex", comm)
  extractor.exchange_fields(['FlowSol'])
  extr_tree = extractor.get_extract_part_tree()

  extr_sol = PT.get_node_from_name(extr_tree, 'FlowSol')
  assert PT.Subset.GridLocation(extr_sol) == 'Vertex'
  if partial:
    assert PT.get_label(extr_sol) == 'ZoneSubRegion_t'
    pl = PT.get_node_from_name(extr_sol, 'PointList')[1][0]
    data = PT.get_node_from_name(extr_sol, 'gnum')[1]
    assert np.array_equal(extractor.exch_tool_box['Base/zone']['parent_elt']['Vertex'][pl-1], data)
  else:
    assert PT.get_label(extr_sol) == 'FlowSolution_t'
    assert np.array_equal(PT.get_node_from_name(extr_sol, 'gnum')[1],
                          extractor.exch_tool_box['Base/zone']['parent_elt']['Vertex'])

@pytest_parallel.mark.parallel(2)
def test_exch_field_from_bc_zsr(comm):
  part_tree = sample_part_tree(comm)

  # Add field
  for zone in PT.get_all_Zone_t(part_tree):
    gnum = PT.maia.getGlobalNumbering(PT.get_node_from_name(zone, 'NGonElements'), 'Element')[1]
    bc_n = PT.get_child_from_predicates(zone, 'ZoneBC_t/Xmin')
    if bc_n is not None:
      bc_pl   = PT.get_value(PT.get_node_from_name(bc_n, "PointList"))
      bc_gnum = gnum[bc_pl[0]-1]
      PT.new_ZoneSubRegion('ZSR_Xmin', bc_name="Xmin", fields={'gnum': bc_gnum}, parent=zone)

  extractor = EP.Extractor(part_tree, [[bc_pl]], "FaceCenter", comm)
  extractor.exchange_fields(['ZSR_Xmin'])
  extr_tree = extractor.get_extract_part_tree()

  extr_sol = PT.get_node_from_name(extr_tree, 'ZSR_Xmin')
  assert PT.get_label(extr_sol) == 'ZoneSubRegion_t'
  assert PT.Subset.GridLocation(extr_sol) == 'CellCenter'
  pl    = PT.get_node_from_name(extr_sol, 'PointList')[1][0]
  data  = PT.get_node_from_name(extr_sol, 'gnum')[1]
  assert np.array_equal(extractor.exch_tool_box['Base/zone']['parent_elt']['FaceCenter'][pl-1], data)


@pytest_parallel.mark.parallel(3)
def test_zsr_api(comm):
  dist_tree = maia.factory.generate_dist_block(4, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  if comm.rank != 1:
    for zone in PT.get_all_Zone_t(part_tree):
      n_face = PT.Zone.n_face(zone)
      pl = np.array([1,2], dtype=np.int32).reshape((1,-1)) + n_face
      zsr_n = PT.new_ZoneSubRegion('ToExtract', loc='CellCenter', point_list=pl, parent=zone)
  extracted_tree = EP.extract_part_from_zsr(part_tree, 'ToExtract', comm)

  n_cell_extr = PT.Zone.n_cell(PT.get_all_Zone_t(extracted_tree)[0])
  assert comm.allreduce(n_cell_extr, op=MPI.SUM) == 4


@pytest_parallel.mark.parallel(3)
def test_bc_name_api(comm):
  dist_tree = maia.factory.generate_dist_block(4, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  extracted_tree = EP.extract_part_from_bc_name(part_tree, 'Xmin', comm)

  n_cell_extr = PT.Zone.n_cell(PT.get_all_Zone_t(extracted_tree)[0])
  assert comm.allreduce(n_cell_extr, op=MPI.SUM) == 9


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("dim_zsr", ["FaceCenter", "CellCenter"])
def test_from_fam_api(dim_zsr, comm):
  dist_tree = maia.factory.generate_dist_block(4, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  part_zone = PT.get_node_from_label(part_tree, 'Zone_t')
  bc_n = PT.get_node_from_name(part_zone, 'Xmin')
  PT.new_node('FamilyName', label='FamilyName_t', value='EXTRACT', parent=bc_n)
  
  if dim_zsr=="FaceCenter":
    if PT.get_node_from_name(part_zone, 'Xmax') is not None:
      zsr_n = PT.new_ZoneSubRegion("ZSR", bc_name='Xmax', family='EXTRACT', parent=part_zone)
    extracted_tree = EP.extract_part_from_family(part_tree, 'EXTRACT', comm)
    n_cell_extr = PT.Zone.n_cell(PT.get_all_Zone_t(extracted_tree)[0])
    assert comm.allreduce(n_cell_extr, op=MPI.SUM) == 18
  
  elif dim_zsr=="CellCenter":
    zsr_n = PT.new_ZoneSubRegion("ZSR", loc=dim_zsr,
      point_list=np.array([[1]], dtype=np.int32), family='EXTRACT', parent=part_zone)
    with pytest.raises(ValueError):
      extracted_tree = EP.extract_part_from_family(part_tree, 'EXTRACT', comm)


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("valid", [False, True])
def test_from_fam_zsr_api(valid, comm):
  dist_tree = maia.factory.generate_dist_block(4, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  part_zone = PT.get_node_from_label(part_tree, 'Zone_t')

  bc_n = PT.get_node_from_name(part_zone, 'Xmin')
  if bc_n is not None :
    zsr_n = PT.new_ZoneSubRegion("ZSR_Xmin", bc_name='Xmin', family='EXTRACT', parent=part_zone)

  bc_n = PT.get_node_from_name(part_zone, 'Xmax')
  if bc_n is not None :
    zsr_n = PT.new_ZoneSubRegion("ZSR_Xmax", bc_name='Xmax', family='EXTRACT', parent=part_zone)
    if not valid:
      PT.set_value(PT.get_child_from_name(bc_n ,'GridLocation'), 'Vertex')
  
  if valid:
    extracted_tree = EP.extract_part_from_family(part_tree, 'EXTRACT', comm)
    n_cell_extr = PT.Zone.n_cell(PT.get_all_Zone_t(extracted_tree)[0])
    assert comm.allreduce(n_cell_extr, op=MPI.SUM) == 18
  else:
    with pytest.raises(ValueError):
      extracted_tree = EP.extract_part_from_family(part_tree, 'EXTRACT', comm)