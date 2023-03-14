from mpi4py import MPI
import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia
import maia.pytree as PT

from maia.algo.part import extract_part as EP

def sample_part_tree(comm):
  dist_tree = maia.factory.generate_dist_block(3, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  return part_tree

@mark_mpi_test(2)
@pytest.mark.parametrize("dim", [0,2,3])
def test_extract_part_simple(dim, sub_comm):
  part_tree = sample_part_tree(sub_comm)

  pl = np.array([1,2], np.int32)
  if dim == 3: 
    pl += PT.Zone.n_face(PT.get_all_Zone_t(part_tree)[0])

  ex_zone, ptp_data = EP.extract_part_one_domain(PT.get_all_Zone_t(part_tree), \
      [pl], dim, sub_comm)

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

@mark_mpi_test(2)
def test_extract_part_obj(sub_comm):
  dist_tree = maia.factory.generate_dist_block(3, "Poly", sub_comm)
  zone_to_parts = maia.factory.partitioning.compute_regular_weights(\
      dist_tree, sub_comm, 2*(sub_comm.Get_rank() == 1)) #2 parts on proc 1, 0 on proc 0
  part_tree = maia.factory.partition_dist_tree(dist_tree, sub_comm, zone_to_parts=zone_to_parts)

  if sub_comm.Get_rank() == 0:
    pl = []
  else:
    pl = [np.array([1,2], np.int32), np.array([1,2], np.int32)]

  extractor = EP.Extractor(part_tree, [pl], "FaceCenter", sub_comm)
  extracted_tree = extractor.get_extract_part_tree()

  assert len(PT.get_all_Zone_t(extracted_tree)) == 1
  assert (PT.get_all_CGNSBase_t(extracted_tree)[0][1] == [2,3]).all()
  
@mark_mpi_test(2)
@pytest.mark.parametrize('partial', [False, True])
def test_exch_field(partial, sub_comm):
  part_tree = sample_part_tree(sub_comm)

  # Add field
  for zone in PT.get_all_Zone_t(part_tree):
    n_vtx = PT.Zone.n_vtx(zone)
    gnum = PT.maia.getGlobalNumbering(zone, 'Vertex')[1]
    if partial: #Take one over two
      pl = np.arange(1,n_vtx+1)[::2].astype(np.int32).reshape((1,-1))
      PT.new_ZoneSubRegion('FlowSol', loc="Vertex", point_list=pl, fields={'gnum': gnum[::2]}, parent=zone)
    else:
      PT.new_FlowSolution('FlowSol', loc="Vertex", fields={'gnum': gnum}, parent=zone)

  extractor = EP.Extractor(part_tree, [[np.array([1,2,3], np.int32)]], "Vertex", sub_comm)
  extractor.exchange_fields(['FlowSol'])
  extr_tree = extractor.get_extract_part_tree()

  extr_sol = PT.get_node_from_name(extr_tree, 'FlowSol')
  assert PT.Subset.GridLocation(extr_sol) == 'Vertex'
  if partial:
    assert PT.get_label(extr_sol) == 'ZoneSubRegion_t'
    pl = PT.get_node_from_name(extr_sol, 'PointList')[1][0]
    data = PT.get_node_from_name(extr_sol, 'gnum')[1]
    assert np.array_equal(extractor.exch_tool_box[0]['parent_elt']['Vertex'][pl-1], data)
  else:
    assert PT.get_label(extr_sol) == 'FlowSolution_t'
    assert np.array_equal(PT.get_node_from_name(extr_sol, 'gnum')[1],
                          extractor.exch_tool_box[0]['parent_elt']['Vertex'])

@mark_mpi_test(2)
def test_exch_field_from_bc_zsr(sub_comm):
  part_tree = sample_part_tree(sub_comm)

  # Add field
  for zone in PT.get_all_Zone_t(part_tree):
    gnum = PT.maia.getGlobalNumbering(PT.get_node_from_name(zone, 'NGonElements'), 'Element')[1]
    bc_n = PT.get_child_from_predicates(zone, 'ZoneBC_t/Xmin')
    if bc_n is not None:
      bc_pl   = PT.get_value(PT.get_node_from_name(bc_n, "PointList"))
      bc_gnum = gnum[bc_pl[0]-1]
      PT.new_ZoneSubRegion('ZSR_Xmin', bc_name="Xmin", fields={'gnum': bc_gnum}, parent=zone)

  extractor = EP.Extractor(part_tree, [bc_pl], "FaceCenter", sub_comm)
  extractor.exchange_fields(['ZSR_Xmin'])
  extr_tree = extractor.get_extract_part_tree()

  extr_sol = PT.get_node_from_name(extr_tree, 'ZSR_Xmin')
  assert PT.get_label(extr_sol) == 'ZoneSubRegion_t'
  assert PT.Subset.GridLocation(extr_sol) == 'CellCenter'
  pl    = PT.get_node_from_name(extr_sol, 'PointList')[1][0]
  data  = PT.get_node_from_name(extr_sol, 'gnum')[1]
  assert np.array_equal(extractor.exch_tool_box[0]['parent_elt']['FaceCenter'][pl-1], data)


@mark_mpi_test(3)
def test_zsr_api(sub_comm):
  dist_tree = maia.factory.generate_dist_block(4, "Poly", sub_comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, sub_comm)

  if sub_comm.rank != 1:
    for zone in PT.get_all_Zone_t(part_tree):
      n_face = PT.Zone.n_face(zone)
      pl = np.array([1,2], dtype=np.int32).reshape((1,-1)) + n_face
      PT.new_ZoneSubRegion('ToExtract', loc='CellCenter', point_list=pl, parent=zone)

  extracted_tree = EP.extract_part_from_zsr(part_tree, 'ToExtract', sub_comm)

  n_cell_extr = PT.Zone.n_cell(PT.get_all_Zone_t(extracted_tree)[0])
  assert sub_comm.allreduce(n_cell_extr, op=MPI.SUM) == 4


@mark_mpi_test(3)
def test_bc_name_api(sub_comm):
  dist_tree = maia.factory.generate_dist_block(4, "Poly", sub_comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, sub_comm)

  extracted_tree = EP.extract_part_from_bc_name(part_tree, 'Xmin', sub_comm)

  n_cell_extr = PT.Zone.n_cell(PT.get_all_Zone_t(extracted_tree)[0])
  assert sub_comm.allreduce(n_cell_extr, op=MPI.SUM) == 9
