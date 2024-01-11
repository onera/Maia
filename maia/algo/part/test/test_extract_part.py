from mpi4py import MPI
import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT
from   maia.utils import s_numbering

from maia.algo.part import extract_part as EP

def sample_part_tree(cgns_name, comm, bc_loc='Vertex'):
  if cgns_name=='Structured':
    dist_tree = maia.factory.dcube_generator.dcube_struct_generate(5, 1., [0.,0.,0.], comm, bc_location=bc_loc)
  else:
    dist_tree = maia.factory.generate_dist_block(3, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  return part_tree

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("location", ['Vertex','FaceCenter','CellCenter'])
def test_extract_part_simple_u(location, comm):
  part_tree = sample_part_tree('Poly', comm)

  pl = np.array([[1,2]], np.int32)
  if location=='CellCenter': 
    pl += PT.Zone.n_face(PT.get_all_Zone_t(part_tree)[0])

  ex_zone, ptp_data = EP.extract_part_one_domain_u(PT.get_all_Zone_t(part_tree), \
      [pl], location, comm)
  assert len(ex_zone)==1
  ex_zone = ex_zone[0]
  if location=='Vertex':
    assert PT.Zone.n_vtx(ex_zone) == 2
    assert PT.Zone.n_cell(ex_zone) == 0
  elif location=='FaceCenter':
    assert PT.Zone.n_vtx(ex_zone) == 6
    assert PT.Zone.n_cell(ex_zone) == 2
  elif location=='CellCenter':
    assert PT.Zone.n_vtx(ex_zone) == 12
    assert PT.Zone.n_cell(ex_zone) == 2
    assert ptp_data['part_to_part']["CellCenter"] is not None

  assert ptp_data['part_to_part']["Vertex"] is not None

@pytest_parallel.mark.parallel([2])
@pytest.mark.parametrize("bc_loc" , ['Vertex','FaceCenter'])
def test_extract_part_simple_s(bc_loc, comm):
  part_tree = sample_part_tree('Structured', comm, bc_loc)

  location = 'Vertex' if bc_loc=='Vertex' else 'KFaceCenter'
  pr = PT.get_value(PT.get_child_from_predicates(part_tree, f'CGNSBase_t/Zone_t/ZoneBC_t/Zmax/PointRange'))
  ex_zones, etb_zones = EP.extract_part_one_domain_s(PT.get_all_Zone_t(part_tree), \
      [pr], location, comm)

  assert PT.Zone.n_vtx(ex_zones[0]) == 15
  assert PT.Zone.n_cell(ex_zones[0]) == 8

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("bc_loc" , ['Vertex','FaceCenter'])
def test_extract_part_simple_s_from_api(bc_loc, comm):

  dist_tree = maia.factory.dcube_generator.dcube_struct_generate(10, 1., [0.,0.,0.], comm, bc_location=bc_loc)
  part_opts = maia.factory.partitioning.compute_regular_weights(dist_tree, comm, n_part=4)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=part_opts)

  # > Initialize flow solution
  for part_zone in PT.get_all_Zone_t(part_tree):
    cx, _, _ = PT.Zone.coordinates(part_zone)
    PT.new_FlowSolution('FlowSol#Vtx', loc='Vertex', fields={'cx':cx}, parent=part_zone)

  for bc_name in ['Xmin','Xmax','Ymin','Ymax','Zmin','Zmax']:
    extract_part_tree = EP.extract_part_from_bc_name(part_tree, bc_name, comm, containers_name=['FlowSol#Vtx'])
    extract_dist_tree = maia.factory.recover_dist_tree(extract_part_tree, comm)
    extract_dist_zone = PT.get_all_Zone_t(extract_dist_tree)[0]
    assert PT.Zone.n_vtx( extract_dist_zone)==100
    assert PT.Zone.n_cell(extract_dist_zone)==81
    coord_x,_,_ = PT.Zone.coordinates(extract_dist_zone)
    field_x = PT.get_node_from_path(extract_dist_zone, 'FlowSol#Vtx/cx')[1]
    assert np.array_equal(coord_x, field_x)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cgns_name" , ['Structured','Poly'])
def test_extract_part_obj(cgns_name, comm):
  if cgns_name=='Structured':
    dist_tree = maia.factory.dcube_generator.dcube_struct_generate(3, 1., [0.,0.,0.], comm)
  else:
    dist_tree = maia.factory.generate_dist_block(3, "Poly", comm)
  zone_to_parts = maia.factory.partitioning.compute_regular_weights(\
      dist_tree, comm, 2*(comm.Get_rank() == 1)) #2 parts on proc 1, 0 on proc 0
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts)

  if comm.Get_rank() == 0:
    pl = []
  else:
    pl = [np.array([[1,2],[1,1],[1,3]], np.int32), np.array([[1,2],[1,1],[1,3]], np.int32)] if cgns_name=='Structured' else\
         [np.array([[1,2]], np.int32), np.array([[1,2]], np.int32)]
  loc = 'Vertex' if cgns_name=='Structured' else 'FaceCenter'
  extractor = EP.Extractor(part_tree, [pl], loc, comm)
  extracted_tree = extractor.get_extract_part_tree()

  if cgns_name=='Structured':
    if comm.rank==0:
      assert len(PT.get_all_Zone_t(extracted_tree)) == 0
    else:
      assert len(PT.get_all_Zone_t(extracted_tree)) == 2
    assert (PT.get_all_CGNSBase_t(extracted_tree)[0][1] == [2,3]).all()
  else:
    assert len(PT.get_all_Zone_t(extracted_tree)) == 1
    assert (PT.get_all_CGNSBase_t(extracted_tree)[0][1] == [2,3]).all()
  
@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cgns_name" , ['Structured','Poly'])
@pytest.mark.parametrize('partial', [False, True])
def test_exch_field(cgns_name, partial, comm):
  part_tree = sample_part_tree(cgns_name, comm)

  # Add field
  for zone in PT.get_all_Zone_t(part_tree):
    n_vtx = PT.Zone.n_vtx(zone)
    gnum = PT.maia.getGlobalNumbering(zone, 'Vertex')[1]
    if partial: #Take one over two
      if cgns_name=='Structured':
        pr = np.array([[1,3],[1,1],[1,5]], np.int32)
        i_ar = np.arange(min(pr[0]), max(pr[0])+1)
        j_ar = np.arange(min(pr[1]), max(pr[1])+1).reshape(-1,1)
        k_ar = np.arange(min(pr[2]), max(pr[2])+1).reshape(-1,1,1)
        pl = s_numbering.ijk_to_index_from_loc(i_ar, j_ar, k_ar, 'Vertex', PT.Zone.VertexSize(zone)).flatten()
        PT.new_ZoneSubRegion('FlowSol', loc="Vertex", point_range=pr, fields={'gnum': gnum[pl-1]}, parent=zone)
      else:
        pl = np.arange(1,n_vtx+1)[::2].astype(np.int32).reshape((1,-1))
        PT.new_ZoneSubRegion('FlowSol', loc="Vertex", point_list=pl, fields={'gnum': gnum[::2]}, parent=zone)
    else:
      fld = gnum.reshape(PT.Zone.VertexSize(zone), order='F') if cgns_name=='Structured' else gnum
      PT.new_FlowSolution('FlowSol', loc="Vertex", fields={'gnum': fld}, parent=zone)

  extractor = EP.Extractor(part_tree, [[np.array([[1,3],[1,5],[1,1]], np.int32)]], "Vertex", comm)
  extractor.exchange_fields(['FlowSol'])
  extr_tree = extractor.get_extract_part_tree()

  extr_zone = PT.get_all_Zone_t(extr_tree)[0]
  extr_sol = PT.get_node_from_name(extr_tree, 'FlowSol')
  assert PT.Subset.GridLocation(extr_sol) == 'Vertex'
  data = PT.get_node_from_name(extr_sol, 'gnum')[1]
  if partial:
    assert PT.get_label(extr_sol) == 'ZoneSubRegion_t'
    if cgns_name=='Structured':
      pr = PT.get_node_from_name(extr_sol, 'PointRange')[1]
      i_ar = np.arange(min(pr[0]), max(pr[0])+1)
      j_ar = np.arange(min(pr[1]), max(pr[1])+1).reshape(-1,1)
      k_ar = np.arange(min(pr[2]), max(pr[2])+1).reshape(-1,1,1)
      pl = s_numbering.ijk_to_index_from_loc(i_ar, j_ar, k_ar, 'Vertex', PT.Zone.VertexSize(extr_zone)).flatten()
      lnum = extractor.exch_tool_box['Base/zone'][PT.get_name(extr_zone)]['parent_lnum_vtx']
      lnum = lnum[pl-1]
      zone = PT.get_all_Zone_t(part_tree)[0]
      gnum = PT.maia.getGlobalNumbering(zone, 'Vertex')[1]
      gnum = gnum[lnum-1]
    else:
      pl = PT.get_node_from_name(extr_sol, 'PointList')[1][0]
      gnum = extractor.exch_tool_box['Base/zone']['parent_elt']['Vertex'][pl-1]
    assert np.array_equal(gnum, data)
  else:
    assert PT.get_label(extr_sol) == 'FlowSolution_t'
    if cgns_name=='Structured':
      lnum = extractor.exch_tool_box['Base/zone'][PT.get_name(extr_zone)]['parent_lnum_vtx']
      zone = PT.get_all_Zone_t(part_tree)[0]
      gnum = PT.maia.getGlobalNumbering(zone, 'Vertex')[1]
      gnum = gnum[lnum-1].reshape(PT.Zone.VertexSize(extr_zone), order='F')
    else:
      gnum = extractor.exch_tool_box['Base/zone']['parent_elt']['Vertex']
    assert np.array_equal(data,gnum)

@pytest_parallel.mark.parallel(2)
def test_exch_field_from_bc_zsr(comm):
  part_tree = sample_part_tree('Poly', comm)

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
@pytest.mark.parametrize("cgns_name" , ['Structured','Poly'])
def test_zsr_api(cgns_name, comm):
  dist_tree = maia.factory.generate_dist_block(4, cgns_name, comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  if comm.rank != 1:
    for zone in PT.get_all_Zone_t(part_tree):
      if cgns_name=="Structured":
        pr = np.array([[1,1],[1,1],[1,2]], dtype=np.int32)
        zsr_n = PT.new_ZoneSubRegion('ToExtract', loc='CellCenter', point_range=pr, parent=zone)
      else:
        n_face = PT.Zone.n_face(zone)
        pl = np.array([1,2], dtype=np.int32).reshape((1,-1)) + n_face
        zsr_n = PT.new_ZoneSubRegion('ToExtract', loc='CellCenter', point_list=pl, parent=zone)
  extracted_tree = EP.extract_part_from_zsr(part_tree, 'ToExtract', comm)

  zone_n = PT.get_all_Zone_t(extracted_tree)
  n_cell_extr = PT.Zone.n_cell(zone_n[0]) if len(zone_n)==1 else 0
  assert comm.allreduce(n_cell_extr, op=MPI.SUM) == 4


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cgns_name" , ['Structured','Poly'])
def test_bc_name_api(cgns_name, comm):
  dist_tree = maia.factory.generate_dist_block(4, cgns_name, comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  extracted_tree = EP.extract_part_from_bc_name(part_tree, 'Xmin', comm)

  zone_n = PT.get_all_Zone_t(extracted_tree)
  n_cell_extr = PT.Zone.n_cell(zone_n[0]) if len(zone_n)==1 else 0
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
