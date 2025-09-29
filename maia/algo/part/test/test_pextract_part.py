from packaging.version import Version
from mpi4py import MPI
import pytest
import pytest_parallel
import numpy as np
import math

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT
from   maia.utils import s_numbering, par_utils
from   maia.utils import logging as mlog

from maia.algo.part import extract_part as EP

from Pypdm.Pypdm import __version__ as _PDM_VERSION
PDM_VERSION = Version(_PDM_VERSION)

class LogCapture():
  def __init__(self):
    self.logs = ''
  def log(self, msg):
      self.logs += msg
  def reset(self):
    self.logs = ''

def sample_part_tree(cgns_name, comm, bc_loc='Vertex'):
  if cgns_name=='Structured':
    dist_tree = maia.factory.dcube_generator.dcube_struct_generate(5, 1., [0.,0.,0.], comm, bc_location=bc_loc)
  else:
    dist_tree = maia.factory.generate_dist_block(3, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, graph_part_tool='gnum')
  return part_tree

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("location", ['Vertex','FaceCenter','CellCenter'])
def test_extract_part_simple_u(location, comm):
  part_tree = sample_part_tree('Poly', comm)

  pl = np.array([[1,2]], np.int32)
  if location=='CellCenter':
    pl += PT.Zone.n_face(PT.get_all_Zone_t(part_tree)[0])

  ex_zone, ptp_data = EP.extract_part_one_domain_u(PT.get_all_Zone_t(part_tree), \
      [pl], (3,EP.LOC_TO_DIM[3][location]), comm)
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

  location = 'Vertex' if bc_loc=='Vertex' else 'JFaceCenter'
  tgt_dim = 0 if bc_loc == 'Vertex' else 2
  pr = PT.get_value(PT.get_child_from_predicates(part_tree, f'CGNSBase_t/Zone_t/ZoneBC_t/Ymax/PointRange'))
  ex_zones, etb_zones = EP.extract_part_one_domain_s(PT.get_all_Zone_t(part_tree), \
      [pr], (3,tgt_dim), location, comm)

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
    if bc_name.endswith('min'):
      extract_part_tree = EP.extract_part_from_bc_name(part_tree, bc_name, comm, containers_name=['FlowSol#Vtx'])
    else:
      extractor = EP.create_extractor_from_bc_name(part_tree, bc_name, comm)
      extract_part_tree = extractor.get_extract_part_tree()
      extractor.exchange_fields(['FlowSol#Vtx'])
    extract_dist_tree = maia.factory.recover_dist_tree(extract_part_tree, comm, data_transfer='FIELDS')
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
    gnum = MT.globalnumbering_value(zone, 'Vertex')
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
  assert PT.Container.GridLocation(extr_sol) == 'Vertex'
  data = PT.get_node_from_name(extr_sol, 'gnum')[1]
  if partial:
    assert PT.get_label(extr_sol) == 'ZoneSubRegion_t'
    if cgns_name=='Structured':
      pr = PT.get_node_from_name(extr_sol, 'PointRange')[1]
      assert pr.shape == (2,2)
      i_ar = np.arange(pr[0,0], pr[0,1]+1)
      j_ar = np.array([[1]]) # This is the extracting direction
      k_ar = np.arange(pr[1,0], pr[1,1]+1).reshape(-1,1,1)
      pl = s_numbering.ijk_to_index_from_loc(i_ar, j_ar, k_ar, 'Vertex', PT.Zone.VertexSize(extr_zone)).flatten()
      lnum = extractor.exch_tool_box['Base/zone'][PT.get_name(extr_zone)]['parent_lnum_vtx']
      lnum = lnum[pl-1]
      zone = PT.get_all_Zone_t(part_tree)[0]
      gnum = MT.globalnumbering_value(zone, 'Vertex')
      gnum = gnum[lnum-1]
    else:
      pl = PT.get_node_from_name(extr_sol, 'PointList')[1][0]
      gnum = extractor.exch_tool_box['Base/zone']['parent_elt']['Vertex'][0][pl-1]
    assert np.array_equal(gnum, data)
  else:
    assert PT.get_label(extr_sol) == 'FlowSolution_t'
    if cgns_name=='Structured':
      lnum = extractor.exch_tool_box['Base/zone'][PT.get_name(extr_zone)]['parent_lnum_vtx']
      zone = PT.get_all_Zone_t(part_tree)[0]
      gnum = MT.globalnumbering_value(zone, 'Vertex')
      gnum = gnum[lnum-1].reshape(PT.Zone.VertexSize(extr_zone), order='F')
    else:
      gnum = extractor.exch_tool_box['Base/zone']['parent_elt']['Vertex'][0]
    assert np.array_equal(data,gnum)

@pytest.mark.parametrize("bc_name" , ['Xmin', 'Zmax'])
@pytest_parallel.mark.parallel(2)
def test_exch_field_from_bc_zsr(bc_name, comm):
  part_tree = sample_part_tree('Poly', comm)

  # Add field
  for zone in PT.get_all_Zone_t(part_tree):
    gnum = MT.globalnumbering_value(PT.get_node_from_name(zone, 'NGonElements'), 'Element')
    bc_n = PT.get_child_from_predicates(zone, f'ZoneBC_t/{bc_name}')
    if bc_n is not None:
      bc_pl   = PT.get_value(PT.get_node_from_name(bc_n, "PointList"))
      bc_gnum = gnum[bc_pl[0]-1]
      PT.new_ZoneSubRegion(f'ZSR_{bc_name}',
                           bc_name=bc_name,
                           fields={'gnum': bc_gnum,
                                   'gnum_d': bc_gnum.astype(np.float64),}, parent=zone)
    else:
      bc_pl   = np.empty((1,0), dtype=np.int32, order='F')

  extractor = EP.Extractor(part_tree, [[bc_pl]], "FaceCenter", comm)
  extractor.exchange_fields([f'ZSR_{bc_name}'])
  extr_tree = extractor.get_extract_part_tree()
  extr_zone = PT.get_all_Zone_t(extr_tree)[0]
  ngon = PT.Zone.NGonNode(extr_zone)

  extr_sol = PT.get_node_from_name(extr_tree, f'ZSR_{bc_name}')
  assert PT.get_label(extr_sol) == 'ZoneSubRegion_t'
  assert PT.Container.GridLocation(extr_sol) == 'CellCenter'
  pl    = PT.get_node_from_name(extr_sol, 'PointList')[1][0]
  data  = PT.get_node_from_name(extr_sol, 'gnum')[1]
  assert np.array_equal(extractor.exch_tool_box['Base/zone']['parent_elt']['FaceCenter'][0][pl-PT.Element.Range(ngon)[0]], data)

def portable_partitioning(dist_tree, comm):
  """ Create a custom partioning (chosing cells for each part) to ensure portability
  The switch will be remove when PDM 2.6 is no longer supported
  """
  from Pypdm.Pypdm import MultiPart
  if hasattr(MultiPart, 'dpart_id_set'):
    zone_to_parts = [{'Base/zone' : [.25,.25]},
                     {'Base/zone' : []},
                     {'Base/zone' : [.5]}][comm.rank]
    target_part = [[np.array([0,0,1,2,2,1,2,2,2], np.int32)],
                   [np.array([0,0,1,2,2,1,2,2,2], np.int32)],
                   [np.array([0,0,1,2,2,1,2,2,2], np.int32)]][comm.rank]

    return maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts,
                                            target_part=target_part, data_transfer='ALL')
  from maia.transfer import protocols as MEP
  from maia.algo.dist.localize import minimal_partitioning
  from maia.factory.partitioning import post_split
  # Reorder cells, to give to each rank its wanted cells
  maia.algo.pe_to_nface(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]
  wanted_cell = [np.array([0,1,9,10,18,19]),
                 np.array([2,5,11,14,20,23]),
                 np.array([3,4,6,7,8,12,13,15,16,17,21,22,24,25,26])][comm.rank]
  nface = PT.Zone.NFaceNode(zone)
  cell_face = MT.Element.connectivity(nface)
  selected_cell_face = MEP.block_to_part(cell_face, MT.distribution_value(nface, 'Element'), wanted_cell, comm)
  PT.set_value(PT.find_child_from_name(nface, 'ElementStartOffset'), selected_cell_face.displs)
  PT.set_value(PT.find_child_from_name(nface, 'ElementConnectivity'), selected_cell_face.values)
  MT.new_Distribution({'Element' : par_utils.dn_to_distribution(len(wanted_cell), comm),
                       'ElementConnectivity' : par_utils.dn_to_distribution(selected_cell_face.dsize, comm)}, nface)
  MT.new_Distribution({'Cell' : par_utils.dn_to_distribution(len(wanted_cell), comm)}, zone)
  # Call minimal_partitioning and reconstruct partitions
  data = minimal_partitioning(zone, comm)
  ptree = PT.new_CGNSTree()
  pbase = PT.new_CGNSBase(parent=ptree)
  pzone = PT.new_Zone(f'zone.P{comm.rank}.N0', type='Unstructured', size=[[data[7].size, data[5].size, 0]], parent=pbase)
  PT.new_GridCoordinates(fields={f'Coordinate{d}': data[4][i::3] for i,d in enumerate('XYZ')}, parent=pzone)
  elt = PT.new_NGonElements(erange=[1,data[6].size], eso=data[2], ec=data[3], parent=pzone)
  MT.new_GlobalNumbering({'Element' : data[6]}, elt)
  elt = PT.new_NFaceElements(erange=[data[6].size+1, data[6].size+data[5].size], eso=data[0], ec=data[1], parent=pzone)
  MT.new_GlobalNumbering({'Element' : data[5]}, elt)
  MT.new_GlobalNumbering({'Vertex' : data[7], 'Cell' : data[5]}, pzone)
  # Complete partition with post_split + transfer
  post_split.post_partitioning(dist_tree, ptree, comm)
  maia.transfer.dist_tree_to_part_tree_all(dist_tree, ptree, comm)
  # P0 was supposed to have 2 partitions
  if comm.rank == 1:
    comm.send(PT.find_node_from_name(ptree, 'zone.P1.N0'), dest=0)
    PT.rm_nodes_from_label(ptree, 'Zone_t')
  if comm.rank == 0:
    zone = comm.recv(source=1)
    PT.set_name(zone, 'zone.P0.N1')
    PT.add_child(pbase, zone)
  return ptree

@pytest_parallel.mark.parallel(3)
def test_extr_U_local(comm):

  dist_tree = maia.factory.generate_dist_block(4, "Poly", comm)
  # Prepare dist tree (add some fields)
  zone = PT.find_node_from_label(dist_tree, 'Zone_t')
  dtype = PT.get_np_value(zone).dtype
  n_vtx = PT.Zone.n_vtx(zone)
  vtx_distri = MT.distribution_value(zone, 'Vertex')

  co_node = PT.find_child_from_name(zone, 'GridCoordinates')

  # VtxSubRegion --> A Vertex located subregion, only one point over two have a value
  zsr = PT.deep_copy(co_node)
  mask_full = np.ones(n_vtx, bool)
  mask_full[1::2] = False
  mask_loc = mask_full[vtx_distri[0]:vtx_distri[1]]
  PT.update_node(zsr, name='VtxSubRegion', label='ZoneSubRegion_t')
  for arr in PT.get_children_from_label(zsr, 'DataArray_t'):
    PT.set_value(arr, PT.get_np_value(arr)[mask_loc])
  pl = np.arange(vtx_distri[0]+1, vtx_distri[1]+1, dtype=dtype)[mask_loc].reshape((1,-1), order='F')
  PT.new_IndexArray('PointList', pl, zsr)
  MT.new_Distribution({'Index' : par_utils.dn_to_distribution(pl.size, comm)}, zsr)
  PT.add_child(zone, zsr)

  # VtxSol --> A vertex located full FlowSolution
  fs = PT.deep_copy(co_node)
  PT.update_node(fs, name='VtxSol', label='FlowSolution_t')
  PT.add_child(zone, fs)

  # Geometry_2d --> A FaceCenter FlowSolution
  maia.algo.compute_elements_center(dist_tree, 2, comm)


  """
  # Choose splitting to have : 2 parts on rank 0,  0 parts on rank 1,  1 part on rank 2
  # --> not portable (with openmpi / alpine), replace with custom func
  zone_to_parts = [{'Base/zone' : [.25,.25]},
                   {'Base/zone' : []},
                   {'Base/zone' : [.5]}][comm.rank]

  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts, data_transfer='FIELDS')
  """
  part_tree = portable_partitioning(dist_tree, comm)

  # NB : BC Xmin is covered by the P0.N0 and P2.N0, BC Xmax by P0.N1 and P2.N0
  for part_zone in PT.get_all_Zone_t(part_tree):
    for bc_name in ['Xmin', 'Xmax']:
      bc = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t')
      if bc is not None:
        PT.new_FamilyName('EXTRACT', parent=bc)

  extracted_tree = EP.extract_part_from_family(part_tree, 'EXTRACT', comm,
                                               containers_name=['Geometry_2d', 'VtxSol', 'VtxSubRegion'],
                                               equilibrate=False)
  extracted_zones = PT.get_all_Zone_t(extracted_tree)
  n_cell_extr = [PT.Zone.n_cell(z) for z in extracted_zones]

  n_cell_extr_expected = [[3,6], [], [9]][comm.rank]
  assert n_cell_extr == n_cell_extr_expected

  # Check interfaces
  if comm.rank == 2:
    match1 = PT.find_node_from_predicate(extracted_tree, PT.pred.IS_GC & PT.pred.value_is('Zone.P0.N0'))
    match2 = PT.find_node_from_predicate(extracted_tree, PT.pred.IS_GC & PT.pred.value_is('Zone.P0.N1'))
    assert (PT.get_np_value(PT.find_child_from_name(match1, 'PointList')) == [3,11,19]).all()
    assert (PT.get_np_value(PT.find_child_from_name(match1, 'PointListDonor')) == [3,6,9]).all()
    assert (PT.get_np_value(PT.find_child_from_name(match2, 'PointList')) == [6,14,22]).all()
    assert (PT.get_np_value(PT.find_child_from_name(match2, 'PointListDonor')) == [5,10,15]).all()

  for zone in extracted_zones:
    vtxsol = PT.find_child_from_name(zone, 'VtxSol')
    assert PT.get_child_from_name(vtxsol, 'PointList') is None
    assert PT.get_np_value(PT.find_child_from_label(vtxsol, 'DataArray_t')).size == PT.Zone.n_vtx(zone)

    facesol = PT.find_child_from_name(zone, 'Geometry_2d')
    assert PT.Container.GridLocation(facesol) == 'CellCenter'
    assert PT.get_np_value(PT.find_child_from_label(facesol, 'DataArray_t')).size == PT.Zone.n_cell(zone)
    if PT.get_name(zone) == 'Zone.P0.N0':
      assert (PT.get_np_value(PT.find_child_from_name(facesol, 'CenterX')) == 0).all()
    if PT.get_name(zone) == 'Zone.P0.N1':
      assert (PT.get_np_value(PT.find_child_from_name(facesol, 'CenterX')) == 1).all()

    vtxzsr = PT.get_child_from_name(zone, 'VtxSubRegion')
    if PT.get_name(zone) == 'Zone.P0.N0':
      assert (MT.globalnumbering_value(vtxzsr, 'Index') == [5, 6, 2, 1, 9, 10, 13, 14]).all()
    if PT.get_name(zone) == 'Zone.P0.N1':
      assert vtxzsr is None
    if PT.get_name(zone) == 'Zone.P2.N0':
      assert (MT.globalnumbering_value(vtxzsr, 'Index') == [6, 7, 3, 2, 8, 4, 10, 11, 12, 14, 15, 16]).all()


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cgns_name" , ['Structured','Poly'])
def test_zsr_api(cgns_name, comm):
  dist_tree = maia.factory.generate_dist_block(4, cgns_name, comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  if comm.rank != 1:
    for zone in PT.get_all_Zone_t(part_tree):
      if cgns_name=="Structured":
        pr = np.array([[1,1],[1,1],[1,2]], dtype=np.int32)
        PT.new_ZoneSubRegion('ToExtract', loc='CellCenter', point_range=pr, parent=zone)
      else:
        n_face = PT.Zone.n_face(zone)
        pl = np.array([1,2], dtype=np.int32).reshape((1,-1)) + n_face
        PT.new_ZoneSubRegion('ToExtract', loc='CellCenter', point_list=pl, parent=zone)
  if cgns_name == 'Structured':
    extracted_tree = EP.extract_part_from_zsr(part_tree, 'ToExtract', comm)
  else:
    extracted_tree = EP.create_extractor_from_zsr(part_tree, 'ToExtract', comm).get_extract_part_tree()
  zone_n = PT.get_all_Zone_t(extracted_tree)
  n_cell_extr = PT.Zone.n_cell(zone_n[0]) if len(zone_n)==1 else 0
  assert comm.allreduce(n_cell_extr, op=MPI.SUM) == 4


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cgns_name" , ['Structured'])
@pytest.mark.parametrize("bc_loc" , ['Face'])
def test_bc_name_api(cgns_name, bc_loc, comm):
  dist_tree = maia.factory.generate_dist_block(4, cgns_name, comm)

  irank = comm.Get_rank()
  bc = PT.get_node_from_name(dist_tree, 'Xmin')
  if cgns_name == 'Structured' and bc_loc == 'Face':
    # Move BC to FaceCenter
    PT.update_child(bc, 'GridLocation', value='IFaceCenter')
    pr = PT.get_child_from_name(bc, 'PointRange')[1]
    distri = PT.get_node_from_name(bc, 'Index')[1]
    distri[:] = [3*irank, 3*(irank+1), 9]
    pr[1:,1] -= 1
  elif cgns_name == 'Poly' and bc_loc == 'Vtx':
    # Move BC to Vertex
    PT.update_child(bc, 'GridLocation', value='Vertex')
    pl = PT.get_child_from_name(bc, 'PointList')
    distri = PT.get_node_from_name(bc, 'Index')[1]
    distri[:] = [5*irank, 16 if irank==2 else 5*(irank+1), 16]
    _pl = (np.arange(1, 4**3, 4, dtype=pl[1].dtype)[distri[0]:distri[1]]).reshape((1,-1), order='F')
    PT.set_value(pl, _pl)

  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  # Add a (full) field to the BC
  bc = PT.get_node_from_name(part_tree, 'Xmin')
  if bc is not None:
    bcds = PT.new_BCDataSet(parent=bc)
    PT.new_BCData('DirichletData', fields={'range': np.arange(PT.Subset.n_elem(bc))}, parent=bcds)

  extracted_tree = EP.extract_part_from_bc_name(part_tree, 'Xmin', comm)

  zone_n = PT.get_all_Zone_t(extracted_tree)
  n_cell_extr = PT.Zone.n_cell(zone_n[0]) if len(zone_n)==1 else 0
  if cgns_name == 'Poly' and bc_loc == 'Vtx':
    # In this case, we are creating a point cloud
    assert comm.allreduce(n_cell_extr, op=MPI.SUM) == 0
  else:
    assert comm.allreduce(n_cell_extr, op=MPI.SUM) == 9

  if len(zone_n) > 0:
    cnt = PT.find_node_from_label(extracted_tree, 'FlowSolution_t')
    expt_loc  = "Vertex" if bc_loc == 'Vtx' else 'CellCenter'
    assert PT.Container.GridLocation(cnt) == expt_loc
    assert PT.get_child_from_predicate(cnt, PT.pred.name_in(['PointList', 'PointRange'])) is None


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

  # Family API is not available for Structured meshes
  dist_tree = maia.factory.generate_dist_block(4, "S", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  with pytest.raises(RuntimeError):
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


@pytest_parallel.mark.parallel(1)
def test_void_extraction(comm):
  is_empty_tree = ~PT.pred.has_child_of_label('CGNSBase_t')

  dist_tree = maia.factory.generate_dist_block(4, "Poly", comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  maia.algo.compute_elements_measure(part_tree, 3, comm)  # Create field

  mlog.add_printer_to_logger('maia-warnings', log_collector := LogCapture())

  extracted_tree = EP.extract_part_from_zsr(part_tree, 'EXTRACT', comm, containers_name=['Geometry_3d'])
  assert is_empty_tree(extracted_tree)
  assert 'ZoneSubRegion "EXTRACT" does not exist in input tree' in log_collector.logs

  log_collector.reset()
  extracted_tree = EP.extract_part_from_bc_name(part_tree, 'EXTRACT', comm, containers_name=['Geometry_3d'])
  assert is_empty_tree(extracted_tree)
  assert 'BC "EXTRACT" does not exist in input tree' in log_collector.logs

  log_collector.reset()
  extractor = EP.create_extractor_from_family(part_tree, 'EXTRACT', comm)
  extractor.exchange_fields(['Geometry_3d'])
  assert is_empty_tree(extractor.get_extract_part_tree())
  assert 'Family "EXTRACT" does not exist in input tree' in log_collector.logs

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("equilibrate", [True, False])
def test_extract_fam_dataset(comm, equilibrate):
  tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  dn_elt = MT.Subset.dn_elem(PT.find_node_from_name(zone, 'Xmin'))
  PT.new_ZoneSubRegion('ZSR', bc_name='Xmin', family='FAM',
                       fields={'One' : np.ones(dn_elt), 'Two' : 2*np.ones(dn_elt)},
                       parent=zone)

  dn_elt = MT.Subset.dn_elem(PT.find_node_from_name(zone, 'Ymin'))
  bc = PT.find_node_from_name(zone, 'Ymin')
  ds = PT.new_BCDataSet(parent=bc)
  PT.new_BCData('NeumannData', fields={'One' : np.ones(dn_elt), 'Three' : 3*np.ones(dn_elt)}, parent=ds)
  PT.new_FamilyName('FAM', parent=bc)
  
  ptree = maia.factory.partition_dist_tree(tree, comm, data_transfer='ALL')
  pext = maia.algo.part.extract_part_from_family(ptree, 'FAM', comm, equilibrate=equilibrate)

  # In this config, with transfert_dataset=True, we should
  # have a full container containing Ones (common to all extracted subsets)
  # and 2 partials containers for Two and Three (not common)
  ext_zone = PT.find_node_from_label(pext, 'Zone_t')
  full = PT.find_child_from_name(ext_zone, 'FAM')
  assert PT.get_label(full) == 'FlowSolution_t' and not PT.Container._is_partial(full)
  assert [PT.get_name(n) for n in PT.get_children_from_label(full, 'DataArray_t')] == ['One']
  for cnt_name, array_name in zip(['ZSR', 'Ymin'], ['Two', 'Three']):
    partial = PT.get_child_from_name(ext_zone, cnt_name)
    assert comm.allreduce(partial is not None, MPI.LOR) == True
    if partial is not None:
      assert PT.get_label(partial) == 'ZoneSubRegion_t'
      assert [PT.get_name(n) for n in PT.get_children_from_label(partial, 'DataArray_t')] == [array_name]

  # Cnts are empty -> cleaning is expected
  PT.rm_nodes_from_predicate(ptree, PT.pred.name_in(['One', 'Two', 'Three']))
  pext = maia.algo.part.extract_part_from_family(ptree, 'FAM', comm)
  ext_zone = PT.find_node_from_label(pext, 'Zone_t')
  assert PT.get_child_from_name(ext_zone, 'FAM') is None
  assert PT.get_child_from_name(ext_zone, 'ZSR') is None
  assert PT.get_child_from_name(ext_zone, 'Ymin') is None

@pytest.mark.parametrize("equilibrate", [True, False])
@pytest_parallel.mark.parallel(2)
def test_extract_from_zsr_U_2d(equilibrate, comm):
  

  # Prepare refs
  ref_face = PT.yaml.to_cgns_tree("""
Base CGNSBase_t I4 [2, 3]:
  Zone Zone_t I4 [[24, 15, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t:
        R8 : [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5,
              -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
      CoordinateY DataArray_t:
        R8 : [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5,
              0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
      CoordinateZ DataArray_t:
        R8 : [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5,
              -2.5, -2.5, -2.5, -2.5, -2.5, -2.5]
    EdgeElements Elements_t I4 [3, 0]:
      ElementRange IndexRange_t I4 [1, 38]:
      ElementConnectivity DataArray_t:
        I4 : [2, 1, 3, 2, 4, 3, 7, 1, 5, 4, 2, 8, 6, 5, 3, 9, 4, 10, 8, 7, 5, 11, 9, 8, 6, 12, 10, 9, 13, 7, 11, 10, 8, 14,
              12, 11, 9, 15, 10, 16, 14, 13, 11, 17, 15, 14, 12, 18, 16, 15, 19, 13, 17, 16, 14, 20, 18, 17, 15, 21, 16,
              22, 20, 19, 17, 23, 21, 20, 18, 24, 22, 21, 23, 22, 24, 23]
    NGonElements Elements_t I4 [22, 0]:
      ElementRange IndexRange_t I4 [39, 53]:
      ElementStartOffset DataArray_t I4 [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]:
      ElementConnectivity DataArray_t:
        I4 : [7, 1, 2, 8, 3, 9, 8, 2, 4, 10, 9, 3, 5, 11, 10, 4, 6, 12, 11, 5, 13, 7, 8, 14, 9, 15, 14, 8, 10, 16, 15, 9,
              11, 17, 16, 10, 12, 18, 17, 11, 19, 13, 14, 20, 15, 21, 20, 14, 16, 22, 21, 15, 17, 23, 22, 16, 18, 24, 23,
              17]
    ZoneBC ZoneBC_t:
      Xmax BC_t 'Null':
        GridLocation GridLocation_t 'EdgeCenter':
        PointList IndexArray_t I4 [[13, 24, 35]]:
      Ymax BC_t 'Null':
        GridLocation GridLocation_t 'EdgeCenter':
        PointList IndexArray_t I4 [[32, 34, 36, 37, 38]]:
      Xmin BC_t 'Null':
        GridLocation GridLocation_t 'EdgeCenter':
        PointList IndexArray_t I4 [[4, 15, 26]]:
    ZSR_Faces FlowSolution_t:
      GridLocation GridLocation_t 'CellCenter':
      cx DataArray_t:
        R8 : [-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    FlowSolution_NC FlowSolution_t:
      GridLocation GridLocation_t 'Vertex':
      cx DataArray_t:
        R8 : [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5,
              -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    FlowSolution_CC FlowSolution_t:
      GridLocation GridLocation_t 'CellCenter':
      cx DataArray_t:
        R8 : [-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0]
  """)
  if not equilibrate: # Somehow in local mode, one PL is different
    node = PT.find_node_from_name(ref_face, 'Ymax')
    pl = PT.find_child_from_name(node, 'PointList')
    PT.set_value(pl, np.array([[32, 34, 37, 36, 38]]))


  ref_edge = PT.yaml.to_cgns_tree("""
Base CGNSBase_t I4 [1, 3]:
  Zone Zone_t I4 [[24, 33, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t:
        R8 : [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5,
              -2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
      CoordinateY DataArray_t:
        R8 : [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5,
               2.5, 2.5, 2.5, 2.5, 2.5]
      CoordinateZ DataArray_t:
        R8 : [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5,
              -2.5, -2.5, -2.5, -2.5, -2.5, -2.5]
    EdgeElements Elements_t I4 [3, 0]:
      ElementRange IndexRange_t I4 [1, 33]:
      ElementConnectivity DataArray_t:
        I4 : [7, 1, 2, 8, 3, 9, 4, 10, 8, 7, 5, 11, 9, 8, 6, 12, 10, 9, 13, 7, 11, 10, 8, 14, 12, 11, 9, 15, 10, 16, 14,
              13, 11, 17, 15, 14, 12, 18, 16, 15, 19, 13, 17, 16, 14, 20, 18, 17, 15, 21, 16, 22, 20, 19, 17, 23, 21, 20,
              18, 24, 22, 21, 23, 22, 24, 23]
    ZoneBC ZoneBC_t:
      Xmax BC_t 'Null':
        GridLocation GridLocation_t 'CellCenter':
        PointList IndexArray_t I4 [[8, 19, 30]]:
      Ymax BC_t 'Null':
        GridLocation GridLocation_t 'CellCenter':
        PointList IndexArray_t I4 [[27, 29, 31, 32, 33]]:
      Xmin BC_t 'Null':
        GridLocation GridLocation_t 'CellCenter':
        PointList IndexArray_t I4 [[1, 10, 21]]:
    ZSR_Edges FlowSolution_t:
      GridLocation GridLocation_t 'CellCenter':
      cx DataArray_t:
        R8 : [-2.5, -1.5, -0.5, 0.5, -2.0, 1.5, -1.0, 2.5, 0.0, -2.5, 1.0,
              -1.5, 2.0, -0.5, 0.5, -2.0, 1.5, -1.0, 2.5, 0.0, -2.5, 1.0,
              -1.5, 2.0, -0.5, 0.5, -2.0, 1.5, -1.0, 2.5, 0.0, 1.0, 2.0]
    FlowSolution_NC FlowSolution_t:
      GridLocation GridLocation_t 'Vertex':
      cx DataArray_t:
        R8 : [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5,
              -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
  """)
  if not equilibrate: # Somehow in local mode, one PL is different
    if Version('2.7') <= PDM_VERSION:
      node = PT.find_node_from_name(ref_edge, 'Ymax')
      pl = PT.find_child_from_name(node, 'PointList')
      PT.set_value(pl, np.array([[27, 32, 29, 33, 31]]))
    else: # Was not supported in 2.6
      PT.rm_nodes_from_label(ref_edge, 'ZoneBC_t')


  # This switch is because Hilbert splitter differs in local / reeq mode.
  # To remove if this is fixed
  if equilibrate:
    cx = "[-0.5, -1.5, -2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2.5, 1.5, 0.5, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]"
    cy = "[2.5, 2.5, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]"
    sol = "[-0.5, -1.5, -2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2.5, 1.5, 0.5, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]"
  else:
    cx = "[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5]"
    cy = "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]"
    sol = "[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5]"
  ref_vtx = PT.yaml.to_cgns_tree(f"""

Base CGNSBase_t I4 [2, 3]:
  Zone Zone_t I4 [[18, 0, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 {cx}:
      CoordinateY DataArray_t R8 {cy}:
      CoordinateZ DataArray_t:
        R8 : [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5]
    ZSR_Vtx FlowSolution_t:
      GridLocation GridLocation_t 'Vertex':
      cx DataArray_t R8 {sol}:
    FlowSolution_NC FlowSolution_t:
      GridLocation GridLocation_t 'Vertex':
      cx DataArray_t R8 {sol}:
  """)


  # > Generate tree
  n_vtx  = 6
  n_part = 2
  dist_tree = maia.factory.generate_dist_block(n_vtx, 'QUAD_4', comm, [-2.5, -2.5, -2.5], 5.)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  zone = PT.find_node_from_label(dist_tree, 'Zone_t')
  ztype = PT.get_np_value(zone).dtype

  # Create CC FlowSolution
  face_center = maia.algo.geometry._compute_elements_center(zone, 2, comm)
  fcx = face_center[0::3]
  fcy = face_center[1::3]
  PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", fields={'cx': fcx}, parent=zone)

  # Create EC FlowSolution (not supported)
  edge_center = maia.algo.geometry._compute_elements_center(zone, 1, comm)
  ecx = edge_center[0::3]
  ecy = edge_center[1::3]
  # PT.new_FlowSolution('FlowSolution_EC', loc="EdgeCenter", fields={'cx': ecx}, parent=zone)

  # Create NC FlowSolution
  gc = PT.find_child_from_name(zone, 'GridCoordinates')
  cx = PT.get_np_value(PT.find_child_from_name(gc, 'CoordinateX'))
  cy = PT.get_np_value(PT.find_child_from_name(gc, 'CoordinateY'))
  PT.new_FlowSolution('FlowSolution_NC', loc="Vertex", fields={'cx': cx}, parent=zone)

  # Create ZSR on Faces with FlowSolution
  pl_faces = (np.where(fcy > 0.)[0]).astype(ztype)
  bck = pl_faces.copy()
  fcx = fcx[pl_faces]
  ngon = PT.Zone.NGonNode(zone)
  distrib_faces = MT.distribution_value(zone, 'Cell')
  pl_faces += PT.Element.Range(ngon)[0] + distrib_faces[0]
  zsr_faces = PT.new_ZoneSubRegion("ZSR_Faces", point_list=pl_faces.reshape((1,-1), order='F'), loc='CellCenter', fields={'cx': fcx}, parent=zone)
  MT.new_Distribution({'Index' : par_utils.dn_to_distribution(pl_faces.size, comm)}, zsr_faces)

  # Create ZSR on Edges with FlowSolution
  pl_edges = (np.where(ecy > 0.)[0]).astype(ztype)
  ecx = ecx[pl_edges]
  bar_2 = MT.Zone.EdgeNode(zone)
  distrib_edges = MT.distribution_value(bar_2, 'Element')
  pl_edges += PT.Element.Range(bar_2)[0] + distrib_edges[0]
  zsr_edges = PT.new_ZoneSubRegion("ZSR_Edges", point_list=pl_edges.reshape((1,-1), order='F'), loc='EdgeCenter', fields={'cx': ecx}, parent=zone)
  MT.new_Distribution({'Index' : par_utils.dn_to_distribution(pl_edges.size, comm)}, zsr_edges)

  # Create ZSR on Vertices with FlowSolution
  pl_vtx = (np.where(cy > 0.)[0]).astype(ztype)
  cx = cx[pl_vtx]
  distrib_vtx = MT.distribution_value(zone, 'Vertex')
  pl_vtx += 1 + distrib_vtx[0]
  zsr_vtx = PT.new_ZoneSubRegion("ZSR_Vtx", point_list=pl_vtx.reshape((1,-1), order='F'), loc='Vertex', fields={'cx': cx}, parent=zone)
  MT.new_Distribution({'Index' : par_utils.dn_to_distribution(pl_vtx.size, comm)}, zsr_vtx)

  # Partionning option
  zone_to_parts = maia.factory.partitioning.compute_regular_weights(dist_tree, comm, n_part)
  part_tree     = maia.factory.partition_dist_tree(dist_tree, comm,
                                                   zone_to_parts=zone_to_parts,
                                                   data_transfer='ALL',
                                                   preserve_orientation=True)

  # Extract part Faces
  part_tree_ep = EP.extract_part_from_zsr(part_tree, "ZSR_Faces", comm,
                                          equilibrate=equilibrate,
                                          transfer_dataset=False,
                                          graph_part_tool="ptscotch",
                                          containers_name=['FlowSolution_NC','FlowSolution_CC','ZSR_Faces'])

  # > Part to dist
  dist_tree_ep = maia.factory.recover_dist_tree(part_tree_ep, comm, 'FIELDS')
  ftree_ep = maia.factory.dist_to_full_tree(dist_tree_ep, comm)
  comp = None
  if comm.rank == 0:
    comp = maia.pytree.is_same_tree(ref_face, ftree_ep, abs_tol=1e-14, type_tol=True)
  assert comm.bcast(comp, root=0)


  # Extract part Edges
  part_tree_ep = EP.extract_part_from_zsr(part_tree, "ZSR_Edges", comm,
                                          equilibrate=equilibrate,
                                          transfer_dataset=False,
                                          graph_part_tool="ptscotch",
                                          containers_name=['FlowSolution_NC','ZSR_Edges'])

  # > Part to dist
  dist_tree_ep = maia.factory.recover_dist_tree(part_tree_ep, comm, 'FIELDS')
  ftree_ep = maia.factory.dist_to_full_tree(dist_tree_ep, comm)
  comp = None
  if comm.rank == 0:
    comp = maia.pytree.is_same_tree(ref_edge, ftree_ep, abs_tol=1e-14, type_tol=True)
  assert comm.bcast(comp, root=0)
    

  # Extract part Vertices
  part_tree_ep = EP.extract_part_from_zsr(part_tree, "ZSR_Vtx", comm,
                                          equilibrate=equilibrate,
                                          transfer_dataset=False,
                                          graph_part_tool='hilbert',
                                          containers_name=['FlowSolution_NC','ZSR_Vtx'])
  # > Part to dist
  dist_tree_ep = maia.factory.recover_dist_tree(part_tree_ep, comm, 'FIELDS')
  ftree_ep = maia.factory.dist_to_full_tree(dist_tree_ep, comm)
  comp = None
  if comm.rank == 0:
    comp = maia.pytree.is_same_tree(ref_vtx, ftree_ep, abs_tol=1e-14, type_tol=True)
  assert comm.bcast(comp, root=0)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("transfer_dataset", [True, False])
@pytest.mark.parametrize("eq", [True, False]) # Equilibrate mode
def test_all_transfer(transfer_dataset, eq, comm):
  tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  # These subsets are going to be util for extraction
  dn_elt = MT.Subset.dn_elem(PT.find_node_from_name(zone, 'Xmin'))
  PT.new_ZoneSubRegion('ZSR', bc_name='Xmin', family='FAM',
                       fields={'One' : np.ones(dn_elt), 'Two' : 2*np.ones(dn_elt)},
                       parent=zone)

  dn_elt = MT.Subset.dn_elem(PT.find_node_from_name(zone, 'Ymin'))
  bc = PT.find_node_from_name(zone, 'Ymin')
  ds = PT.new_BCDataSet(parent=bc)
  PT.new_BCData('NeumannData', fields={'One' : np.ones(dn_elt), 'Three' : 3*np.ones(dn_elt)}, parent=ds)
  PT.new_FamilyName('FAM', parent=bc)

  # Create additional data containers to test 'ALL' mode
  coords = PT.shallow_copy(PT.find_child_from_label(zone, 'GridCoordinates_t'))
  PT.update_node(coords, 'FSVtx', label='FlowSolution_t')
  PT.add_child(zone, coords)
  maia.algo.compute_elements_measure(tree, 2, comm)
  maia.algo.compute_elements_measure(tree, 3, comm)
  dn_elt = MT.Subset.dn_elem(PT.find_node_from_name(zone, 'Xmin'))
  PT.new_ZoneSubRegion('OtherZSR', bc_name='Xmin',
                       fields={'Array' : np.ones(dn_elt)},
                       parent=zone)

  PT.new_ZoneSubRegion('FakeZSR', bc_name='Xmin', parent=zone) # Should be ignored (no arrays)

  ptree = maia.factory.partition_dist_tree(tree, comm, data_transfer='ALL')

  pext = maia.algo.part.extract_part_from_zsr(ptree, 'ZSR', comm, transfer_dataset, 'ALL', equilibrate=eq)
  # Tr DS = True or False + local mode + ALL ===> KO
  ext_zones = PT.get_nodes_from_label(pext, 'Zone_t')
  for name in ['FSVtx', 'Geometry_2d', 'OtherZSR']:
    assert par_utils.exists_anywhere(ext_zones, name, comm) == True
  for name in ['Geometry_3d', 'FakeZSR']:
    assert par_utils.exists_anywhere(ext_zones, name, comm) == False
  assert par_utils.exists_anywhere(ext_zones, 'ZSR', comm) == transfer_dataset
  
  pext = maia.algo.part.extract_part_from_bc_name(ptree, 'Ymin', comm, transfer_dataset, 'ALL', equilibrate=eq)
  ext_zone = PT.find_node_from_label(pext, 'Zone_t')
  for name in ['FSVtx', 'Geometry_2d']:
    assert par_utils.exists_anywhere([ext_zone], name, comm) == True
  for name in ['Geometry_3d', 'OtherZSR', 'FakeZSR']:
    assert par_utils.exists_anywhere([ext_zone], name, comm) == False
  assert par_utils.exists_anywhere([ext_zone], 'Ymin', comm) == transfer_dataset

  pext = maia.algo.part.extract_part_from_family(ptree, 'FAM', comm, transfer_dataset, 'ALL', equilibrate=eq)
  ext_zone = PT.find_node_from_label(pext, 'Zone_t')
  for name in ['FSVtx', 'Geometry_2d', 'ZSR', 'OtherZSR']:
    assert par_utils.exists_anywhere([ext_zone], name, comm) == True
  for name in ['Geometry_3d', 'FakeZSR']:
    assert par_utils.exists_anywhere([ext_zone], name, comm) == False
  assert par_utils.exists_anywhere([ext_zone], 'FAM', comm) == transfer_dataset
  assert par_utils.exists_anywhere([ext_zone], 'Ymin', comm) == transfer_dataset

@pytest_parallel.mark.parallel(2)
def test_extract_S_2d(comm):

  tree = maia.factory.generate_dist_block([5,5], 'S', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  # Make BC EdgeCenter to try extraction from edges
  ymax = PT.find_node_from_name(tree, 'Ymax')
  pr = PT.get_np_value(PT.find_child_from_name(ymax, 'PointRange'))
  pr[0,1] -= 1
  PT.update_child(ymax, 'GridLocation', 'GridLocation_t', 'JEdgeCenter')

  # Create CellCenter ZSR
  PT.new_ZoneSubRegion('CellZSR', loc='CellCenter', point_range=[[2,4], [3,4]], parent=zone)

  ptree = maia.factory.partition_dist_tree(tree, comm)

  # Start extractions

  pext = maia.algo.part.extract_part_from_zsr(ptree, 'CellZSR', comm)
  ext_zone = PT.get_all_Zone_t(pext)[0]

  if comm.rank == 0:
    expt_vtx_shape = (2,3)
    expt_edge_gnum = np.array([1,2,5,6,9,12,15])
  else:
    expt_vtx_shape = (3,3)
    expt_edge_gnum = np.array([2,3,4,6,7,8,10,11,13,14,16,17])

  assert PT.Zone.Type(ext_zone) == 'Structured'
  assert PT.Zone.n_cell(ext_zone) == math.prod(k-1 for k in expt_vtx_shape)
  cx = PT.find_node_from_name(ext_zone, 'CoordinateX')
  cz = PT.find_node_from_name(ext_zone, 'CoordinateZ')
  assert PT.get_np_value(cx).shape == expt_vtx_shape
  assert len(PT.get_nodes_from_label(ext_zone, 'GridConnectivity1to1_t')) == 1
  assert MT.get_GlobalNumbering(ext_zone, 'Face') is None
  assert (MT.globalnumbering_value(ext_zone, 'Edge') == expt_edge_gnum).all()
     
  dext = maia.factory.recover_dist_tree(pext, comm)
  maia.io.dist_tree_to_file(dext, 'ext_face_d.cgns', comm)
    

  # Remove CZ for this test
  base  = PT.get_all_CGNSBase_t(ptree)[0]
  PT.set_value(base, [2, 2])
  PT.rm_nodes_from_name(ptree, 'CoordinateZ')

  pext = maia.algo.part.extract_part_from_bc_name(ptree, 'Ymax', comm)
  ext_zone = PT.get_all_Zone_t(pext)[0]

  gnum_t = 'I4' if MT.distribution_value(zone, 'Vertex').dtype == np.int32 else 'I8'
  if comm.rank == 0:
    expt = PT.yaml.to_node(f"""
    zone.P0.N0 Zone_t [[3, 2, 0]]:
      ZoneType ZoneType_t "Structured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [0.,   0.25, 0.5 ]:
        CoordinateY DataArray_t R8 [1., 1., 1.]:
      ZoneGridConnectivity ZoneGridConnectivity_t:
        JN.P0.N0.LT.P1.N0 GridConnectivity1to1_t "zone.P1.N0":
          PointRange IndexRange_t [[3, 3]]:
          PointRangeDonor IndexRange_t [[1, 1]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t {gnum_t} [1, 2, 3]:
        Cell DataArray_t {gnum_t} [1, 2]:
    """)
  else:
    expt = PT.yaml.to_node(f"""
    zone.P1.N0 Zone_t [[3, 2, 0]]:
      ZoneType ZoneType_t "Structured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [0.5,  0.75, 1.  ]:
        CoordinateY DataArray_t R8 [1., 1., 1.]:
      ZoneGridConnectivity ZoneGridConnectivity_t:
        JN.P1.N0.LT.P0.N0 GridConnectivity1to1_t "zone.P0.N0":
          PointRange IndexRange_t [[1,1]]:
          PointRangeDonor IndexRange_t [[3,3]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t {gnum_t} [3, 4, 5]:
        Cell DataArray_t {gnum_t} [3, 4]:
    """)
  assert PT.is_same_tree(ext_zone, expt)

@pytest.mark.skipif(PDM_VERSION < Version('2.7'), reason="Require PDM >= 2.7")
@pytest_parallel.mark.parallel(2)
def test_vol_groups(comm):
  tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  maia.algo.pe_to_nface(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  # Create BC groups on dist tree
  distri = MT.distribution_value(zone, 'Cell')
  cell_gn = np.arange(distri[0]+1, distri[1]+1)
  is_pair = (cell_gn % 2) == 0
  offset = PT.Element.Range(PT.Zone.NFaceNode(zone))[0] + distri[0]
  pair_pl = (np.where(is_pair) + offset).astype(np.int32)
  impair_pl = (np.where(~is_pair) + offset).astype(np.int32)

  zbc = PT.find_node_from_label(zone, 'ZoneBC_t')
  bc = PT.new_BC("Pair", loc='CellCenter', point_list=pair_pl, parent=zbc)
  MT.new_Distribution({'Index' : par_utils.dn_to_distribution(pair_pl.size, comm)}, bc)
  bc = PT.new_BC("Impair", loc='CellCenter', point_list=impair_pl, parent=zbc)
  MT.new_Distribution({'Index' : par_utils.dn_to_distribution(impair_pl.size, comm)}, bc)

  to_extract = (50 <= cell_gn) & (cell_gn < 150)
  extr_pl = (np.where(to_extract)[0]+offset).astype(zone[1].dtype)
  extr_pl = extr_pl.reshape((1,-1), order='F')
  zsr = PT.new_ZoneSubRegion('ZSR', loc='CellCenter', point_list=extr_pl, parent=zone)
  MT.new_Distribution({'Index' : par_utils.dn_to_distribution(extr_pl.size, comm)}, zsr)

  ptree = maia.factory.partition_dist_tree(tree, comm)

  pext = maia.algo.part.extract_part_from_zsr(ptree, "ZSR", comm)

  pair = PT.find_node_from_name_and_label(pext, 'Pair', 'BC_t')
  impair = PT.find_node_from_name_and_label(pext, 'Impair', 'BC_t')
  assert MT.Subset.n_elem([pair], comm) == 50
  assert MT.Subset.n_elem([impair], comm) == 50
