import pytest
import pytest_parallel
import os
import mpi4py.MPI as MPI
import numpy      as np

import Pypdm.Pypdm  as PDM
import maia.pytree  as PT

import maia
import maia.factory as MF
import maia.io      as Mio

from maia.algo.part import extract_part as EXP
from maia.utils     import s_numbering

# > Reference directory
ref_dir  = os.path.join(os.path.dirname(__file__), 'references')


PART_TOOLS = []
if maia.pdm_has_parmetis:
  PART_TOOLS.append("parmetis")
if maia.pdm_has_ptscotch:
  PART_TOOLS.append("ptscotch")


def plane_eq(x,y,z) :
  peq1 = [ 1., 0., 0., 0.6]
  peq2 = [-1., 0., 0., 0.6]
  behind_plane1 = x*peq1[0] + y*peq1[1] + z*peq1[2] - peq1[3] < 0.
  behind_plane2 = x*peq2[0] + y*peq2[1] + z*peq2[2] - peq2[3] < 0.
  between_planes = np.logical_and(behind_plane1, behind_plane2)
  return between_planes

def initialize_bc(zone, bc_name):
  is_struct = PT.Zone.Type(zone)=='Structured'
  
  face_center = maia.algo.part.geometry.compute_face_center(zone)
  cfx = face_center[0::3]
  cfy = face_center[1::3]
  cfz = face_center[2::3]

  bc_n = PT.get_node_from_name(zone, bc_name)
  if bc_n is not None:
    bc_patch = PT.Subset.getPatch(bc_n)
    if is_struct:
      pr = PT.get_value(bc_patch)
      i_ar = np.arange(min(pr[0]), max(pr[0])+1)
      j_ar = np.arange(min(pr[1]), max(pr[1])+1).reshape(-1,1)
      k_ar = np.arange(min(pr[2]), max(pr[2])+1).reshape(-1,1,1)
      bc_pl = s_numbering.ijk_to_index_from_loc(i_ar, j_ar, k_ar, PT.Subset.GridLocation(bc_n), PT.Zone.VertexSize(zone)).flatten()
    else:
      bc_pl = PT.get_value(bc_patch)[0]

    bc_cfx = cfx[bc_pl-1]
    bc_cfy = cfy[bc_pl-1]
    bc_cfz = cfz[bc_pl-1]
    bc_dataset_n = PT.new_node(name='BCDataSet'  , label='BCDataSet_t', value='UserDefined', parent=bc_n)
    neuma_data_n = PT.new_node(name='DirichletData', label='BCData_t'   , value=None         , parent=bc_dataset_n)
    sphere_field = bc_cfx**2 + bc_cfy**2 + bc_cfz**2 - 1
    cylind_field = bc_cfx**2 + bc_cfy**2             - 1
    sphere       = PT.new_DataArray('sphere_bc'  , sphere_field, parent=neuma_data_n)
    cylinder     = PT.new_DataArray('cylinder_bc', cylind_field, parent=neuma_data_n)

def initialize_zsr_by_eq(zone, variables, function, location):
  is_struct = PT.Zone.Type(zone)=='Structured'
  
  # In/out selection array
  in_extract_part = function(*variables)
  
  if is_struct:
    extract_lnum = np.where(in_extract_part)[0]+1
    if extract_lnum.size!=0:
      ijk = s_numbering.index_to_ijk_from_loc(extract_lnum, location, PT.Zone.VertexSize(zone))
      pr = np.array([[min(ijk[0]),max(ijk[0])],
                     [min(ijk[1]),max(ijk[1])],
                     [min(ijk[2]),max(ijk[2])]])
      PT.new_ZoneSubRegion("ZSR_FlowSolution", point_range=pr, loc=location, parent=zone)
    return extract_lnum
  else:
    # Beware of elt numerotation for PointList
    loc_elt_range  = {'Vertex'    : None,
                      'FaceCenter':'NGonElements/ElementRange',
                      'CellCenter':'NFaceElements/ElementRange'}
    if location=='Vertex': starting_range = 1
    else                 : starting_range = PT.get_node_from_path(zone,loc_elt_range[location])[1][0]
    
    # Get loc zsr
    extract_lnum = np.where(in_extract_part)[0]
    extract_lnum = extract_lnum.astype(np.int32)
    extract_lnum = np.flip(extract_lnum) # to be sure that desorganized PL is not a problem
    extract_lnum = extract_lnum + starting_range

    extract_lnum = extract_lnum.reshape((1,-1), order='F') # Ordering in shape (1,N) because of CGNS standard

    PT.new_ZoneSubRegion("ZSR_FlowSolution", point_list=extract_lnum, loc=location, parent=zone)
    return np.ascontiguousarray(extract_lnum)


def generate_test_tree(n_vtx,n_part,location,cgns_name,comm):
  is_struct = cgns_name=='Structured'

  # > Cube generation and partitioning
  # Cube generation
  if is_struct:
    dist_tree = maia.factory.dcube_generator.dcube_struct_generate(n_vtx, 5., [-2.5, -2.5, -2.5], comm, bc_location='FaceCenter')
  else:
    dist_tree = MF.generate_dist_block(n_vtx, cgns_name, comm, [-2.5, -2.5, -2.5], 5.)

  # Partionning option
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, comm, n_part)
  part_tree     = MF.partition_dist_tree(dist_tree, comm,
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)


  # > Init ZSR fields
  point_list = list()
  for zone in PT.get_all_Zone_t(part_tree):
    # Nodes coordinates
    gc = PT.get_child_from_name(zone, 'GridCoordinates')
    cx = PT.get_child_from_name(gc, 'CoordinateX')[1]
    cy = PT.get_child_from_name(gc, 'CoordinateY')[1]
    cz = PT.get_child_from_name(gc, 'CoordinateZ')[1]
    
    cell_center = maia.algo.part.geometry.compute_cell_center(zone)
    ccx = cell_center[0::3]
    ccy = cell_center[1::3]
    ccz = cell_center[2::3]

    # Fields
    sphere_fld_nc =  cx**2 +  cy**2 +  cz**2 - 1
    cylind_fld_nc =  cx**2 +  cy**2          - 1
    sphere_fld_cc = ccx**2 + ccy**2 + ccz**2 - 1
    cylind_fld_cc = ccx**2 + ccy**2          - 1
    if is_struct:
      zone_vtx_dim  = PT.Zone.VertexSize(zone)
      sphere_fld_nc = sphere_fld_nc.reshape(zone_vtx_dim, order='F')
      cylind_fld_nc = cylind_fld_nc.reshape(zone_vtx_dim, order='F')
      zone_cell_dim = PT.Zone.CellSize(zone)
      sphere_fld_cc = sphere_fld_cc.reshape(zone_cell_dim, order='F')
      cylind_fld_cc = cylind_fld_cc.reshape(zone_cell_dim, order='F')

    # Placement FlowSolution
    FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
    FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    PT.new_DataArray('sphere'  , sphere_fld_nc, parent=FS_NC)
    PT.new_DataArray('cylinder', cylind_fld_nc, parent=FS_NC)
    PT.new_DataArray('sphere'  , sphere_fld_cc, parent=FS_CC)
    PT.new_DataArray('cylinder', cylind_fld_cc, parent=FS_CC)


    if is_struct:
      if location=="Vertex":
        ccx, ccy, ccz = cx.reshape(PT.Zone.n_vtx(zone), order='F'),\
                        cy.reshape(PT.Zone.n_vtx(zone), order='F'),\
                        cz.reshape(PT.Zone.n_vtx(zone), order='F')
      elt_range      = [1]
      point_list_loc = [initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, location)]
    else:
      if   location=="CellCenter":
        path_elt_rge  = 'NFaceElements/ElementRange'
        elt_range     = PT.get_node_from_path(zone,path_elt_rge)[1]
      elif location=="FaceCenter":
        path_elt_rge  = 'NGonElements/ElementRange'
        elt_range     = PT.get_node_from_path(zone,path_elt_rge)[1]
        face_center   = maia. algo.part.geometry.compute_face_center(zone)
        ccx = face_center[0::3]
        ccy = face_center[1::3]
        ccz = face_center[2::3]
      elif location=="Vertex":
        ccx, ccy, ccz = cx, cy, cz
        elt_range     = [1]
      else:
        sys.exit()

      point_list_loc = initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, location)
    if point_list_loc[0].size!=0:
      # Put fld in ZSR
      zsr_node = PT.get_node_from_name(zone,'ZSR_FlowSolution')
      PT.new_DataArray("ZSR_ccx", ccx[point_list_loc[0]-elt_range[0]], parent=zsr_node)
      PT.new_DataArray("ZSR_ccy", ccy[point_list_loc[0]-elt_range[0]], parent=zsr_node)
      PT.new_DataArray("ZSR_ccz", ccz[point_list_loc[0]-elt_range[0]], parent=zsr_node)

      point_list.append(point_list_loc)

  return part_tree, point_list



# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
# @pytest.mark.parametrize("graph_part_tool", ["hilbert",'parmetis'])
@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@pytest_parallel.mark.parallel([1,3])
def test_extract_cell_from_zsr_U(graph_part_tool, comm, write_output):

  # > Generate tree
  n_vtx  = 6
  n_part = 2
  part_tree, _ = generate_test_tree(n_vtx,n_part,'CellCenter','Poly',comm)

  # > Extract part
  part_tree_ep = EXP.extract_part_from_zsr( part_tree, "ZSR_FlowSolution", comm,
                                            # equilibrate=1,
                                            transfer_dataset=False,
                                            graph_part_tool=graph_part_tool,
                                            containers_name=['FlowSolution_NC','FlowSolution_CC','ZSR_FlowSolution']
                                            )

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_cell_from_zsr.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), comm)
  
  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)


# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@pytest_parallel.mark.parallel([1,3])
def test_extractor_cell_from_zsr_U(graph_part_tool, comm, write_output):

  # > Generate tree
  n_vtx  = 6
  n_part = 2
  part_tree, _ = generate_test_tree(n_vtx,n_part,'CellCenter','Poly',comm)

  # > Extract part
  extractor = EXP.create_extractor_from_zsr(part_tree, "ZSR_FlowSolution", comm,
                                            # equilibrate=1,
                                            # graph_part_tool="hilbert"
                                            )
  extractor.exchange_fields(['FlowSolution_NC','FlowSolution_CC'])
  # extractor.exchange_zsr_fields("ZSR_FlowSolution", comm)
  extractor.exchange_fields(["ZSR_FlowSolution"]) # Works also with the ZSR node
  part_tree_ep = extractor.get_extract_part_tree()

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_cell_from_zsr.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), comm)
  
  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)


# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@pytest_parallel.mark.parallel([1,3])
def test_extract_cell_from_point_list_U(graph_part_tool, comm, write_output):

  # > Generate tree
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,'CellCenter','Poly',comm)

  # > Extract part
  extractor = EXP.Extractor(part_tree, [point_list], "CellCenter", comm,
                            # equilibrate=1,
                            # graph_part_tool=graph_part_tool,
                           )
  extractor.exchange_fields(['FlowSolution_NC','FlowSolution_CC'])
  part_tree_ep = extractor.get_extract_part_tree()

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_cell_from_point_list.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), comm)
  
  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)


# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
@pytest.mark.parametrize("graph_part_tool", PART_TOOLS)
@pytest_parallel.mark.parallel([1,3])
def test_extract_face_from_point_list_U(graph_part_tool, comm, write_output):

  # > Generate tree
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,'FaceCenter','Poly',comm)

  # > Extract part
  extractor = EXP.Extractor(part_tree, [point_list], "FaceCenter", comm,
                            # equilibrate=1,
                            graph_part_tool=graph_part_tool
                           )  
  extractor.exchange_fields(['FlowSolution_NC',"ZSR_FlowSolution"])
  part_tree_ep = extractor.get_extract_part_tree()

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_face_from_point_list.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_face_from_point_list.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)


@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@pytest_parallel.mark.parallel([1,3])
def test_extract_vertex_from_zsr_U(graph_part_tool, comm, write_output):

  # > Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,'Vertex','Poly',comm)

  # > Extract part
  part_tree_ep = EXP.extract_part_from_zsr( part_tree, "ZSR_FlowSolution", comm,
                                            transfer_dataset=True,
                                            graph_part_tool=graph_part_tool,
                                            containers_name=['FlowSolution_NC']
                                            )
  # Sortie VTK for visualisation
  # part_zones = PT.get_all_Zone_t(part_tree_ep)
  # for i_zone, part_zone in enumerate(part_zones):
  #   write_part_zone_vtx(i_zone,part_zone,comm)
  # Mio.write_trees(part_tree_ep,'OUT_TEST_VERTEX/part_tree_extract.cgns',comm)

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_vertex_from_zsr.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_vertex_from_zsr.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')                , comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)


@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@pytest_parallel.mark.parallel([1,3])
def test_extract_bc_from_bc_name_U(graph_part_tool, comm, write_output):

  # > Generate tree
  n_vtx  = 6
  n_part = 4
  part_tree, _ = generate_test_tree(n_vtx,n_part,'CellCenter','Poly',comm)

  for zone in PT.get_all_Zone_t(part_tree):
    initialize_bc(zone, 'Xmin')
    
  # > Extract part
  part_tree_ep = EXP.extract_part_from_bc_name( part_tree, "Xmin", comm,
                                                graph_part_tool=graph_part_tool,
                                                containers_name=['FlowSolution_NC'],
                                                )

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)
  PT.get_node_from_label(dist_tree_ep,'ZoneSubRegion_t')[3] = 'FlowSolution_t'

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_bc_from_bc_name.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_bc_from_bc_name.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')                , comm)

  # > Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)


@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@pytest_parallel.mark.parallel([1,3])
def test_extract_bcs_from_family_U(graph_part_tool, comm, write_output):

  # > Generate tree
  n_vtx  = 6
  n_part = 4
  part_tree, _ = generate_test_tree(n_vtx,n_part,'CellCenter','Poly',comm)

  for zone in PT.get_all_Zone_t(part_tree):
    initialize_bc(zone, 'Xmin')
    initialize_bc(zone, 'Ymin')
    initialize_bc(zone, 'Zmin')

    for bc_n in PT.get_nodes_from_label(zone, 'BC_t'):
      PT.new_node('FamilyName', label='FamilyName_t', value='ALL_BCS', parent=bc_n)

  part_base = PT.get_child_from_label(part_tree, 'CGNSBase_t')
  PT.new_Family('ALL_BCS', parent=part_base)

  # > Extract part
  part_tree_ep = EXP.extract_part_from_family(part_tree, "ALL_BCS", comm,
                                              transfer_dataset=True,
                                              graph_part_tool=graph_part_tool,
                                              containers_name=['FlowSolution_NC'],
                                              )

  # # > For paraview visu
  # part_zone_ep = PT.get_node_from_label(part_tree_ep,'Zone_t')
  # fld1 = np.zeros(PT.Zone.n_cell(part_zone_ep), dtype=np.float64)
  # fld2 = np.zeros(PT.Zone.n_cell(part_zone_ep), dtype=np.float64)
  # for zsr_name in ['Xmin','Ymin','Zmin']:
  #   zsr_n = PT.get_child_from_name(part_zone_ep, zsr_name)
  #   if zsr_n is not None:
  #     pl = PT.get_value(PT.get_child_from_name(zsr_n, 'PointList'))[0]-1
  #     fld1[pl] = PT.get_value(PT.get_child_from_name(zsr_n, 'sphere_bc'))
  #     fld2[pl] = PT.get_value(PT.get_child_from_name(zsr_n, 'cylinder_bc'))
  # fsfc = PT.new_FlowSolution('FlowSolBCs', loc='CellCenter',
  #                             fields={'spere_bc'   : fld1,
  #                                     'cylinder_bc': fld2},
  #                             parent=part_zone_ep)

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_bcs_from_family.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_bcs_from_family.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')               , comm)

  # > Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)




@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@pytest_parallel.mark.parallel([1,3])
def test_extract_zsr_from_family_U(graph_part_tool, comm, write_output):

  def plane_eq(x,y,z) :
    peq1 = [0.,  1., 0., 0.6]
    peq2 = [0., -1., 0., 0.6]
    behind_plane1 = x*peq1[0] + y*peq1[1] + z*peq1[2] - peq1[3] < 0.
    behind_plane2 = x*peq2[0] + y*peq2[1] + z*peq2[2] - peq2[3] < 0.
    between_planes = np.logical_and(behind_plane1, behind_plane2)
    return between_planes

  # > Generate tree
  n_vtx  = 6
  n_part = 4
  part_tree, _ = generate_test_tree(n_vtx,n_part,'CellCenter','Poly',comm)

  part_base = PT.get_child_from_label(part_tree, 'CGNSBase_t')
  PT.new_Family('ZSRs', parent=part_base)

  for zone in PT.get_all_Zone_t(part_tree):
    # Rename zsr in tree
    zsr_n = PT.get_node_from_name(zone, "ZSR_FlowSolution")
    PT.set_name(zsr_n , "ZSR_x")
    PT.new_node('FamilyName', label='FamilyName_t', value='ZSRs', parent=zsr_n)

    # Create second zsr in tree
    cell_center = maia.algo.part.geometry.compute_cell_center(zone)
    ccx = cell_center[0::3]
    ccy = cell_center[1::3]
    ccz = cell_center[2::3]
    initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, "CellCenter")
    zsr_n = PT.get_node_from_name(zone, "ZSR_FlowSolution")
    PT.set_name(zsr_n , "ZSR_y")
    PT.new_node('FamilyName', label='FamilyName_t', value='ZSRs', parent=zsr_n)

  # > Extract part
  part_tree_ep = EXP.extract_part_from_family(part_tree, "ZSRs", comm,
                                              transfer_dataset=False,
                                              graph_part_tool=graph_part_tool,
                                              containers_name=['FlowSolution_NC','ZSR_x'],
                                              )

  # # > For paraview visu
  # part_zone_ep = PT.get_node_from_label(part_tree_ep,'Zone_t')
  # fld1 = np.zeros(PT.Zone.n_cell(part_zone_ep), dtype=np.float64)
  # fld2 = np.zeros(PT.Zone.n_cell(part_zone_ep), dtype=np.float64)
  # fld3 = np.zeros(PT.Zone.n_cell(part_zone_ep), dtype=np.float64)
  # for zsr_name in ['ZSR_x']:
  #   zsr_n = PT.get_child_from_name(part_zone_ep, zsr_name)
  #   if zsr_n is not None:
  #     pl = PT.get_value(PT.get_child_from_name(zsr_n, 'PointList'))[0]-1-194
  #     fld1[pl] = PT.get_value(PT.get_child_from_name(zsr_n, 'ZSR_ccx'))
  #     fld2[pl] = PT.get_value(PT.get_child_from_name(zsr_n, 'ZSR_ccy'))
  #     fld3[pl] = PT.get_value(PT.get_child_from_name(zsr_n, 'ZSR_ccz'))
  # fsfc = PT.new_FlowSolution('FlowSolBCs', loc='CellCenter',
  #                             fields={'ZSR_ccx': fld1,
  #                                     'ZSR_ccy': fld2,
  #                                     'ZSR_ccz': fld3},
  #                             parent=part_zone_ep)

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_zsr_from_family.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_zsr_from_family.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')               , comm)

  # > Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)


@pytest_parallel.mark.parallel([1,3])
def test_extract_from_bc_name_S(comm, write_output):

  # > Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,'Vertex','Structured',comm)

  # > Initialize BCDataSet
  for zone in PT.get_all_Zone_t(part_tree):
    initialize_bc(zone, 'Ymax')
    for bc_n in PT.get_nodes_from_label(zone, 'BC_t'):
      PT.set_value(bc_n, 'FamilySpecified')
      PT.new_node('FamilyName', label='FamilyName_t', value='ALL_BCS', parent=bc_n)

  part_base = PT.get_child_from_label(part_tree, 'CGNSBase_t')
  PT.new_Family('ALL_BCS', parent=part_base)

  # > Extract part
  part_tree_ep = EXP.extract_part_from_bc_name(part_tree, "Ymax", comm,
                                               transfer_dataset=True,
                                               containers_name=['FlowSolution_NC']
                                               )

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # # > Compare to reference solution
  # ref_file = os.path.join(ref_dir, f'extract_s_ymax.cgns')
  # ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_s_ymax.cgns'), comm)
    # Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')       , comm)

  dist_zone_ep = PT.get_node_from_label(dist_tree_ep, 'Zone_t')
  assert PT.Zone.n_vtx( dist_zone_ep)==36
  assert PT.Zone.n_cell(dist_zone_ep)==25


@pytest_parallel.mark.parallel([1,3])
def test_extract_cell_from_zsr_S(comm, write_output):

  # > Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,'CellCenter','Structured',comm)

  # > Extract part
  part_tree_ep = EXP.extract_part_from_zsr(part_tree, "ZSR_FlowSolution", comm,
                                           containers_name=['FlowSolution_NC','FlowSolution_CC','ZSR_FlowSolution'],
                                           )

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_cell_from_zsr_S.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell_from_zsr_S.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')       , comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)


@pytest.mark.parametrize("zsr_loc", ["Vertex", "CellCenter"])
@pytest_parallel.mark.parallel([1,3])
def test_extractor_cell_from_zsr_S(zsr_loc, comm, write_output):

  def plane_eq(x,y,z) :
    peq1 = [0.,  1., 0., 0.6]
    peq2 = [0., -1., 0., 0.6]
    behind_plane1 = x*peq1[0] + y*peq1[1] + z*peq1[2] - peq1[3] < 0.
    behind_plane2 = x*peq2[0] + y*peq2[1] + z*peq2[2] - peq2[3] < 0.
    between_planes = np.logical_and(behind_plane1, behind_plane2)
    return between_planes

  # > Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,zsr_loc,'Structured',comm)

  for zone in PT.get_all_Zone_t(part_tree):
    # Rename zsr in tree
    zsr_n = PT.get_node_from_name(zone, "ZSR_FlowSolution")
    if zsr_n is not None:
      PT.set_name(zsr_n , "ZSR_x")

    # Create second zsr in tree
    cell_center = maia.algo.part.geometry.compute_cell_center(zone)
    ccx = cell_center[0::3]
    ccy = cell_center[1::3]
    ccz = cell_center[2::3]
    initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, "CellCenter")
    zsr_n = PT.get_node_from_name(zone, "ZSR_FlowSolution")
    if zsr_n is not None:
      PT.set_name(zsr_n , "ZSR_y")

  # > Extract part
  part_tree_ep = EXP.extract_part_from_zsr(part_tree, "ZSR_y", comm,
                                           transfer_dataset=False,
                                           containers_name=['ZSR_x'],
                                           )

  # > Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,comm)

  # > Compare to reference solution
  file_name = f'extractor_cell_from_zsr_vtx_S' if zsr_loc=='Vertex' else f'extractor_cell_from_zsr_cell_S'
  ref_file = os.path.join(ref_dir, f'{file_name}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, f'{file_name}.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')       , comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_ep, type_tol=True)
