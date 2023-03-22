import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import mpi4py.MPI as MPI
import numpy      as np

import Pypdm.Pypdm  as PDM
import maia.pytree  as PT

import maia
import maia.factory as MF
import maia.io      as Mio

from maia.algo.part import extract_part as EXP

# ========================================================================================
# ----------------------------------------------------------------------------------------
# Reference directory
ref_dir  = os.path.join(os.path.dirname(__file__), 'references')
# ----------------------------------------------------------------------------------------
# ========================================================================================


PART_TOOLS = []
if maia.pdm_has_parmetis:
  PART_TOOLS.append("parmetis")
if maia.pdm_has_ptscotch:
  PART_TOOLS.append("ptscotch")


# =======================================================================================
# ---------------------------------------------------------------------------------------
def plane_eq(x,y,z) :
  peq1 = [ 1., 0., 0., 0.6]
  peq2 = [-1., 0., 0., 0.6]
  behind_plane1 = x*peq1[0] + y*peq1[1] + z*peq1[2] - peq1[3] < 0.
  behind_plane2 = x*peq2[0] + y*peq2[1] + z*peq2[2] - peq2[3] < 0.
  between_planes = np.logical_and(behind_plane1, behind_plane2)
  return between_planes
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def initialize_zsr_by_eq(zone, variables, function, location, sub_comm):
  # In/out selection array
  in_extract_part = function(*variables)
  
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
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def generate_test_tree(n_vtx,n_part,location,sub_comm):

  # --- CUBE GEN AND PART ---------------------------------------------------------------
  # Cube generation
  dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)

  # Partionning option
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, n_part)
  part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)
  # -------------------------------------------------------------------------------------

  # --- INIT ZSR FIELDS -----------------------------------------------------------------
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

    # Placement FlowSolution
    FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
    FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    PT.new_DataArray('sphere'  , sphere_fld_nc, parent=FS_NC)
    PT.new_DataArray('cylinder', cylind_fld_nc, parent=FS_NC)
    PT.new_DataArray('sphere'  , sphere_fld_cc, parent=FS_CC)
    PT.new_DataArray('cylinder', cylind_fld_cc, parent=FS_CC)

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
    point_list_loc = initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, location,sub_comm)
    
    # Put fld in ZSR
    zsr_node = PT.get_node_from_name(zone,'ZSR_FlowSolution')
    PT.new_DataArray("ZSR_ccx", ccx[point_list_loc[0]-elt_range[0]], parent=zsr_node)
    PT.new_DataArray("ZSR_ccy", ccy[point_list_loc[0]-elt_range[0]], parent=zsr_node)
    PT.new_DataArray("ZSR_ccz", ccz[point_list_loc[0]-elt_range[0]], parent=zsr_node)

    point_list.append(point_list_loc[0])
  # ---------------------------------------------------------------------------------------

  return part_tree, point_list
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# =======================================================================================







# =======================================================================================
# ---------------------------------------------------------------------------------------
# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
# @pytest.mark.parametrize("graph_part_tool", ["hilbert",'parmetis'])
@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@mark_mpi_test([1,3])
def test_extract_cell_from_zsr_U(graph_part_tool, sub_comm, write_output):

  # --- GENERATE TREE -------------------------------------------------------------------
  n_vtx  = 6
  n_part = 2
  part_tree, _ = generate_test_tree(n_vtx,n_part,'CellCenter',sub_comm)
  # ------------------------------------------------------------------------------------- 

  # --- EXTRACT PART --------------------------------------------------------------------
  part_tree_ep = EXP.extract_part_from_zsr( part_tree, "ZSR_FlowSolution", sub_comm,
                                            # equilibrate=1,
                                            graph_part_tool=graph_part_tool,
                                            containers_name=['FlowSolution_NC','FlowSolution_CC',"ZSR_FlowSolution"]
                                            )
  # ------------------------------------------------------------------------------------- 

  # -------------------------------------------------------------------------------------
  # Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_cell_from_zsr.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), sub_comm)
  
  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_ep)[0])
  # -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@mark_mpi_test([1,3])
def test_extractor_cell_from_zsr_U(graph_part_tool, sub_comm, write_output):

  # --- GENERATE TREE -------------------------------------------------------------------
  n_vtx  = 6
  n_part = 2
  part_tree, _ = generate_test_tree(n_vtx,n_part,'CellCenter',sub_comm)
  # ------------------------------------------------------------------------------------- 

  # --- EXTRACT PART --------------------------------------------------------------------
  extractor = EXP.create_extractor_from_zsr(part_tree, "ZSR_FlowSolution", sub_comm,
                                            # equilibrate=1,
                                            # graph_part_tool="hilbert"
                                            )
  extractor.exchange_fields(['FlowSolution_NC','FlowSolution_CC'])
  # extractor.exchange_zsr_fields("ZSR_FlowSolution", sub_comm)
  extractor.exchange_fields(["ZSR_FlowSolution"]) # Works also with the ZSR node
  part_tree_ep = extractor.get_extract_part_tree()
  # ------------------------------------------------------------------------------------- 

  # -------------------------------------------------------------------------------------
  # Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_cell_from_zsr.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), sub_comm)
  
  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_ep)[0])
  # -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@mark_mpi_test([1,3])
def test_extract_cell_from_point_list_U(graph_part_tool, sub_comm, write_output):

  # --- GENERATE TREE -------------------------------------------------------------------
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,'CellCenter',sub_comm)
  # ------------------------------------------------------------------------------------- 

  # --- EXTRACT PART --------------------------------------------------------------------
  extractor = EXP.Extractor(part_tree, [point_list], "CellCenter", sub_comm,
                            # equilibrate=1,
                            # graph_part_tool=graph_part_tool,
                           )
  extractor.exchange_fields(['FlowSolution_NC','FlowSolution_CC'])
  part_tree_ep = extractor.get_extract_part_tree()
  # ------------------------------------------------------------------------------------- 

  # -------------------------------------------------------------------------------------
  # Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_cell_from_point_list.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), sub_comm)
  
  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_ep)[0])
  # -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# =======================================================================================




# =======================================================================================
# ---------------------------------------------------------------------------------------
# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
@pytest.mark.parametrize("graph_part_tool", PART_TOOLS)
@mark_mpi_test([1,3])
def test_extract_face_from_point_list_U(graph_part_tool, sub_comm, write_output):

  # --- GENERATE TREE -------------------------------------------------------------------
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,'FaceCenter',sub_comm)
  # -------------------------------------------------------------------------------------

  # --- EXTRACT PART --------------------------------------------------------------------
  extractor = EXP.Extractor(part_tree, [point_list], "FaceCenter", sub_comm,
                            # equilibrate=1,
                            graph_part_tool=graph_part_tool
                           )  
  extractor.exchange_fields(['FlowSolution_NC',"ZSR_FlowSolution"])
  part_tree_ep = extractor.get_extract_part_tree()
  # -------------------------------------------------------------------------------------

  # -------------------------------------------------------------------------------------
  # Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_face_from_point_list.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_face_from_point_list.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), sub_comm)

  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_ep)[0])
  # -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# =======================================================================================





# =======================================================================================
# ---------------------------------------------------------------------------------------
# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@mark_mpi_test([1,3])
def test_extract_vertex_from_zsr_U(graph_part_tool, sub_comm, write_output):

  # --- GENERATE TREE -------------------------------------------------------------------
  n_vtx  = 6
  n_part = 2
  part_tree, point_list = generate_test_tree(n_vtx,n_part,'Vertex',sub_comm)
  # -------------------------------------------------------------------------------------

  # # --- EXTRACT PART --------------------------------------------------------------------
  part_tree_ep = EXP.extract_part_from_zsr( part_tree, "ZSR_FlowSolution", sub_comm,
                                            # equilibrate=1,
                                            graph_part_tool=graph_part_tool,
                                            containers_name=['FlowSolution_NC',"ZSR_FlowSolution"]
                                            )
  # Sortie VTK for visualisation
  # part_zones = PT.get_all_Zone_t(part_tree_ep)
  # for i_zone, part_zone in enumerate(part_zones):
  #   write_part_zone_vtx(i_zone,part_zone,sub_comm)
  # Mio.write_trees(part_tree_ep,'OUT_TEST_VERTEX/part_tree_extract.cgns',sub_comm)
  # -------------------------------------------------------------------------------------

  # -------------------------------------------------------------------------------------
  # Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_vertex_from_zsr.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_vertex_from_zsr.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')                , sub_comm)

  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_ep)[0])
  # -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# =======================================================================================


# =======================================================================================
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("graph_part_tool", ["hilbert"])
@mark_mpi_test([1,3])
def test_extract_bc_from_bc_name_U(graph_part_tool, sub_comm, write_output):

  # --- GENERATE TREE -------------------------------------------------------------------
  n_vtx  = 6
  n_part = 4
  part_tree, _ = generate_test_tree(n_vtx,n_part,'CellCenter',sub_comm)

  for zone in PT.get_all_Zone_t(part_tree):
    face_center = maia.algo.part.geometry.compute_face_center(zone)
    cfx = face_center[0::3]
    cfy = face_center[1::3]
    cfz = face_center[2::3]

    bc_n = PT.get_node_from_name(zone, 'Xmin')
    if bc_n is not None:
      bc_pl  = PT.get_node_from_name(bc_n, 'PointList')[1][0]
      bc_cfx = cfx[bc_pl-1]
      bc_cfy = cfy[bc_pl-1]
      bc_cfz = cfz[bc_pl-1]
      bc_dataset_n = PT.new_node(name='BCDataSet'  , label='BCDataSet_t', value='UserDefined', parent=bc_n)
      neuma_data_n = PT.new_node(name='NeumannData', label='BCData_t'   , value=None         , parent=bc_dataset_n)
      grid_loc_n   = PT.new_GridLocation('FaceCenter', parent=neuma_data_n)
      sphere       = PT.new_DataArray('sphere_bc'  , bc_cfx**2 + bc_cfy**2 + bc_cfz**2 - 1, parent=neuma_data_n)
      cylinder     = PT.new_DataArray('cylinder_bc', bc_cfx**2 + bc_cfy**2             - 1, parent=neuma_data_n)
  # ------------------------------------------------------------------------------------- 

  # --- EXTRACT PART --------------------------------------------------------------------
  part_tree_ep = EXP.extract_part_from_bc_name( part_tree, "Xmin", sub_comm,
                                                graph_part_tool=graph_part_tool,
                                                containers_name=['FlowSolution_NC'],
                                                )
  # ------------------------------------------------------------------------------------- 

  # -------------------------------------------------------------------------------------
  # Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)
  PT.get_node_from_label(dist_tree_ep,'ZoneSubRegion_t')[3] = 'FlowSolution_t'

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_bc_from_bc_name.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_bc_from_bc_name.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns')                , sub_comm)

  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_ep)[0])
  # -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# =======================================================================================
