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

from maia.algo.part import geometry     as GEO
from maia.algo.part import extract_part as EXP

# ========================================================================================
# ----------------------------------------------------------------------------------------
# Reference directory
ref_dir  = os.path.join(os.path.dirname(__file__), 'references')
# ----------------------------------------------------------------------------------------
# ========================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def plane_eq(x,y,z) :
  peq1 = [ 1., 0., 0., 0.5]
  peq2 = [-1., 0., 0., 0.5]
  behind_plane1 = x*peq1[0] + y*peq1[1] + z*peq1[2] - peq1[3] < 0.
  behind_plane2 = x*peq2[0] + y*peq2[1] + z*peq2[2] - peq2[3] < 0.
  between_planes = np.logical_and(behind_plane1, behind_plane2)
  return between_planes
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def compute_face_center(zone):
  cx,cy,cz = PT.Zone.coordinates(zone)

  if PT.Zone.Type(zone) == "Unstructured":
    ngons  = [e for e in PT.iter_children_from_label(zone, 'Elements_t') if PT.Element.CGNSName(e) == 'NGON_n']
    if len(ngons) != 1:
      raise NotImplementedError(f"Cell center computation is only available for NGON connectivity")
    face_vtx_idx, face_vtx, ngon_pe = PT.Zone.ngon_connectivity(zone)
    center_face_x = np.add.reduceat(cx[face_vtx-1], face_vtx_idx[:-1])/np.diff(face_vtx_idx)
    center_face_y = np.add.reduceat(cy[face_vtx-1], face_vtx_idx[:-1])/np.diff(face_vtx_idx)
    center_face_z = np.add.reduceat(cz[face_vtx-1], face_vtx_idx[:-1])/np.diff(face_vtx_idx)
  else:
    print('[compute_face_center] Structured not implemented')

  return center_face_x,center_face_y,center_face_z
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def initialize_zsr_by_eq(zone, variables, function, location, sub_comm):
  # In/out selection array
  in_extract_part = function(*variables)
  
  # Get loc zsr
  extract_lnum = np.where(in_extract_part)[0]
  extract_lnum = extract_lnum.astype(np.int32)+1
  extract_lnum = np.flip(extract_lnum) # to be sure that desorganized PL is not a problem
  extract_lnum = extract_lnum.reshape((1,-1), order='F') # Ordering in shape (1,N) because of CGNS standard

  PT.new_ZoneSubRegion("ZSR_FlowSolution", point_list=extract_lnum, loc=location, parent=zone)
  return np.ascontiguousarray(extract_lnum)
# -----------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# =======================================================================================







# ========================================================================================
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
@mark_mpi_test([1,3])
def test_extract_cell_from_zsr_U(graph_part_tool, sub_comm, write_output):

  # --- CUBE GEN AND PART -----------------------------------------------------------------
  # Cube generation
  n_vtx = 6
  dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)

  # Partionning option
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, 2)
  part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)
  # ---------------------------------------------------------------------------------------

  # --- INIT ZSR FIELDS -------------------------------------------------------------------
  for zone in PT.get_all_Zone_t(part_tree):
    # Nodes coordinates
    GC = PT.get_child_from_name(zone, 'GridCoordinates')
    CX = PT.get_child_from_name(GC, 'CoordinateX')[1]
    CY = PT.get_child_from_name(GC, 'CoordinateY')[1]
    CZ = PT.get_child_from_name(GC, 'CoordinateZ')[1]
    
    # Connectivity
    nface         = PT.get_child_from_name(zone , 'NFaceElements')
    cell_face_idx = PT.get_child_from_name(nface, 'ElementStartOffset' )[1]
    cell_face     = PT.get_child_from_name(nface, 'ElementConnectivity')[1]
    ngon          = PT.get_child_from_name(zone, 'NGonElements')
    face_vtx_idx  = PT.get_child_from_name(ngon, 'ElementStartOffset' )[1]
    face_vtx      = PT.get_child_from_name(ngon, 'ElementConnectivity')[1]
    cell_vtx_idx,cell_vtx = PDM.combine_connectivity(cell_face_idx,cell_face,face_vtx_idx,face_vtx)

    # Fields
    name_f  = ["sphere"                 , "cylinder"      ]
    flds    = [CX**2 + CY**2 + CZ**2 - 1, CX**2 + CY**2 -1]
    
    # Placement FlowSolution
    FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
    FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    
    # Placement flds
    for name,fld in zip(name_f,flds):
      # Node sol -> Cell sol
      fld_cc = np.add.reduceat(fld[cell_vtx-1], cell_vtx_idx[:-1])/ np.diff(cell_vtx_idx)
      
      PT.new_DataArray(name, fld_cc, parent=FS_CC)
      PT.new_DataArray(name, fld   , parent=FS_NC)

    cell_center = GEO.compute_cell_center(zone)
    ccx = cell_center[0::3]
    ccy = cell_center[1::3]
    ccz = cell_center[2::3]
    initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, 'CellCenter',sub_comm)
    
    # Put fld in ZSR
    zsr_node   = PT.get_node_from_name(zone,'ZSR_FlowSolution')
    point_list = PT.get_value(PT.get_child_from_name(zsr_node,'PointList'))
    PT.new_DataArray("ZSR_ccx", ccx[point_list[0]-1], parent=zsr_node)
    PT.new_DataArray("ZSR_ccy", ccy[point_list[0]-1], parent=zsr_node)
    PT.new_DataArray("ZSR_ccz", ccz[point_list[0]-1], parent=zsr_node)
  # ---------------------------------------------------------------------------------------

  # --- EXTRACT PART ----------------------------------------------------------------------
  part_tree_ep = EXP.extract_part_from_zsr( part_tree, "ZSR_FlowSolution", sub_comm,
                                            equilibrate=1,
                                            # exchange=['FlowSolution_NC','FlowSolution_CC',"ZSR_FlowSolution"],
                                            exchange=["ZSR_FlowSolution"],
                                            graph_part_tool=graph_part_tool)
  # --------------------------------------------------------------------------------------- 

  # ---------------------------------------------------------------------------------------
  import Converter.Internal as I
  I.printTree(part_tree_ep)
  # Part to dist
  dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'extract_cell_from_zsr.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    # Mio.write_trees(part_tree, os.path.join(   out_dir, f'volume_{graph_part_tool}.cgns'), sub_comm)
    # Mio.write_trees(part_tree_ep, os.path.join(out_dir, f'extract_cell_{graph_part_tool}.cgns'), sub_comm)
    Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), sub_comm)
  

  # Check that bases are similar (because CGNSLibraryVersion is R4)
  # print(ref_sol     [2][0][1])
  # print(dist_tree_ep[2][0][1])
  # print("RESULT FORM ASSERT = ",maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
  #                                 PT.get_all_CGNSBase_t(dist_tree_ep)[0]))
  assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_ep)[0])
  # ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# =======================================================================================



# # ========================================================================================
# # ----------------------------------------------------------------------------------------
# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
# @mark_mpi_test([1,3])
# def test_extractor_cell_from_zsr_U(graph_part_tool, sub_comm, write_output):

#   # --- CUBE GEN AND PART -----------------------------------------------------------------
#   # Cube generation
#   n_vtx = 6
#   dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)

#   # Partionning option
#   zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, 2)
#   part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
#                                          zone_to_parts=zone_to_parts,
#                                          preserve_orientation=True)
#   # ---------------------------------------------------------------------------------------

#   # --- INIT ZSR FIELDS -------------------------------------------------------------------
#   for zone in PT.get_all_Zone_t(part_tree):
#     # Nodes coordinates
#     GC = PT.get_child_from_name(zone, 'GridCoordinates')
#     CX = PT.get_child_from_name(GC, 'CoordinateX')[1]
#     CY = PT.get_child_from_name(GC, 'CoordinateY')[1]
#     CZ = PT.get_child_from_name(GC, 'CoordinateZ')[1]
    
#     # Connectivity
#     nface         = PT.get_child_from_name(zone , 'NFaceElements')
#     cell_face_idx = PT.get_child_from_name(nface, 'ElementStartOffset' )[1]
#     cell_face     = PT.get_child_from_name(nface, 'ElementConnectivity')[1]
#     ngon          = PT.get_child_from_name(zone, 'NGonElements')
#     face_vtx_idx  = PT.get_child_from_name(ngon, 'ElementStartOffset' )[1]
#     face_vtx      = PT.get_child_from_name(ngon, 'ElementConnectivity')[1]
#     cell_vtx_idx,cell_vtx = PDM.combine_connectivity(cell_face_idx,cell_face,face_vtx_idx,face_vtx)

#     # Fields
#     name_f  = ["sphere"                 , "cylinder"      ]
#     flds    = [CX**2 + CY**2 + CZ**2 - 1, CX**2 + CY**2 -1]
    
#     # Placement FlowSolution
#     FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
#     FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    
#     # Placement flds
#     for name,fld in zip(name_f,flds):
#       # Node sol -> Cell sol
#       fld_cc = np.add.reduceat(fld[cell_vtx-1], cell_vtx_idx[:-1])/ np.diff(cell_vtx_idx)
      
#       PT.new_DataArray(name, fld_cc, parent=FS_CC)
#       PT.new_DataArray(name, fld   , parent=FS_NC)

#     cell_center = GEO.compute_cell_center(zone)
#     ccx = cell_center[0::3]
#     ccy = cell_center[1::3]
#     ccz = cell_center[2::3]
#     initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, 'CellCenter',sub_comm)

#     # Put fld in ZSR
#     zsr_node   = PT.get_node_from_name(zone,'ZSR_FlowSolution')
#     point_list = PT.get_value(PT.get_child_from_name(zsr_node,'PointList'))
#     PT.new_DataArray("ZSR_ccx", ccx[point_list[0]-1], parent=zsr_node)
#     PT.new_DataArray("ZSR_ccy", ccy[point_list[0]-1], parent=zsr_node)
#     PT.new_DataArray("ZSR_ccz", ccz[point_list[0]-1], parent=zsr_node)
#   # ---------------------------------------------------------------------------------------

#   # --- EXTRACT PART ----------------------------------------------------------------------
#   extractor = EXP.create_extractor_from_zsr(part_tree, "ZSR_FlowSolution", sub_comm,
#                                             equilibrate=1,
#                                             graph_part_tool="hilbert")
#   extractor.exchange_fields(['FlowSolution_NC','FlowSolution_CC'], sub_comm)
#   # extractor.exchange_zsr_fields("ZSR_FlowSolution", sub_comm)
#   extractor.exchange_fields(["ZSR_FlowSolution"], sub_comm) # Works also with the ZSR node
#   part_tree_ep = extractor.extract_part_tree
#   # --------------------------------------------------------------------------------------- 

#   # ---------------------------------------------------------------------------------------
#   # Part to dist
#   dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

#   # Compare to reference solution
#   ref_file = os.path.join(ref_dir, f'extract_cell_from_zsr.yaml')
#   ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

#   if write_output:
#     out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
#     # Mio.write_trees(part_tree, os.path.join(   out_dir, f'volume_{graph_part_tool}.cgns'), sub_comm)
#     # Mio.write_trees(part_tree_ep, os.path.join(out_dir, f'extract_cell_{graph_part_tool}.cgns'), sub_comm)
#     Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), sub_comm)
#     Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), sub_comm)
  

#   # Check that bases are similar (because CGNSLibraryVersion is R4)
#   # print(ref_sol     [2][0][1])
#   # print(dist_tree_ep[2][0][1])
#   # print("RESULT FORM ASSERT = ",maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
#   #                                 PT.get_all_CGNSBase_t(dist_tree_ep)[0]))
#   assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
#                                   PT.get_all_CGNSBase_t(dist_tree_ep)[0])
#   # ---------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------------
# # =======================================================================================



# # ========================================================================================
# # ----------------------------------------------------------------------------------------
# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
# @mark_mpi_test([1,3])
# def test_extract_cell_from_point_list_U(graph_part_tool, sub_comm, write_output):

#   # --- CUBE GEN AND PART -----------------------------------------------------------------
#   # Cube generation
#   n_vtx = 6
#   dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)

#   # Partionning option
#   zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, 2)
#   part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
#                                          zone_to_parts=zone_to_parts,
#                                          preserve_orientation=True)
#   # ---------------------------------------------------------------------------------------

#   # --- INIT ZSR FIELDS -------------------------------------------------------------------
#   point_list = list()
#   for zone in PT.get_all_Zone_t(part_tree):
#     # Nodes coordinates
#     GC = PT.get_child_from_name(zone, 'GridCoordinates')
#     CX = PT.get_child_from_name(GC, 'CoordinateX')[1]
#     CY = PT.get_child_from_name(GC, 'CoordinateY')[1]
#     CZ = PT.get_child_from_name(GC, 'CoordinateZ')[1]
    
#     # Connectivity
#     nface         = PT.get_child_from_name(zone , 'NFaceElements')
#     cell_face_idx = PT.get_child_from_name(nface, 'ElementStartOffset' )[1]
#     cell_face     = PT.get_child_from_name(nface, 'ElementConnectivity')[1]
#     ngon          = PT.get_child_from_name(zone, 'NGonElements')
#     face_vtx_idx  = PT.get_child_from_name(ngon, 'ElementStartOffset' )[1]
#     face_vtx      = PT.get_child_from_name(ngon, 'ElementConnectivity')[1]
#     cell_vtx_idx,cell_vtx = PDM.combine_connectivity(cell_face_idx,cell_face,face_vtx_idx,face_vtx)

#     # Fields
#     name_f  = ["sphere"                 , "cylinder"      ]
#     flds    = [CX**2 + CY**2 + CZ**2 - 1, CX**2 + CY**2 -1]
    
#     # Placement FlowSolution
#     FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
#     FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    
#     # Placement flds
#     for name,fld in zip(name_f,flds):
#       # Node sol -> Cell sol
#       fld_cc = np.add.reduceat(fld[cell_vtx-1], cell_vtx_idx[:-1])/ np.diff(cell_vtx_idx)
      
#       PT.new_DataArray(name, fld_cc, parent=FS_CC)
#       PT.new_DataArray(name, fld   , parent=FS_NC)

#     cell_center = GEO.compute_cell_center(zone)
#     ccx = cell_center[0::3]
#     ccy = cell_center[1::3]
#     ccz = cell_center[2::3]
#     point_list_cell = initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, 'CellCenter',sub_comm)
#     point_list.append(point_list_cell[0])
#   # ---------------------------------------------------------------------------------------

#   # --- EXTRACT PART ----------------------------------------------------------------------
#   part_tree_ep = EXP.extract_part_from_point_list(part_tree, point_list, "CellCenter", sub_comm,
#                                                   equilibrate=1,
#                                                   exchange=['FlowSolution_NC','FlowSolution_CC'],
#                                                   graph_part_tool=graph_part_tool)
#   # --------------------------------------------------------------------------------------- 

#   # ---------------------------------------------------------------------------------------
#   # Part to dist
#   dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

#   # Compare to reference solution
#   ref_file = os.path.join(ref_dir, f'extract_cell_from_point_list.yaml')
#   ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

#   if write_output:
#     out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
#     # Mio.write_trees(part_tree, os.path.join(   out_dir, f'volume_{graph_part_tool}.cgns'), sub_comm)
#     # Mio.write_trees(part_tree_ep, os.path.join(out_dir, f'extract_cell_{graph_part_tool}.cgns'), sub_comm)
#     Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), sub_comm)
#     Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), sub_comm)
  

#   # Check that bases are similar (because CGNSLibraryVersion is R4)
#   # print(ref_sol     [2][0][1])
#   # print(dist_tree_ep[2][0][1])
#   # print("RESULT FORM ASSERT = ",maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
#   #                                 PT.get_all_CGNSBase_t(dist_tree_ep)[0]))
#   assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
#                                   PT.get_all_CGNSBase_t(dist_tree_ep)[0])
#   # ---------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------------
# # =======================================================================================



# # ========================================================================================
# # ----------------------------------------------------------------------------------------
# @pytest.mark.parametrize("graph_part_tool", ["hilbert","ptscotch","parmetis"])
# @mark_mpi_test([1,3])
# def test_extractor_cell_from_point_list_U(graph_part_tool, sub_comm, write_output):

#   # --- CUBE GEN AND PART -----------------------------------------------------------------
#   # Cube generation
#   n_vtx = 6
#   dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)

#   # Partionning option
#   zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, 2)
#   part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
#                                          zone_to_parts=zone_to_parts,
#                                          preserve_orientation=True)
#   # ---------------------------------------------------------------------------------------

#   # --- INIT ZSR FIELDS -------------------------------------------------------------------
#   point_list = list()
#   for zone in PT.get_all_Zone_t(part_tree):
#     # Nodes coordinates
#     GC = PT.get_child_from_name(zone, 'GridCoordinates')
#     CX = PT.get_child_from_name(GC, 'CoordinateX')[1]
#     CY = PT.get_child_from_name(GC, 'CoordinateY')[1]
#     CZ = PT.get_child_from_name(GC, 'CoordinateZ')[1]
    
#     # Connectivity
#     nface         = PT.get_child_from_name(zone , 'NFaceElements')
#     cell_face_idx = PT.get_child_from_name(nface, 'ElementStartOffset' )[1]
#     cell_face     = PT.get_child_from_name(nface, 'ElementConnectivity')[1]
#     ngon          = PT.get_child_from_name(zone, 'NGonElements')
#     face_vtx_idx  = PT.get_child_from_name(ngon, 'ElementStartOffset' )[1]
#     face_vtx      = PT.get_child_from_name(ngon, 'ElementConnectivity')[1]
#     cell_vtx_idx,cell_vtx = PDM.combine_connectivity(cell_face_idx,cell_face,face_vtx_idx,face_vtx)

#     # Fields
#     name_f  = ["sphere"                 , "cylinder"      ]
#     flds    = [CX**2 + CY**2 + CZ**2 - 1, CX**2 + CY**2 -1]
    
#     # Placement FlowSolution
#     FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
#     FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    
#     # Placement flds
#     for name,fld in zip(name_f,flds):
#       # Node sol -> Cell sol
#       fld_cc = np.add.reduceat(fld[cell_vtx-1], cell_vtx_idx[:-1])/ np.diff(cell_vtx_idx)
      
#       PT.new_DataArray(name, fld_cc, parent=FS_CC)
#       PT.new_DataArray(name, fld   , parent=FS_NC)

#     cell_center = GEO.compute_cell_center(zone)
#     ccx = cell_center[0::3]
#     ccy = cell_center[1::3]
#     ccz = cell_center[2::3]
#     point_list_cell = initialize_zsr_by_eq(zone, [ccx,ccy,ccz], plane_eq, 'CellCenter',sub_comm)
#     point_list.append(point_list_cell[0])
#   # ---------------------------------------------------------------------------------------

#   # --- EXTRACT PART ----------------------------------------------------------------------
#   extractor = EXP.create_extractor_from_point_list( part_tree, point_list, 'CellCenter', sub_comm,
#                                                     equilibrate=1,
#                                                     graph_part_tool="hilbert")
#   extractor.exchange_fields(['FlowSolution_NC','FlowSolution_CC'], sub_comm)
#   part_tree_ep = extractor.extract_part_tree
#   # --------------------------------------------------------------------------------------- 

#   # ---------------------------------------------------------------------------------------
#   # Part to dist
#   dist_tree_ep = MF.recover_dist_tree(part_tree_ep,sub_comm)

#   # Compare to reference solution
#   ref_file = os.path.join(ref_dir, f'extract_cell_from_point_list.yaml')
#   ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

#   if write_output:
#     out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
#     # Mio.write_trees(part_tree, os.path.join(   out_dir, f'volume_{graph_part_tool}.cgns'), sub_comm)
#     # Mio.write_trees(part_tree_ep, os.path.join(out_dir, f'extract_cell_{graph_part_tool}.cgns'), sub_comm)
#     Mio.dist_tree_to_file(dist_tree_ep, os.path.join(out_dir, 'extract_cell.cgns'), sub_comm)
#     Mio.dist_tree_to_file(ref_sol     , os.path.join(out_dir, 'ref_sol.cgns'), sub_comm)
  

#   # Check that bases are similar (because CGNSLibraryVersion is R4)
#   # print(ref_sol     [2][0][1])
#   # print(dist_tree_ep[2][0][1])
#   # print("RESULT FORM ASSERT = ",maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
#   #                                 PT.get_all_CGNSBase_t(dist_tree_ep)[0]))
#   assert maia.pytree.is_same_tree(PT.get_all_CGNSBase_t(ref_sol     )[0],
#                                   PT.get_all_CGNSBase_t(dist_tree_ep)[0])
#   # ---------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------------
# # =======================================================================================

