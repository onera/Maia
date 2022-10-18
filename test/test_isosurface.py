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

from maia.algo.part import isosurf as ISS

# ========================================================================================
# ----------------------------------------------------------------------------------------
# Reference directory
ref_dir  = os.path.join(os.path.dirname(__file__), 'references')
# ----------------------------------------------------------------------------------------
# ========================================================================================



# ========================================================================================
# ----------------------------------------------------------------------------------------
@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["QUAD_4","NGON_n"])
@mark_mpi_test([1, 3])
def test_isosurf_U(elt_type,sub_comm, write_output):
  
  # Cube generation
  n_vtx = 6
  dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)
  
  # Partionning option
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, 2)
  part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)

  # Solution initialisation
  for zone in PT.get_all_Zone_t(part_tree):
    # Coordinates
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
    fld1 =  CX**2 + CY**2 + CZ**2 - 1
    fld2 =  CX**2 + CY**2 -1

    flds    = [fld1,fld2]
    name_f  = ["sphere","cylinder"]

    # Placement
    FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
    FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    
    for name,fld in zip(name_f,flds):
      # Node sol -> Cell sol
      fld_cell_vtx  = fld[cell_vtx-1]
      fld_cc        = np.add.reduceat(fld_cell_vtx, cell_vtx_idx[:-1])
      fld_cc        = fld_cc/ np.diff(cell_vtx_idx)
      
      # Placement
      PT.new_DataArray(name, fld_cc, parent=FS_CC)
      PT.new_DataArray(name, fld   , parent=FS_NC)


  container     = ['FlowSolution_NC','FlowSolution_CC']
  part_tree_iso = ISS.iso_surface(part_tree,
                                  "FlowSolution_NC/cylinder",
                                  sub_comm,
                                  iso_val=0.,
                                  interpolate=container,
                                  elt_type=elt_type)
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, 'isosurf.cgns'), sub_comm)
  
  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'isosurf_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_node(PT.get_all_CGNSBase_t(ref_sol      )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_iso)[0])

# ----------------------------------------------------------------------------------------
# ========================================================================================




# ========================================================================================
# ----------------------------------------------------------------------------------------
@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["TRI_3","NGON_n"])
@mark_mpi_test([1, 3])
def test_plane_slice_U(elt_type,sub_comm, write_output):
  
  # Cube generation
  n_vtx = 6
  dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)
  
  # Partionning option
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, 2)
  part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)

  # Solution initialisation
  for zone in PT.get_all_Zone_t(part_tree):
    # Coordinates
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
    fld1 =  CX**2 + CY**2 + CZ**2 - 1
    fld2 =  CX**2 + CY**2 -1

    flds    = [fld1,fld2]
    name_f  = ["sphere","cylinder"]

    # Placement
    FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
    FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    
    for name,fld in zip(name_f,flds):
      # Node sol -> Cell sol
      fld_cell_vtx  = fld[cell_vtx-1]
      fld_cc        = np.add.reduceat(fld_cell_vtx, cell_vtx_idx[:-1])
      fld_cc        = fld_cc/ np.diff(cell_vtx_idx)
      
      # Placement
      PT.new_DataArray(name, fld_cc, parent=FS_CC)
      PT.new_DataArray(name, fld   , parent=FS_NC)

  container     = ['FlowSolution_NC','FlowSolution_CC']
  part_tree_iso = ISS.plane_slice(part_tree,
                                  [1.,1.,1.,0.2],
                                  sub_comm,
                                  interpolate=container,
                                  elt_type=elt_type)
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, f'plane_slice.cgns'), sub_comm)
  
  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'plane_slice_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_node(PT.get_all_CGNSBase_t(ref_sol      )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_iso)[0])

# ----------------------------------------------------------------------------------------
# ========================================================================================




# ========================================================================================
# ----------------------------------------------------------------------------------------
@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["TRI_3","QUAD_4"])
@mark_mpi_test([1, 3])
def test_spherical_slice_U(elt_type,sub_comm, write_output):
  
  # Cube generation
  n_vtx = 5
  dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)
  
  # Partionning option
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, 2)
  part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)

  # Solution initialisation
  for zone in PT.get_all_Zone_t(part_tree):
    # Coordinates
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
    fld1 =  CX**2 + CY**2 + CZ**2 - 1
    fld2 =  CX**2 + CY**2 -1

    flds    = [fld1,fld2]
    name_f  = ["sphere","cylinder"]

    # Placement
    FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
    FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    
    for name,fld in zip(name_f,flds):
      # Node sol -> Cell sol
      fld_cell_vtx  = fld[cell_vtx-1]
      fld_cc        = np.add.reduceat(fld_cell_vtx, cell_vtx_idx[:-1])
      fld_cc        = fld_cc/ np.diff(cell_vtx_idx)
      
      # Placement
      PT.new_DataArray(name, fld_cc, parent=FS_CC)
      PT.new_DataArray(name, fld   , parent=FS_NC)

  container     = ['FlowSolution_NC','FlowSolution_CC']
  part_tree_iso = ISS.spherical_slice(part_tree,
                                      [0.,0.,0.,2.],
                                      sub_comm,
                                      interpolate=container,
                                      elt_type=elt_type)
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,sub_comm)


  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, f'spherical_slice.cgns'), sub_comm)
  
  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'spherical_slice_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  # Check that bases are similar (because CGNSLibraryVersion is R4)
  assert maia.pytree.is_same_node(PT.get_all_CGNSBase_t(ref_sol      )[0],
                                  PT.get_all_CGNSBase_t(dist_tree_iso)[0])

# ----------------------------------------------------------------------------------------
# ========================================================================================

