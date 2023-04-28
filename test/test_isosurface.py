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

# Reference directory
ref_dir  = os.path.join(os.path.dirname(__file__), 'references')


def generate_test_tree(n_vtx,n_part,sub_comm, build_bc_zsr=False):

  dist_tree = MF.generate_dist_block(n_vtx, "Poly", sub_comm, [-2.5, -2.5, -2.5], 5.)
  
  # Partionning option
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, n_part)
  part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)

  # Solution initialisation
  for zone in PT.get_all_Zone_t(part_tree):
    # Coordinates
    gc = PT.get_child_from_name(zone, 'GridCoordinates')
    cx = PT.get_child_from_name(gc, 'CoordinateX')[1]
    cy = PT.get_child_from_name(gc, 'CoordinateY')[1]
    cz = PT.get_child_from_name(gc, 'CoordinateZ')[1]

    cell_center = maia.algo.part.geometry.compute_cell_center(zone)
    ccx = cell_center[0::3]
    ccy = cell_center[1::3]
    ccz = cell_center[2::3]

    # Fields
    fld1_nc =   cx**2 +  cy**2 +  cz**2 - 4.
    fld2_nc =   cx**2 +  cy**2          - 1.
    fld1_cc =  ccx**2 + ccy**2 + ccz**2 - 4.
    fld2_cc =  ccx**2 + ccy**2          - 1.

    # Placement
    FS_NC = PT.new_FlowSolution('FlowSolution_NC', loc="Vertex"    , parent=zone)
    FS_CC = PT.new_FlowSolution('FlowSolution_CC', loc="CellCenter", parent=zone)
    PT.new_DataArray('sphere'  , fld1_nc, parent=FS_NC)
    PT.new_DataArray('cylinder', fld2_nc, parent=FS_NC)
    PT.new_DataArray('sphere'  , fld1_cc, parent=FS_CC)
    PT.new_DataArray('cylinder', fld2_cc, parent=FS_CC)

    # BCs ZSR
    bcs_pl = np.concatenate([PT.get_value(pl_n)[0] for pl_n in PT.get_children_from_predicates(zone, 'ZoneBC_t/BC_t/PointList')])
    bcs_pl = np.flip(bcs_pl) # Reverse to be sure of the p2p_gnum_come_from (as in extract_part)
    bcs_gnum = PT.maia.getGlobalNumbering(PT.get_child_from_name(zone, 'NGonElements'), 'Element')[1][bcs_pl-1]
    zsr_n = PT.new_ZoneSubRegion("ZSR_BC", loc='FaceCenter', point_list=bcs_pl.reshape(1,-1), parent=zone)
    PT.new_DataArray('face_gnum', bcs_gnum, parent=zsr_n)

  return part_tree


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["QUAD_4","NGON_n"])
@mark_mpi_test([1, 3])
def test_isosurf_U(elt_type,sub_comm, write_output):
  
  # Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree = generate_test_tree(n_vtx, n_part, sub_comm)

  containers    = ['FlowSolution_NC','FlowSolution_CC']
  part_tree_iso = ISS.iso_surface(part_tree,
                                  "FlowSolution_NC/cylinder",
                                  sub_comm,
                                  iso_val=0.,
                                  containers_name=containers,
                                  elt_type=elt_type)
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,sub_comm)
  
  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'isosurf_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, 'isosurf.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol, os.path.join(out_dir, f'ref_sol.cgns'), sub_comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_iso, abs_tol=1E-15, type_tol=True)


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["TRI_3","NGON_n"])
@mark_mpi_test([1, 3])
def test_plane_slice_U(elt_type,sub_comm, write_output):
  
  # Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree = generate_test_tree(n_vtx, n_part, sub_comm, build_bc_zsr=True)

  containers    = ['FlowSolution_NC','FlowSolution_CC','ZSR_BC']
  part_tree_iso = ISS.plane_slice(part_tree,
                                  [1.,1.,1.,0.2],
                                  sub_comm,
                                  containers_name=containers,
                                  elt_type=elt_type)
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,sub_comm)
  
  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'plane_slice_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, f'plane_slice.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol, os.path.join(out_dir, f'ref_sol.cgns'), sub_comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_iso, abs_tol=5E-15, type_tol=True)


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["TRI_3","QUAD_4"])
@mark_mpi_test([1, 3])
def test_spherical_slice_U(elt_type,sub_comm, write_output):
  
  # Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree = generate_test_tree(n_vtx, n_part, sub_comm)

  containers    = ['FlowSolution_NC','FlowSolution_CC']
  part_tree_iso = ISS.spherical_slice(part_tree,
                                      [0.,0.,0.,2.],
                                      sub_comm,
                                      containers_name=containers,
                                      elt_type=elt_type)
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,sub_comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'spherical_slice_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, f'spherical_slice.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol, os.path.join(out_dir, f'ref_sol.cgns'), sub_comm)
  
  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_iso, abs_tol=5E-15, type_tol=True)


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["TRI_3"])
@mark_mpi_test([1, 3])
def test_plane_slice_gc_U(elt_type,sub_comm, write_output):
  
  # Load mesh with GCs
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_Naca0012_multizone.yaml', sub_comm)

  n_part = 2
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, n_part)
  part_tree     = MF.partition_dist_tree(dist_tree, sub_comm,
                                         graph_part_tool='hilbert',
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)
  
  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree, os.path.join(out_dir, f'volumic_mesh.cgns'), sub_comm)
    Mio.write_trees(part_tree, os.path.join(out_dir, f'part_tree.cgns'), sub_comm)

  part_tree_iso = ISS.plane_slice(part_tree,
                                  [0.,0.,1.,0.5],
                                  sub_comm,
                                  # containers_name=containers,
                                  elt_type=elt_type)

  iso_base = PT.get_node_from_label(part_tree_iso, "CGNSBase_t")
  PT.new_Family("DeBuG", parent=iso_base)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.write_trees(part_tree_iso, os.path.join(out_dir, f'iso_part_tree.cgns'), sub_comm)

  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,sub_comm)
  # iso_base = PT.get_node_from_label(dist_tree_iso, "CGNSBase_t")
  # PT.new_Family("DeBuG", parent=iso_base)
  
  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'plane_slice_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, sub_comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, f'plane_slice.cgns'), sub_comm)
    Mio.dist_tree_to_file(ref_sol, os.path.join(out_dir, f'ref_sol.cgns'), sub_comm)

  # # Recover dist tree force R4 so use type_tol=True
  # assert maia.pytree.is_same_tree(ref_sol, dist_tree_iso, type_tol=True)
