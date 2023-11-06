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

from maia.algo.part import isosurf as ISS

# Reference directory
ref_dir  = os.path.join(os.path.dirname(__file__), 'references')


def generate_test_tree(n_vtx,n_part,comm, build_bc_zsr=False):

  dist_tree = MF.generate_dist_block(n_vtx, "Poly", comm, [-2.5, -2.5, -2.5], 5.)
  
  # Partionning option
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, comm, n_part)
  part_tree     = MF.partition_dist_tree(dist_tree, comm,
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
@pytest_parallel.mark.parallel([1, 3])
def test_isosurf_U(elt_type,comm, write_output):
  
  # Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree = generate_test_tree(n_vtx, n_part, comm)

  containers    = ['FlowSolution_NC','FlowSolution_CC']
  part_tree_iso = ISS.iso_surface(part_tree,
                                  "FlowSolution_NC/cylinder",
                                  comm,
                                  iso_val=0.,
                                  containers_name=containers,
                                  elt_type=elt_type,
                                  graph_part_tool='hilbert') # Parallelism independant
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,comm)
  
  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'isosurf_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, 'isosurf.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol, os.path.join(out_dir, f'ref_sol.cgns'), comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_iso, abs_tol=1E-15, type_tol=True)


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["TRI_3","NGON_n"])
@pytest_parallel.mark.parallel([1, 3])
def test_plane_slice_U(elt_type,comm, write_output):
  
  # Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree = generate_test_tree(n_vtx, n_part, comm, build_bc_zsr=True)

  containers    = ['FlowSolution_NC','FlowSolution_CC','ZSR_BC']
  part_tree_iso = ISS.plane_slice(part_tree,
                                  [1.,1.,1.,0.2],
                                  comm,
                                  containers_name=containers,
                                  elt_type=elt_type,
                                  graph_part_tool='hilbert') # Parallelism independant
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,comm)
  
  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'plane_slice_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, f'plane_slice.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol, os.path.join(out_dir, f'ref_sol.cgns'), comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_iso, abs_tol=5E-15, type_tol=True)


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["TRI_3","QUAD_4"])
@pytest_parallel.mark.parallel([1, 3])
def test_spherical_slice_U(elt_type,comm, write_output):
  
  # Cube generation
  n_vtx  = 6
  n_part = 2
  part_tree = generate_test_tree(n_vtx, n_part, comm)

  containers    = ['FlowSolution_NC','FlowSolution_CC']
  part_tree_iso = ISS.spherical_slice(part_tree,
                                      [0.,0.,0.,2.],
                                      comm,
                                      containers_name=containers,
                                      elt_type=elt_type,
                                      graph_part_tool='hilbert') # Parallelism independant
  
  # Part to dist
  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'spherical_slice_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, f'spherical_slice.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol, os.path.join(out_dir, f'ref_sol.cgns'), comm)
  
  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_iso, abs_tol=5E-15, type_tol=True)


@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("elt_type", ["TRI_3"])
@pytest_parallel.mark.parallel(3)
def test_plane_slice_gc_U(elt_type,comm, write_output):
  
  # Load mesh with GCs
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_Naca0012_multizone.yaml', comm)

  if   comm.Get_rank()==0: zone_to_parts = {"BaseA/blk1":[0.2], "BaseA/blk3":[1.], "BaseB/blk2":[0.1]}
  elif comm.Get_rank()==1: zone_to_parts = {"BaseA/blk1":[0.8], "BaseA/blk3":[]  , "BaseB/blk2":[0.2]}
  elif comm.Get_rank()==2: zone_to_parts = {"BaseA/blk1":[]   , "BaseA/blk3":[]  , "BaseB/blk2":[0.7]}
  part_tree     = MF.partition_dist_tree(dist_tree, comm,
                                         zone_to_parts=zone_to_parts,
                                         preserve_orientation=True)
  
  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree, os.path.join(out_dir, f'volumic_mesh.cgns'), comm)
    Mio.write_trees(part_tree, os.path.join(out_dir, f'part_tree.cgns'), comm)

  part_tree_iso = ISS.plane_slice(part_tree,
                                  [0.,0.,1.,0.5],
                                  comm,
                                  elt_type=elt_type,
                                  graph_part_tool='hilbert')

  dist_tree_iso = MF.recover_dist_tree(part_tree_iso,comm)

  # Compare to reference solution
  ref_file = os.path.join(ref_dir, f'plane_slice_with_gc_{elt_type}.yaml')
  ref_sol  = Mio.file_to_dist_tree(ref_file, comm)

  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
    Mio.dist_tree_to_file(dist_tree_iso, os.path.join(out_dir, f'plane_slice_with_gc.cgns'), comm)
    Mio.dist_tree_to_file(ref_sol, os.path.join(out_dir, f'ref_sol.cgns'), comm)

  # Recover dist tree force R4 so use type_tol=True
  assert maia.pytree.is_same_tree(ref_sol, dist_tree_iso, type_tol=True)
