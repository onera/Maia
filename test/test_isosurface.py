import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import mpi4py.MPI as MPI

import Converter.Internal as I
import Converter.PyTree   as C
import maia.pytree        as PT

from maia.factory   import partitioning       as PPA
from maia.factory   import dcube_generator    as DCG
from maia.io        import dist_tree_to_file  as DTF
from maia.algo.part import isosurf            as ISS
from maia.factory   import recover_dist_tree  as part_to_dist

import Pypdm.Pypdm  as PDM
import numpy        as np
import maia


# ----------------------------------------------------------------------------------------
@mark_mpi_test([1, 2, 3])
def test_isosurf_U(sub_comm, write_output):
  
  # Cube generation
  n_vtx = 10
  dist_tree = DCG.dcube_generate(n_vtx, 5., [-2.5, -2.5, -2.5], sub_comm)
  
  # Partionning option
  zone_to_parts = PPA.compute_regular_weights(dist_tree, sub_comm, 2)
  part_tree     = PPA.partition_dist_tree(dist_tree, sub_comm, zone_to_parts=zone_to_parts)

  # Solution initialisation
  for zone in I.getZones(part_tree):
    # Coordinates
    GC = I.getNodeFromName1(zone, 'GridCoordinates')
    CX = I.getNodeFromName1(GC, 'CoordinateX')[1]
    CY = I.getNodeFromName1(GC, 'CoordinateY')[1]
    CZ = I.getNodeFromName1(GC, 'CoordinateZ')[1]

    # Connectivity
    nface         = I.getNodeFromName1(zone , 'NFaceElements')
    cell_face_idx = I.getNodeFromName1(nface, 'ElementStartOffset' )[1]
    cell_face     = I.getNodeFromName1(nface, 'ElementConnectivity')[1]
    ngon          = I.getNodeFromName1(zone, 'NGonElements')
    face_vtx_idx  = I.getNodeFromName1(ngon, 'ElementStartOffset' )[1]
    face_vtx      = I.getNodeFromName1(ngon, 'ElementConnectivity')[1]
    cell_vtx_idx,cell_vtx = PDM.combine_connectivity(cell_face_idx,cell_face,face_vtx_idx,face_vtx)

    # Fields
    fld1 =  CX**2 + CY**2 + CZ**2 - 1
    fld2 =  CX**2 + CY**2 -1
    # fld3 = (CX**2 + 9*CY**2 + CZ**2 -1 )**3 - CX**2*CZ**3 - CY**2*CZ**3

    flds    = [fld1,fld2]#,fld3]
    name_f  = ["sphere","cylinder"]#,"heart"]

    # Placement
    FS_NC = I.newFlowSolution('FlowSolution_NC', gridLocation="Vertex"    , parent=zone)
    FS_CC = I.newFlowSolution('FlowSolution_CC', gridLocation="CellCenter", parent=zone)
    
    for name,fld in zip(name_f,flds):
      # Node sol -> Cell sol
      fld_cell_vtx  = fld[cell_vtx-1]
      fld_cc        = np.add.reduceat(fld_cell_vtx, cell_vtx_idx[:-1])
      fld_cc        = fld_cc/ np.diff(cell_vtx_idx)
      
      # Placement
      I.newDataArray(name, fld_cc, parent=FS_CC)
      I.newDataArray(name, fld   , parent=FS_NC)


  #iso_kind = ["PLANE", [1.,0.,0.,0.]]

  container     = ['FlowSolution_NC','FlowSolution_CC']
  part_tree_iso = ISS.iso_surface(part_tree, "FlowSolution_NC/cylinder", sub_comm, interpolate=container)
  
  # Part to dist
  dist_tree_iso = part_to_dist(part_tree_iso,sub_comm)

  # Verification
  node_sol_nc   =            I.getNodeFromName(dist_tree_iso,'FlowSolution_NC')
  sphere_nc     = I.getValue(I.getNodeFromName(node_sol_nc  ,'sphere'        ))
  cylinder_nc   = I.getValue(I.getNodeFromName(node_sol_nc  ,'cylinder'      ))
  node_sol_cc   =            I.getNodeFromName(dist_tree_iso,'FlowSolution_CC')
  sphere_cc     = I.getValue(I.getNodeFromName(node_sol_cc  ,'sphere'        ))
  cylinder_cc   = I.getValue(I.getNodeFromName(node_sol_cc  ,'cylinder'      ))

  min_sphere_nc   = sub_comm.allreduce(np.min(sphere_nc  ),MPI.MIN)
  min_sphere_cc   = sub_comm.allreduce(np.min(sphere_cc  ),MPI.MIN)
  max_sphere_nc   = sub_comm.allreduce(np.max(sphere_nc  ),MPI.MAX)
  max_sphere_cc   = sub_comm.allreduce(np.max(sphere_cc  ),MPI.MAX)

  min_cylinder_nc = sub_comm.allreduce(np.min(cylinder_nc),MPI.MIN)
  min_cylinder_cc = sub_comm.allreduce(np.min(cylinder_cc),MPI.MIN)
  max_cylinder_nc = sub_comm.allreduce(np.max(cylinder_nc),MPI.MAX)
  max_cylinder_cc = sub_comm.allreduce(np.max(cylinder_cc),MPI.MAX)


  if (sub_comm.Get_rank()==0) :
    print( "\n========================================")
    print(f" * NPROC = {sub_comm.Get_size()}")
    print( "    -> NODE_CENTERED FIELD : ")
    print(f"        - MIN(sphere)   = {min_sphere_nc  } ; MAX(sphere)   = {max_sphere_nc  } ")
    print(f"        - MIN(cylinder) = {min_cylinder_nc} ; MAX(cylinder) = {max_cylinder_nc} ")
    print( "    -> CELL_CENTERED FIELD : ")
    print(f"        - MIN(sphere)   = {min_sphere_cc  } ; MAX(sphere)   = {max_sphere_cc  } ")
    print(f"        - MIN(cylinder) = {min_cylinder_cc} ; MAX(cylinder) = {max_cylinder_cc} ")
    print( "========================================")

  if write_output:
    out_dir = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    dist_tree = maia.factory.recover_dist_tree(part_tree,sub_comm)
    DTF(dist_tree, os.path.join(out_dir, 'dist_volume.cgns'), sub_comm)
    DTF(dist_tree_iso, os.path.join(out_dir, 'dist_isosurf.cgns'), sub_comm)

# ----------------------------------------------------------------------------------------