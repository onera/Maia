# ========================================================================================
# ----------------------------------------------------------------------------------------
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
from maia.algo.part import extract_part       as EXP
from maia.algo.part import geometry           as GEO
from maia.factory   import recover_dist_tree  as part_to_dist

import Pypdm.Pypdm  as PDM
import numpy        as np
import maia
# ----------------------------------------------------------------------------------------
# ========================================================================================


# # ========================================================================================
# # ----------------------------------------------------------------------------------------
# # Reference directory
# ref_dir  = os.path.join(os.path.dirname(__file__), 'references')
# # ----------------------------------------------------------------------------------------
# # ========================================================================================


# =======================================================================================
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def initialize_fld(zone):
  # Vtx coordinates
  CX, CY, CZ = PT.Zone.coordinates(zone)

  # Cell centers coordinates
  cell_center = GEO.compute_cell_center(zone)
  ccx = cell_center[0::3]
  ccy = cell_center[1::3]
  ccz = cell_center[2::3]

  # Fields node centered
  FS  = I.newFlowSolution('FlowSolution', gridLocation="Vertex", parent=zone)
  I.newDataArray("sphere"  , CX**2 + CY**2 + CZ**2 - 1, parent=FS)
  I.newDataArray("cylinder", CX**2 + CY**2         - 1, parent=FS)
  # Fields cell centered
  FS  = I.newFlowSolution('FlowSolutionCC', gridLocation="CellCenter", parent=zone)
  I.newDataArray("sphere"  , ccx**2 + ccy**2 + ccz**2 - 1, parent=FS)
  I.newDataArray("cylinder", ccx**2 + ccy**2          - 1, parent=FS)
# ---------------------------------------------------------------------------------------
  
# ---------------------------------------------------------------------------------------
def plane_eq(x,y,z) :
  plane_eq1 = [ 1.,  1.,  1., 2.]
  plane_eq2 = [-1., -1., -1., 2.]

  behind_plane1 = x*plane_eq1[0] \
                + y*plane_eq1[1] \
                + z*plane_eq1[2] \
                -   plane_eq1[3] < 0.
  behind_plane2 = x*plane_eq2[0] \
                + y*plane_eq2[1] \
                + z*plane_eq2[2] \
                -   plane_eq2[3] < 0.

  between_planes = np.logical_and(behind_plane1, behind_plane2)
  
  return between_planes
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def sphere_eq(x,y,z) :
  seq1 = [-0.75, -0.75, -0.75, 1. ]
  seq2 = [ 0.75, -0.75, -0.75, 1. ]
  seq3 = [ 0.  ,  0.75, -0.75, 1. ]
  seq4 = [ 0.  ,  0.  ,  0.75, 1. ]
  seq5 = [ 0.  ,  0.  ,  0.  , 3.5]

  in_sphere1  = (x-seq1[0])**2 + (y-seq1[1])**2 + (z-seq1[2])**2 - seq1[3]**2 < 0.
  in_sphere2  = (x-seq2[0])**2 + (y-seq2[1])**2 + (z-seq2[2])**2 - seq2[3]**2 < 0.
  in_sphere3  = (x-seq3[0])**2 + (y-seq3[1])**2 + (z-seq3[2])**2 - seq3[3]**2 < 0.
  in_sphere4  = (x-seq4[0])**2 + (y-seq4[1])**2 + (z-seq4[2])**2 - seq4[3]**2 < 0.
  out_sphere5 = (x-seq5[0])**2 + (y-seq5[1])**2 + (z-seq5[2])**2 - seq5[3]**2 > 0.

  gather = np.logical_or(in_sphere1, in_sphere2)
  gather = np.logical_or(gather    , in_sphere3)
  gather = np.logical_or(gather    , in_sphere4)
  gather = np.logical_or(gather    ,out_sphere5)

  return gather
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def initialize_zsr_eq(zone, variables, function):

  # In/out selection array
  in_extract_part = function(*variables)

  # Get loc zsr
  extract_lnum = np.where(in_extract_part)[0]
  extract_lnum = extract_lnum.astype(np.int32)
  
  I.newZoneSubRegion("ZSR", pointList=extract_lnum, gridLocation='CellCenter', parent=zone)
# -----------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def initialize_zsr(zone,bbox):
  cell_center = GEO.compute_cell_center(zone)
  n_cell      = cell_center.shape[0]//3

  extract_lnum = np.zeros(n_cell, dtype='int32')
  n_select_cell = 0
  for i_cell in range(n_cell):
    inside = 1

    for i_dim in range(3):
      if(cell_center[3*i_cell+i_dim] > bbox[i_dim+3] or cell_center[3*i_cell+i_dim] < bbox[i_dim]):
        inside = 0

    if(inside):
      extract_lnum[n_select_cell] = i_cell
      n_select_cell += 1

  extract_lnum.resize(n_select_cell)
  I.newZoneSubRegion("ZSR", pointList=extract_lnum, gridLocation='CellCenter', parent=zone)
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# =======================================================================================






# ========================================================================================
# ----------------------------------------------------------------------------------------
@mark_mpi_test([1, 2, 3])
def test_isosurf_U(sub_comm, write_output):

  # --- PARAMS ----------------------------------------------------------------------------
  n_vtx = 10
  h     = 5./(n_vtx-1)
  # ---------------------------------------------------------------------------------------

  # --- CUBE GEN AND PART -----------------------------------------------------------------
  dist_tree_target = DCG.dcube_generate(n_vtx, 5., [-2.5, -2.5, -2.5], sub_comm)

  zone_to_parts    = PPA.compute_regular_weights(dist_tree_target, sub_comm, 1)
  part_tree_target = PPA.partition_dist_tree(dist_tree_target, sub_comm,
                                             zone_to_parts=zone_to_parts,
                                             preserve_orientation=True,
                                             graph_part_tool="hilbert")
  # ---------------------------------------------------------------------------------------

  # --- INIT ZSR FIELDS -------------------------------------------------------------------
  for zone in I.getZones(part_tree_target):
    initialize_fld(zone)

    cell_center = GEO.compute_cell_center(zone)
    ccx = cell_center[0::3]
    ccy = cell_center[1::3]
    ccz = cell_center[2::3]
    initialize_zsr_eq(  zone, [ccx,ccy,ccz], sphere_eq)
    # initialize_zsr_eq(  zone, [ccx,ccy,ccz], plane_eq)
  # ---------------------------------------------------------------------------------------

  # --- EXTRACT PART ----------------------------------------------------------------------
  part_tree_extract = EXP.extract_part( part_tree_target, "ZSR", sub_comm,
                                        equilibrate=1,
                                        exchange=['FlowSolution','FlowSolutionCC'])
  # ---------------------------------------------------------------------------------------

  # --- SORTIES FICHIER -------------------------------------------------------------------
  if write_output:
    out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    # dist_tree = maia.factory.recover_dist_tree(part_tree,sub_comm)
    DTF(dist_tree_target, os.path.join(out_dir, 'dist_volume.cgns'), sub_comm)
    C.convertPyTree2File(part_tree_extract, os.path.join(out_dir, f'extract_part_{sub_comm.Get_rank()}.cgns'))
    # DTF(dist_tree_iso, os.path.join(out_dir, 'dist_extract_part.cgns'), sub_comm)
  # ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# =======================================================================================