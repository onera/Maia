import numpy   as np
import logging as LOG
from mpi4py import MPI

import Converter.PyTree   as C
import Converter.Internal as I

import Pypdm.Pypdm as PDM

import maia.sids.cgns_keywords as CGK
import maia.sids.Internal_ext  as IE
from maia                    import npy_pdm_gnum_dtype as pdm_dtype
from maia.sids.cgns_keywords import Label              as CGL
from maia.sids               import sids               as SIDS
from maia.sids               import conventions        as conv
from maia.utils              import py_utils

from cmaia.geometry.wall_distance import prepare_extract_bc_u
from cmaia.geometry.wall_distance import prepare_extract_bc_s
from cmaia.geometry.wall_distance import compute_extract_bc_u
from cmaia.geometry.wall_distance import compute_extract_bc_s
import cmaia.utils.extract_from_indices as EX

# ------------------------------------------------------------------------
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

fmt = f'%(levelname)s[{mpi_rank}/{mpi_size}]:%(message)s '
LOG.basicConfig(filename = f"maia_workflow_log.{mpi_rank}.log",
                level    = 10,
                format   = fmt,
                filemode = 'w')

# ------------------------------------------------------------------------
def extract_surf_from_bc(skeleton_tree, part_tree, families, comm=MPI.COMM_WORLD):

  # 1. Count all Boundary Vtx/Face
  # ==============================
  n_face_bnd_t     = 0
  n_face_vtx_bnd_t = 0
  n_vtx_bnd_t      = 0

  # Parse zone
  part_zones = I.getNodesFromType(part_tree, 'Zone_t')
  for part_zone in part_zones:
    n_vtx = SIDS.Zone.n_vtx(part_zone)
    LOG.info(f"extract_surf_from_bc: n_vtx = {n_vtx}")
    work_vtx = np.empty(n_vtx, dtype=np.int32, order='F')
    work_vtx.fill(-1)

    # Parse filtered bc
    if SIDS.Zone.Type(part_zone) == 'Structured':
      vtx_size = SIDS.Zone.VertexSize(part_zone)
      for bc_node in SIDS.Zone.getBCsFromFamily(part_zone, families):
        bctype = I.getValue(bc_node)
        LOG.info(f"extract_surf_from_bc: Treat bc [S]: {I.getName(bc_node)}, {bctype}")

        point_range_node = IE.requireNodeFromName1(bc_node, 'PointRange')
        point_range      = I.getVal(point_range_node)

        n_face_bnd = SIDS.PointRange.n_face(point_range_node)
        LOG.info(f"extract_surf_from_bc: n_face_bnd [S]={n_face_bnd}, n_face_bnd_t={n_face_bnd_t}")
        n_face_vtx_bnd, n_vtx_bnd = prepare_extract_bc_s(vtx_size, point_range, work_vtx)
        LOG.info(f"extract_surf_from_bc: n_face_vtx_bnd [S]={n_face_vtx_bnd}, n_vtx_bnd [S]={n_vtx_bnd}")
        # Cumulate counters for each bc
        n_face_bnd_t     += n_face_bnd
        n_face_vtx_bnd_t += n_face_vtx_bnd
        n_vtx_bnd_t      += n_vtx_bnd
    else: # SIDS.Zone.Type(part_zone) == "Unstructured":
      element_node = IE.requireNodeFromType1(part_zone, CGL.Elements_t.name)
      # NGon elements
      if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
        face_vtx, face_vtx_idx, _ = SIDS.face_connectivity(part_zone)
        for bc_node in SIDS.Zone.getBCsFromFamily(part_zone, families):
          bctype = I.getValue(bc_node)
          LOG.info(f"extract_surf_from_bc: Treat bc [U]: {I.getName(bc_node)}, {bctype}")

          point_list_node = IE.requireNodeFromName1(bc_node, 'PointList')
          point_list      = I.getVal(point_list_node)

          n_face_bnd = SIDS.PointList.n_face(point_list_node)
          LOG.info(f"extract_surf_from_bc: n_face_bnd [U]={n_face_bnd}, n_face_bnd_t={n_face_bnd_t}")
          n_face_vtx_bnd, n_vtx_bnd = prepare_extract_bc_u(point_list, face_vtx, face_vtx_idx, work_vtx)
          LOG.info(f"extract_surf_from_bc: n_face_vtx_bnd [U]={n_face_vtx_bnd}, n_vtx_bnd [U]={n_vtx_bnd}")
          # Cumulate counters for each bc
          n_face_bnd_t     += n_face_bnd
          n_face_vtx_bnd_t += n_face_vtx_bnd
          n_vtx_bnd_t      += n_vtx_bnd
      else:
        raise ERR.NotImplementedForElementError(part_zone, element_node)
  LOG.info(f"extract_surf_from_bc: n_face_bnd_t={n_face_bnd_t}")
  LOG.info(f"extract_surf_from_bc: n_face_vtx_bnd_t={n_face_vtx_bnd_t}, n_vtx_bnd_t={n_vtx_bnd_t}, n_face_bnd_t={n_face_bnd_t}")

  # 2. Prepare the connectivity
  # ===========================
  face_vtx_bnd     = np.empty(n_face_vtx_bnd_t, order='F', dtype=np.int32)
  face_vtx_bnd_idx = np.zeros(n_face_bnd_t+1,   order='F', dtype=np.int32)
  vtx_bnd          = np.empty(3*n_vtx_bnd_t,    order='F', dtype=np.float64)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  LOG.info("extract_surf_from_bc: ### -> extract_surf_from_bc --> Prepare the connectivity")
  LOG.info(f"extract_surf_from_bc: ### n_face_bnd_t      : {n_face_bnd_t}")
  LOG.info(f"extract_surf_from_bc: ### n_face_vtx_bnd_t  : {n_face_vtx_bnd_t}")
  LOG.info(f"extract_surf_from_bc: ### n_vtx_bnd_t       : {n_vtx_bnd_t}")
  LOG.info(f"extract_surf_from_bc: ### face_vtx_bnd      : {face_vtx_bnd}")
  LOG.info(f"extract_surf_from_bc: ### face_vtx_bnd_idx  : {face_vtx_bnd_idx}")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # 3. Compute the connectivity
  # ===========================
  ibeg_face_vtx_idx = 0
  i_vtx_bnd         = 0
  for part_zone in part_zones:
    # Get coordinates
    cx, cy, cz = SIDS.coordinates(part_zone)

    # n_vtx = SIDS.Zone.n_vtx(part_zone)
    # work_vtx = np.empty(n_vtx, dtype=np.int32, order='F')
    work_vtx.fill(-1)

    # Parse filtered bc
    if SIDS.Zone.Type(part_zone) == 'Structured':
      vtx_size = SIDS.Zone.VertexSize(part_zone)
      for bc_node in SIDS.Zone.getBCsFromFamily(part_zone, families):
        bctype = I.getValue(bc_node)
        LOG.info(f"extract_surf_from_bc: Treat bc : {I.getName(bc_node)}, {bctype}")

        point_range_node = IE.requireNodeFromName1(bc_node, 'PointRange')
        point_range      = I.getVal(point_range_node)
        n_face_bnd = SIDS.PointRange.n_face(point_range_node)
        i_vtx_bnd = compute_extract_bc_s(ibeg_face_vtx_idx,
                                         vtx_size, point_range,
                                         work_vtx,
                                         cx, cy, cz,
                                         i_vtx_bnd,
                                         face_vtx_bnd, face_vtx_bnd_idx,
                                         vtx_bnd)
        ibeg_face_vtx_idx += n_face_bnd
    else: # SIDS.Zone.Type(part_zone) == "Unstructured":
      element_node = I.getNodeFromType1(part_zone, CGL.Elements_t.name)
      # NGon elements
      if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
        face_vtx, face_vtx_idx, _ = SIDS.face_connectivity(part_zone)
        for bc_node in SIDS.Zone.getBCsFromFamily(part_zone, families):
          bctype = I.getValue(bc_node)
          LOG.info(f"extract_surf_from_bc: Treat bc : {I.getName(bc_node)}, {bctype}")

          point_list_node = IE.requireNodeFromName1(bc_node, 'PointList')
          point_list      = I.getVal(point_list_node)
          n_face_bnd = SIDS.PointList.n_face(point_list_node)
          i_vtx_bnd = compute_extract_bc_u(ibeg_face_vtx_idx,
                                           point_list,
                                           face_vtx, face_vtx_idx,
                                           work_vtx,
                                           cx, cy, cz,
                                           i_vtx_bnd,
                                           face_vtx_bnd, face_vtx_bnd_idx,
                                           vtx_bnd)
          ibeg_face_vtx_idx += n_face_bnd
      else:
        raise ERR.NotImplementedForElementError(part_zone, element_node)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  LOG.info("extract_surf_from_bc: -> extract_surf_from_bc --> Compute the connectivity")
  LOG.info(f"extract_surf_from_bc: n_face_bnd_t      : {n_face_bnd_t}")
  LOG.info(f"extract_surf_from_bc: ibeg_face_vtx_idx : {ibeg_face_vtx_idx}")
  LOG.info(f"extract_surf_from_bc: i_vtx_bnd         : {i_vtx_bnd}")
  LOG.info(f"extract_surf_from_bc: n_vtx_bnd_t       : {n_vtx_bnd_t}")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  assert(n_face_bnd_t == ibeg_face_vtx_idx)
  assert(i_vtx_bnd == n_vtx_bnd_t)

  # 4. Compute the global numbering for Vertex
  # ==========================================
  shift_vtx_ln_to_gn = comm.scan(n_vtx_bnd_t , op=MPI.SUM) - n_vtx_bnd_t
  # Shift to global
  vtx_ln_to_gn = np.linspace(shift_vtx_ln_to_gn +1, shift_vtx_ln_to_gn +n_vtx_bnd_t , num=n_vtx_bnd_t , dtype=pdm_dtype)

  # 5. Compute the global numbering for Face
  # ========================================
  # shift_face_ln_to_gn = comm.scan(n_face_bnd_t, op=MPI.SUM) - n_face_bnd_t
  # # Shift to global
  # face_ln_to_gn = np.linspace(shift_face_ln_to_gn+1, shift_face_ln_to_gn+n_face_bnd_t, num=n_face_bnd_t, dtype=pdm_dtype)

  # Create ParaDiGM structure
  gen_gnum = PDM.GlobalNumbering(3, # Dimension
                                 1, # n_part
                                 0, # Merge
                                 0.,
                                 comm)

  bnd_ln_to_gns = []
  for i_part, part_zone in enumerate(part_zones):
    _, _, face_ln_to_gn_part_zone = SIDS.Zone.get_ln_to_gn(part_zone)

    n_face_bnd_part_zone = 0
    point_lists_part_zone = []

    # Parse filtered all bc
    if SIDS.Zone.Type(part_zone) == 'Structured':
      raise NotImplementedError()
    else: # SIDS.Zone.Type(part_zone) == "Unstructured":
      element_node = I.getNodeFromType1(part_zone, CGL.Elements_t.name)
      # NGon elements
      if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
        for bc_node in SIDS.Zone.getBCsFromFamily(part_zone, families):
          bctype = I.getValue(bc_node)
          LOG.info(f"extract_surf_from_bc: Treat bc : {I.getName(bc_node)}, {bctype}")

          point_list_node = IE.requireNodeFromName1(bc_node, 'PointList')
          point_list      = I.getVal(point_list_node)
          n_face_bnd = SIDS.PointList.n_face(point_list_node)
          n_face_bnd_part_zone += n_face_bnd

          LOG.info(f"  point_list = {point_list}")
          point_lists_part_zone.append(point_list)
      else:
        raise ERR.NotImplementedForElementError(part_zone, element_node)

    LOG.info(f"face_ln_to_gn_part_zone.shape = {face_ln_to_gn_part_zone.shape}")
    LOG.info(f"face_ln_to_gn_part_zone = {face_ln_to_gn_part_zone}")
    LOG.info(f"point_lists_part_zone = {point_lists_part_zone}")
    merge_pl_idx_part_zone, merge_pl_part_zone = py_utils.concatenate_point_list(point_lists_part_zone)
    LOG.info(f"merge_pl_idx_part_zone.shape[0] = {merge_pl_idx_part_zone.shape[0]}")

    bnd_ln_to_gn_part_zone = EX.extract_from_indices(face_ln_to_gn_part_zone, merge_pl_part_zone, 1, 1)
    # bnd_ln_to_gn_part_zone += n_face_bnd_part_zone
    bnd_ln_to_gns.append(bnd_ln_to_gn_part_zone)

  LOG.info(f"bnd_ln_to_gns = {bnd_ln_to_gns}")
  bnd_ln_to_gn = py_utils.concatenate_numpy(bnd_ln_to_gns)

  LOG.info(f"face_ln_to_gn_part_zone.shape[0] = {face_ln_to_gn_part_zone.shape[0]}")
  LOG.info(f"bnd_ln_to_gn.shape[0] = {bnd_ln_to_gn.shape[0]}")
  LOG.info(f"bnd_ln_to_gn = {bnd_ln_to_gn}")

  gen_gnum.gnum_set_from_parent(0, bnd_ln_to_gn.shape[0], bnd_ln_to_gn)

  gen_gnum.gnum_compute()

  face_ln_to_gn = gen_gnum.gnum_get(0)['gnum']

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  # LOG.info(f"extract_surf_from_bc: shift_face_ln_to_gn : {shift_face_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: shift_vtx_ln_to_gn  : {shift_vtx_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: n_face_bnd_t        : {n_face_bnd_t}")
  LOG.info(f"extract_surf_from_bc: n_vtx_bnd_t         : {n_vtx_bnd_t}")
  LOG.info(f"extract_surf_from_bc: face_ln_to_gn       : {face_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: vtx_ln_to_gn        : {vtx_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: face_vtx_bnd        : {face_vtx_bnd}")
  LOG.info(f"extract_surf_from_bc: face_vtx_bnd_idx    : {face_vtx_bnd_idx}")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  return face_vtx_bnd, face_vtx_bnd_idx, face_ln_to_gn, vtx_bnd, vtx_ln_to_gn

if __name__ == "__main__":
  # t = C.convertFile2PyTree("cubeS_join_bnd.hdf")
  # t = C.convertFile2PyTree("cubeU_join_bnd.hdf")
  # I._adaptNGon12NGon2(t)
  t = C.convertFile2PyTree("cubeU_join_bnd-new.hdf")
  I.printTree(t)

  families = ['Wall']

  extract_surf_from_bc(t, families)
