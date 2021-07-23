import numpy as np
from   mpi4py import MPI
import logging as LOG

import Converter.PyTree   as C
import Converter.Internal as I

from maia import npy_pdm_gnum_dtype as pdm_dtype
import maia.sids.Internal_ext as IE
from maia.sids import sids as SIDS

import maia.sids.cgns_keywords as CGK
from maia.sids.cgns_keywords import Label as CGL

from cmaia.geometry.wall_distance import prepare_extract_bc_u
from cmaia.geometry.wall_distance import prepare_extract_bc_s
from cmaia.geometry.wall_distance import compute_extract_bc_u
from cmaia.geometry.wall_distance import compute_extract_bc_s

# ------------------------------------------------------------------------
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

fmt = f"{mpi_rank}:{mpi_size}"
LOG.basicConfig(filename = f"maia_workflow_log.{mpi_rank}.log",
                level    = 10,
                format   = fmt,
                filemode = 'w')

# ------------------------------------------------------------------------
def extract_surf_from_bc(part_tree, families, comm = MPI.COMM_WORLD):

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
        print(f"Treat bc : {I.getName(bc_node)}, {bctype}")

        point_range_node = I.getNodeFromName1(bc_node, 'PointRange')
        point_range      = I.getVal(point_range_node)

        n_face_bnd = SIDS.PointRange.n_face(point_range_node)
        print(f"S: n_face_bnd={n_face_bnd}")
        n_face_vtx_bnd, n_vtx_bnd = prepare_extract_bc_s(vtx_size, point_range, work_vtx)
        print(f"S: n_face_vtx_bnd={n_face_vtx_bnd}, n_vtx_bnd={n_vtx_bnd}")
        # Cumulate counters for each bc
        n_face_bnd_t     += n_face_bnd
        n_face_vtx_bnd_t += n_face_vtx_bnd
        n_vtx_bnd_t      += n_vtx_bnd
    elif SIDS.Zone.Type(part_zone) == "Unstructured":
      element_node = I.getNodeFromType1(part_zone, CGL.Elements_t.name)
      # NGon elements
      if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
        face_vtx, face_vtx_idx, _ = SIDS.face_connectivity(part_zone)
        for bc_node in SIDS.Zone.getBCsFromFamily(part_zone, families):
          bctype = I.getValue(bc_node)
          print(f"Treat bc : {I.getName(bc_node)}, {bctype}")

          point_list_node = I.getNodeFromName1(bc_node, 'PointList')
          point_list      = I.getVal(point_list_node)

          n_face_bnd    = SIDS.PointList.n_face(point_list_node)
          print(f"U: n_face_bnd={n_face_bnd}, n_face_bnd_t={n_face_bnd_t}")
          n_face_vtx_bnd, n_vtx_bnd = prepare_extract_bc_u(point_list, face_vtx, face_vtx_idx, work_vtx)
          print(f"U: n_face_vtx_bnd={n_face_vtx_bnd}, n_vtx_bnd={n_vtx_bnd}")
          # Cumulate counters for each bc
          n_face_bnd_t     += n_face_bnd
          n_face_vtx_bnd_t += n_face_vtx_bnd
          n_vtx_bnd_t      += n_vtx_bnd
      else:
        raise NotImplementedError(f"Unstructured Zone {I.getName(part_zone)} with {SIDS.ElementCGNSName(element_node)} not yet implemented.")
    else:
      raise TypeError(f"Unable to determine the ZoneType for Zone {I.getName(part_zone)}")

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
  # print(" ### face_vtx_bnd      : ", face_vtx_bnd)
  # print(" ### face_vtx_bnd_idx  : ", face_vtx_bnd_idx)
  LOG.info(f"extract_surf_from_bc: ### n_face_bnd_t      : {n_face_bnd_t}")
  LOG.info(f"extract_surf_from_bc: ### n_face_vtx_bnd_t  : {n_face_vtx_bnd_t}")
  LOG.info(f"extract_surf_from_bc: ### n_vtx_bnd_t       : {n_vtx_bnd_t}")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # 3. Compute the connectivity
  # ===========================
  # print " ### -> extract_surf_from_bc --> Compute the connectivity"
  ibeg_face_vtx_idx = 0
  i_vtx_bnd         = 0
  for part_zone in part_zones:
    # Get coordinates
    cx, cy, cz = SIDS.coordinates(part_zone)

    # n_vtx = SIDS.Zone.n_vtx(part_zone)
    # print(f"n_vtx = {n_vtx}")
    # work_vtx = np.empty(n_vtx, dtype=np.int32, order='F')
    work_vtx.fill(-1)

    # Parse filtered bc
    if SIDS.Zone.Type(part_zone) == 'Structured':
      vtx_size = SIDS.Zone.VertexSize(part_zone)
      for bc_node in SIDS.Zone.getBCsFromFamily(part_zone, families):
        bctype = I.getValue(bc_node)
        LOG.info(f"extract_surf_from_bc: Treat bc : {I.getName(bc_node)}, {bctype}")

        point_range_node = I.getNodeFromName1(bc_node, 'PointRange')
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
    elif SIDS.Zone.Type(part_zone) == "Unstructured":
      element_node = I.getNodeFromType1(part_zone, CGL.Elements_t.name)
      # NGon elements
      if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
        face_vtx, face_vtx_idx, _ = SIDS.face_connectivity(part_zone)
        for bc_node in SIDS.Zone.getBCsFromFamily(part_zone, families):
          bctype = I.getValue(bc_node)
          LOG.info(f"extract_surf_from_bc: Treat bc : {I.getName(bc_node)}, {bctype}")

          point_list_node = I.getNodeFromName1(bc_node, 'PointList')
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
        raise NotImplementedError(f"Unstructured Zone {I.getName(part_zone)} with {SIDS.ElementCGNSName(element_node)} not yet implemented.")
    else:
      raise TypeError(f"Unable to determine the ZoneType for Zone {I.getName(part_zone)}")

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

  shift_face_ln_to_gn = comm.scan(n_face_bnd_t, op=MPI.SUM) - n_face_bnd_t
  shift_vtx_ln_to_gn  = comm.scan(n_vtx_bnd_t   , op=MPI.SUM) - n_vtx_bnd_t

  # Shift to global
  face_ln_to_gn = np.linspace(shift_face_ln_to_gn+1, shift_face_ln_to_gn+n_face_bnd_t, num=n_face_bnd_t, dtype=pdm_dtype)
  vtx_ln_to_gn  = np.linspace(shift_vtx_ln_to_gn +1, shift_vtx_ln_to_gn +n_vtx_bnd_t , num=n_vtx_bnd_t , dtype=pdm_dtype)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  # debug = file('debug_{0}'.format(comm.Get_rank()), 'w')
  LOG.info(f"extract_surf_from_bc: shift_face_ln_to_gn : {shift_face_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: shift_vtx_ln_to_gn  : {shift_vtx_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: n_face_bnd_t        : {n_face_bnd_t}")
  LOG.info(f"extract_surf_from_bc: n_vtx_bnd_t         : {n_vtx_bnd_t}")
  LOG.info(f"extract_surf_from_bc: face_ln_to_gn       : {face_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: vtx_ln_to_gn        : {vtx_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: face_vtx_bnd        : {face_vtx_bnd}")
  LOG.info(f"extract_surf_from_bc: face_vtx_bnd_idx    : {face_vtx_bnd_idx}")
  # print "face_vtx_bnd     : ", face_vtx_bnd
  # print "face_vtx_bnd_idx : ", face_vtx_bnd_idx
  # print "vtx_bnd          : ", vtx_bnd
  # print "face_ln_to_gn    : ", face_ln_to_gn
  # print "vtx_ln_to_gn     : ", vtx_ln_to_gn
  # print " ### -> extract_surf_from_bc end "
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
