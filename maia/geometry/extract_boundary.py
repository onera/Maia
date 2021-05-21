import numpy as np
from   mpi4py import MPI

import Converter.PyTree   as C
import Converter.Internal as I

import maia.sids.Internal_ext as IE
from maia.sids import sids as SIDS

from cmaia.geometry.wall_distance import prepare_extract_bc_u
from cmaia.geometry.wall_distance import prepare_extract_bc_s
from cmaia.geometry.wall_distance import compute_extract_bc_u
from cmaia.geometry.wall_distance import compute_extract_bc_s

def extract_surf_from_bc(tree, bc_families, comm = MPI.COMM_WORLD):

  # 1. Count all Boundary Vtx/Face
  # ==============================
  n_face_bnd_t   = 0
  n_face_vtx_bnd = 0
  n_vtx_bnd      = 0

  # Parse zone
  zones = I.getNodesFromType(tree, 'Zone_t')
  for zone in zones:
    vtx_size = SIDS.Zone.VertexSize(zone)
    if SIDS.Zone.Type(zone) == "Structured":
      im, jm, km = vtx_size[:]
      print(f"im = {im}, jm = {jm}, km = {km}")
      n_vtx = np.prod(vtx_size)
    else:
      n_vtx = vtx_size
      face_vtx, face_vtx_idx, ngon_pe = SIDS.face_connectivity(zone)
    print(f"n_vtx = {n_vtx}")

    work_vtx = np.empty(n_vtx, dtype=np.int32, order='F')
    work_vtx.fill(-1)

    # Parse filtered bc
    for bc_node in SIDS.Zone.getBCsFromFamily(zone, bc_families):
      bctype = I.getValue(bc_node)
      print(f"Treat bc : {I.getName(bc_node)}, {bctype}")

      if SIDS.Zone.Type(zone) == 'Structured':
        point_range_node = I.getNodeFromName1(bc_node, 'PointRange')
        point_range      = I.getVal(point_range_node)

        n_face_bnd = SIDS.PointRange.n_face(point_range_node)
        print(f"S: n_face_bnd={n_face_bnd}, n_face_bnd_t={n_face_bnd_t}")

        in_face_vtx_bnd, in_vtx_bnd = prepare_extract_bc_s(vtx_size, point_range, work_vtx)
        print(f"S: in_face_vtx_bnd={in_face_vtx_bnd}, in_vtx_bnd={in_vtx_bnd}")
      else:
        point_list_node = I.getNodeFromName1(bc_node, 'PointList')
        point_list      = I.getVal(point_list_node)

        n_face_bnd    = SIDS.PointList.n_face(point_list_node)
        n_face_bnd_t += n_face_bnd
        print(f"U: n_face_bnd={n_face_bnd}, n_face_bnd_t={n_face_bnd_t}")

        in_face_vtx_bnd, in_vtx_bnd = prepare_extract_bc_u(point_list, face_vtx, face_vtx_idx, work_vtx)
        print(f"U: in_face_vtx_bnd={in_face_vtx_bnd}, in_vtx_bnd={in_vtx_bnd}")
      n_face_bnd_t += n_face_bnd

      # Cumulate counters for each bc
      n_face_vtx_bnd += in_face_vtx_bnd
      n_vtx_bnd      += in_vtx_bnd
      print(f"n_face_vtx_bnd={n_face_vtx_bnd}, n_vtx_bnd={n_vtx_bnd}, n_face_bnd_t={n_face_bnd_t}")

  # 2. Prepare the connectivity
  # ===========================
  face_vtx_bnd     = np.empty(n_face_vtx_bnd, order='F', dtype=np.int32  )
  face_vtx_bnd_idx = np.zeros(n_face_bnd_t+1, order='F', dtype=np.int32  )
  vtx_bnd          = np.empty(3*n_vtx_bnd   , order='F', dtype=np.float64)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  print(" ### -> extract_surf_from_bc --> Prepare the connectivity")
  # print(" ### face_vtx_bnd      : ", face_vtx_bnd)
  # print(" ### face_vtx_bnd_idx  : ", face_vtx_bnd_idx)
  print(" ### n_face_bnd_t      : ", n_face_bnd_t)
  print(" ### n_face_vtx_bnd    : ", n_face_vtx_bnd)
  print(" ### n_vtx_bnd         : ", n_vtx_bnd)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # 3. Compute the connectivity
  # ===========================
  # print " ### -> extract_surf_from_bc --> Compute the connectivity"
  ibeg_face_vtx_idx = 0
  i_vtx_bnd         = 0
  for zone in zones:
    vtx_size = SIDS.Zone.VertexSize(zone)
    if SIDS.Zone.Type(zone) == "Structured":
      n_vtx = np.prod(vtx_size)
    else:
      n_vtx = vtx_size
      face_vtx, face_vtx_idx, ngon_pe = SIDS.face_connectivity(zone)
    print(f"n_vtx = {n_vtx}")
    # Get coordinates
    x, y, z = SIDS.coordinates(zone)

    work_vtx = np.empty(n_vtx, dtype=np.int32, order='F')
    work_vtx.fill(-1)

    # Parse filtered bc
    for bc_node in SIDS.Zone.getBCsFromFamily(zone, bc_families):
      bctype = I.getValue(bc_node)
      print(f"Treat bc : {I.getName(bc_node)}, {bctype}")

      if SIDS.Zone.Type(zone) == 'Structured':
        point_range_node = I.getNodeFromName1(bc_node, 'PointRange')
        point_range      = I.getVal(point_range_node)

        n_face_bnd = SIDS.PointRange.n_face(point_range_node)

        i_vtx_bnd = compute_extract_bc_s(ibeg_face_vtx_idx,
                                         vtx_size, point_range,
                                         work_vtx,
                                         x, y, z,
                                         i_vtx_bnd,
                                         face_vtx_bnd, face_vtx_bnd_idx,
                                         vtx_bnd)
      else:
        point_list_node = I.getNodeFromName1(bc_node, 'PointList')
        point_list      = I.getVal(point_list_node)

        n_face_bnd = SIDS.PointList.n_face(point_list_node)

        i_vtx_bnd = compute_extract_bc_u(ibeg_face_vtx_idx,
                                         point_list,
                                         face_vtx, face_vtx_idx,
                                         work_vtx,
                                         x, y, z,
                                         i_vtx_bnd,
                                         face_vtx_bnd, face_vtx_bnd_idx,
                                         vtx_bnd)
      ibeg_face_vtx_idx += n_face_bnd

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  print(" ### -> extract_surf_from_bc --> Compute the connectivity")
  print(" ### n_face_bnd_t      : ", n_face_bnd_t)
  print(" ### ibeg_face_vtx_idx : ", ibeg_face_vtx_idx)
  print(" ### i_vtx_bnd         : ", i_vtx_bnd)
  print(" ### n_vtx_bnd         : ", n_vtx_bnd)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  assert(n_face_bnd_t == ibeg_face_vtx_idx)
  assert(i_vtx_bnd == n_vtx_bnd)

  shift_face_ln_to_gn = comm.scan(n_face_bnd_t, op=MPI.SUM) - n_face_bnd_t
  shift_vtx_ln_to_gn  = comm.scan(n_vtx_bnd   , op=MPI.SUM) - n_vtx_bnd

  # Shift to global
  face_ln_to_gn = np.linspace(shift_face_ln_to_gn+1, shift_face_ln_to_gn+n_face_bnd_t, num=n_face_bnd_t, dtype=np.int32)
  vtx_ln_to_gn  = np.linspace(shift_vtx_ln_to_gn +1, shift_vtx_ln_to_gn +n_vtx_bnd   , num=n_vtx_bnd   , dtype=np.int32)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  # debug = file('debug_{0}'.format(comm.Get_rank()), 'w')
  # debug.write("shift_face_ln_to_gn : {0}\n".format(shift_face_ln_to_gn))
  # debug.write("shift_vtx_ln_to_gn  : {0}\n".format(shift_vtx_ln_to_gn))
  # debug.write("n_face_bnd_t        : {0}\n".format(n_face_bnd_t))
  # debug.write("n_vtx_bnd           : {0}\n".format(n_vtx_bnd))
  # debug.write("face_ln_to_gn       : {0}\n".format(face_ln_to_gn))
  # debug.write("vtx_ln_to_gn        : {0}\n".format(vtx_ln_to_gn))
  # print "face_vtx_bnd     : ", face_vtx_bnd
  # print "face_vtx_bnd_idx : ", face_vtx_bnd_idx
  # print "vtx_bnd          : ", vtx_bnd
  # print "face_ln_to_gn    : ", face_ln_to_gn
  # print "vtx_ln_to_gn     : ", vtx_ln_to_gn
  # print " ### -> extract_surf_from_bc end "
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  return face_vtx_bnd, face_vtx_bnd_idx, face_ln_to_gn, vtx_bnd, vtx_ln_to_gn

if __name__ == "__main__":
  t = C.convertFile2PyTree("cubeS_join_bnd.hdf")
  # t = C.convertFile2PyTree("cubeU_join_bnd.hdf")
  # t = C.convertFile2PyTree("cubeH.hdf")
  I._adaptNGon12NGon2(t)
  I.printTree(t)

  bc_families = ['Wall']

  extract_surf_from_bc(t, bc_families)
