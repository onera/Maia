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
from cmaia.geometry.wall_distance import compute_point_list_vertex_bc_u
from cmaia.geometry.wall_distance import compute_point_list_vertex_bc_s
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
def compute_gnum_from_parent_gnum(bcs_ln_to_gn):
  gnum = PDM.GlobalNumbering(3, # Dimension
                             1, # n_part
                             0, # Merge
                             0.,
                             comm)
  gnum.gnum_set_from_parent(0, bcs_ln_to_gn.shape[0], bcs_ln_to_gn)
  gnum.gnum_compute()
  return gnum.gnum_get(0)['gnum']

# ------------------------------------------------------------------------
def extract_surf_from_bc(part_tree, families, comm=MPI.COMM_WORLD):

  # Count all vertex
  zone_nodes = I.getNodesFromType(part_tree, 'Zone_t')
  n_vtx = sum([SIDS.Zone.n_vtx(zone_node) for zone_node in zone_nodes])
  LOG.info(f"extract_surf_from_bc [1]: n_vtx = {n_vtx}")

  # 1. Count all Boundary vertex/face
  # =================================
  n_vtx_bcs        = {}
  n_face_bcs       = {}
  n_face_vtx_bcs   = {}
  point_list_vtxs  = {}
  point_list_faces = {}
  marked_vtxs      = {}

  for zone_node in zone_nodes:
    zone_path = I.getPath(part_tree, zone_node)
    n_vtx_zone = SIDS.Zone.n_vtx(zone_node)
    LOG.info(f"extract_surf_from_bc [1]: n_vtx_zone = {n_vtx_zone}")
    marked_vtx = np.empty(n_vtx_zone, dtype=np.int32, order='F')
    marked_vtx.fill(-1)
    marked_vtxs[zone_path] = marked_vtx

    # Parse filtered bc
    if SIDS.Zone.Type(zone_node) == 'Structured':
      vtx_size = SIDS.Zone.VertexSize(zone_node)
      for bc_node in SIDS.Zone.getBCsFromFamily(zone_node, families):
        bc_type = I.getValue(bc_node)
        LOG.info(f"extract_surf_from_bc [1]: Treat bc [S]: {I.getName(bc_node)}, {bc_type}")

        # Get PointRange
        point_range_node = IE.getIndexRange(bc_node)
        point_range      = I.getVal(point_range_node)

        # Convert PointRange -> PointList
        input_loc = sids.GridLocation(bc_node)
        bnd_axis  = CSU.guess_bnd_normal_index(point_range, input_loc)
        shift     = CSU.normal_index_shift(point_range, vtx_size, bnd_axis, input_loc, "FaceCenter")
        # Prepare sub pointRanges from slabs
        sub_pr_list = [point_range]
        for sub_pr in sub_pr_list:
          sub_pr[bnd_axis,:] += shift
        point_list = CSU.compute_pointList_from_pointRanges(sub_pr_list, vtx_size, "FaceCenter", bnd_axis)

        n_vtx_bc, n_face_vtx_bc = prepare_extract_bc_s(vtx_size, point_range, marked_vtx)

        # Cumulate counters for each bc
        bc_path = I.getPath(part_tree, bc_node)
        n_vtx_bcs[bc_path]      = n_vtx_bc
        n_face_vtx_bcs[bc_path] = n_face_vtx_bc
        n_face_bcs[bc_path]     = SIDS.PointRange.n_face(point_range_node)
        # Register bc point list(s) for face
        point_list_faces[bc_path] = point_list
        LOG.info(f"extract_surf_from_bc [1]: n_vtx_bc [S]={n_vtx_bc}, n_face_bc [S]={n_face_bcs[bc_path]}, n_face_vtx_bc [S]={n_face_vtx_bc}")
    else: # SIDS.Zone.Type(zone_node) == "Unstructured":
      element_node = IE.getChildFromLabel1(zone_node, CGL.Elements_t.name)
      # NGon elements
      if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
        face_vtx, face_vtx_idx, _ = SIDS.face_connectivity(zone_node)
        LOG.info(f"extract_surf_from_bc: face_vtx [{face_vtx.shape[0]}] = {face_vtx}")
        LOG.info(f"extract_surf_from_bc: face_vtx_idx [{face_vtx_idx.shape[0]}] = {face_vtx_idx}")
        for bc_node in SIDS.Zone.getBCsFromFamily(zone_node, families):
          bc_type = I.getValue(bc_node)
          LOG.info(f"extract_surf_from_bc [1]: Treat bc [U]: {I.getName(bc_node)}, {bc_type}")

          # Get PointList
          point_list_node = IE.getIndexArray(bc_node)
          point_list      = I.getVal(point_list_node)

          n_vtx_bc, n_face_vtx_bc = prepare_extract_bc_u(point_list, face_vtx, face_vtx_idx, marked_vtx)

          # Cumulate counters for each bc
          bc_path = I.getPath(part_tree, bc_node)
          n_vtx_bcs[bc_path]      = n_vtx_bc
          n_face_vtx_bcs[bc_path] = n_face_vtx_bc
          n_face_bcs[bc_path]     = SIDS.PointList.n_face(point_list_node)
          # Register bc point list(s) for face
          point_list_faces[bc_path] = point_list
          LOG.info(f"extract_surf_from_bc [1]: n_vtx_bc [U]={n_vtx_bc}, n_face_bc [U]={n_face_bcs[bc_path]}, n_face_vtx_bc [U]={n_face_vtx_bc}")
      else:
        raise ERR.NotImplementedForElementError(zone_node, element_node)

  LOG.info(f"extract_surf_from_bc [1]: n_vtx_bcs  = {n_vtx_bcs}")
  LOG.info(f"extract_surf_from_bc [1]: n_face_bcs = {n_face_bcs}")
  n_vtx_bcs_t      = sum(n_vtx_bcs.values())
  n_face_bcs_t     = sum(n_face_bcs.values())
  n_face_vtx_bcs_t = sum(n_face_vtx_bcs.values())
  LOG.info(f"extract_surf_from_bc [1]: n_vtx_bcs_t      = {n_vtx_bcs_t}")
  LOG.info(f"extract_surf_from_bc [1]: n_face_bcs_t     = {n_face_bcs_t}")
  LOG.info(f"extract_surf_from_bc [1]: n_face_vtx_bcs_t = {n_face_vtx_bcs_t}")
  LOG.info(f"extract_surf_from_bc [1]: point_list_faces = {point_list_faces}")
  LOG.info(f"extract_surf_from_bc [1]: end\n\n")

  # 2. Compute vertex point list
  # ============================
  # Parse zone
  for zone_node in zone_nodes:
    zone_path = I.getPath(part_tree, zone_node)
    n_vtx_zone = SIDS.Zone.n_vtx(zone_node)
    LOG.info(f"extract_surf_from_bc [2]: n_vtx_zone = {n_vtx_zone}")
    marked_vtx = marked_vtxs[zone_path]
    marked_vtx.fill(-1)

    # Parse filtered bc
    if SIDS.Zone.Type(zone_node) == 'Structured':
      vtx_size = SIDS.Zone.VertexSize(zone_node)
      for bc_node in SIDS.Zone.getBCsFromFamily(zone_node, families):
        bc_type = I.getValue(bc_node)
        LOG.info(f"extract_surf_from_bc [2]: Treat bc [S]: {I.getName(bc_node)}, {bc_type}")

        # Get PointRange
        point_range_node = IE.getIndexRange(bc_node)
        point_range      = I.getVal(point_range_node)

        bc_path = I.getPath(part_tree, bc_node)
        n_vtx_bc = n_vtx_bcs[bc_path]
        point_list_vtx = compute_vertex_point_list_s(n_vtx_bc, vtx_size, point_range, marked_vtx)

        # Register bc point list(s) for vertex
        point_list_vtxs[bc_path] = np.reshape(point_list_vtx, (1, point_list_vtx.shape[0]),)
        LOG.info(f"extract_surf_from_bc [2]: point_list_vtx [S]={point_list_vtx}")
    else: # SIDS.Zone.Type(zone_node) == "Unstructured":
      element_node = IE.getChildFromLabel1(zone_node, CGL.Elements_t.name)
      # NGon elements
      if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
        face_vtx, face_vtx_idx, _ = SIDS.face_connectivity(zone_node)
        for bc_node in SIDS.Zone.getBCsFromFamily(zone_node, families):
          bc_type = I.getValue(bc_node)
          LOG.info(f"extract_surf_from_bc [2]: Treat bc [U]: {I.getName(bc_node)}, {bc_type}")

          # Get PointList
          point_list_node = IE.getIndexArray(bc_node)
          point_list      = I.getVal(point_list_node)

          bc_path = I.getPath(part_tree, bc_node)
          n_vtx_bc = n_vtx_bcs[bc_path]
          point_list_vtx = compute_point_list_vertex_bc_u(n_vtx_bc, point_list, face_vtx, face_vtx_idx, marked_vtx)

          # Register bc point list(s) for vertex
          point_list_vtxs[bc_path] = np.reshape(point_list_vtx, (1, point_list_vtx.shape[0]),)
          LOG.info(f"extract_surf_from_bc [2]: point_list_vtx [U]={point_list_vtx}")
      else:
        raise ERR.NotImplementedForElementError(zone_node, element_node)
  LOG.info(f"extract_surf_from_bc [2]: point_list_vtxs [2] = {point_list_vtxs}")
  LOG.info(f"extract_surf_from_bc [1]: end\n\n")

  # 3. Prepare the connectivity
  # ===========================
  vtx_bcs          = np.empty(3*n_vtx_bcs_t,    order='F', dtype=np.float64)
  face_vtx_bcs_idx = np.zeros(n_face_bcs_t+1,   order='F', dtype=np.int32)
  face_vtx_bcs     = np.empty(n_face_vtx_bcs_t, order='F', dtype=np.int32)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  LOG.info(f"extract_surf_from_bc: ### -> extract_surf_from_bc --> Prepare the connectivity")
  LOG.info(f"extract_surf_from_bc: ### n_face_bcs_t           : {n_face_bcs_t}")
  LOG.info(f"extract_surf_from_bc: ### n_face_vtx_bcs_t       : {n_face_vtx_bcs_t}")
  LOG.info(f"extract_surf_from_bc: ### n_vtx_bcs_t            : {n_vtx_bcs_t}")
  LOG.info(f"extract_surf_from_bc: ### face_vtx_bn.shape      : {face_vtx_bcs.shape}")
  LOG.info(f"extract_surf_from_bc: ### face_vtx_bcs_idx.shape : {face_vtx_bcs_idx.shape}\n")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # 3. Compute the connectivity
  # ===========================
  vtx_bcs_ln_to_gns  = []
  face_bcs_ln_to_gns = []
  i_vtx_bcs         = 0
  ibeg_face_vtx_idx = 0
  for zone_node in zone_nodes:
    zone_path = I.getPath(part_tree, zone_node)
    n_vtx_zone = SIDS.Zone.n_vtx(zone_node)
    LOG.info(f"extract_surf_from_bc: n_vtx_zone = {n_vtx_zone}")
    marked_vtx = marked_vtxs[zone_path]
    marked_vtx.fill(-1)

    vtx_ln_to_gn_zone, _, face_ln_to_gn_zone = SIDS.Zone.get_ln_to_gn(zone_node)
    assert(n_vtx_zone == vtx_ln_to_gn_zone.shape[0])
    LOG.info(f"extract_surf_from_bc: vtx_ln_to_gn_zone.shape[0] = {vtx_ln_to_gn_zone.shape[0]}")

    # Get coordinates
    cx, cy, cz = SIDS.coordinates(zone_node)

    # Parse filtered bc
    point_list_vtxs_zone  = []
    point_list_faces_zone = []
    if SIDS.Zone.Type(zone_node) == 'Structured':
      vtx_size = SIDS.Zone.VertexSize(zone_node)
      for bc_node in SIDS.Zone.getBCsFromFamily(zone_node, families):
        bc_type = I.getValue(bc_node)
        LOG.info(f"extract_surf_from_bc: Treat bc : {I.getName(bc_node)}, {bc_type}")
        # Get PointRange
        point_range_node = IE.getIndexRange(bc_node)
        point_range      = I.getVal(point_range_node)

        # Fill face_vtx_bcs, face_vtx_bcs_idx and vtx_bcs
        i_vtx_bcs = compute_extract_bc_s(ibeg_face_vtx_idx,
                                         vtx_size, point_range,
                                         marked_vtx,
                                         cx, cy, cz,
                                         i_vtx_bcs,
                                         face_vtx_bcs, face_vtx_bcs_idx,
                                         vtx_bcs)

        bc_path = I.getPath(part_tree, bc_node)
        # assert(n_face_bcs[bc_path] == point_list.shape[0])
        ibeg_face_vtx_idx += n_face_bcs[bc_path]
        point_list_vtxs_zone.append(point_list_vtxs[bc_path])
        point_list_faces_zone.append(point_list_faces[bc_path])
    else: # SIDS.Zone.Type(zone_node) == "Unstructured":
      element_node = IE.getChildFromLabel1(zone_node, CGL.Elements_t.name)
      # NGon elements
      if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
        face_vtx, face_vtx_idx, _ = SIDS.face_connectivity(zone_node)
        for bc_node in SIDS.Zone.getBCsFromFamily(zone_node, families):
          bc_type = I.getValue(bc_node)
          LOG.info(f"extract_surf_from_bc: Treat bc : {I.getName(bc_node)}, {bc_type}")
          # Get PointList
          point_list_node = IE.getIndexArray(bc_node)
          point_list      = I.getVal(point_list_node)

          # Fill face_vtx_bcs, face_vtx_bcs_idx and vtx_bcs
          i_vtx_bcs = compute_extract_bc_u(ibeg_face_vtx_idx,
                                           point_list,
                                           face_vtx, face_vtx_idx,
                                           marked_vtx,
                                           cx, cy, cz,
                                           i_vtx_bcs,
                                           face_vtx_bcs, face_vtx_bcs_idx,
                                           vtx_bcs)

          bc_path = I.getPath(part_tree, bc_node)
          # assert(n_face_bcs[bc_path] == point_list.shape[1])
          ibeg_face_vtx_idx += n_face_bcs[bc_path]
          point_list_vtxs_zone.append(point_list_vtxs[bc_path])
          point_list_faces_zone.append(point_list_faces[bc_path])
      else:
        raise ERR.NotImplementedForElementError(zone_node, element_node)

    # Concatenate global numbering for all vertex presents in merged bc(s) for one partition
    LOG.info(f"vtx_ln_to_gn_zone [{vtx_ln_to_gn_zone.shape}] = {vtx_ln_to_gn_zone}")
    LOG.info(f"point_list_vtxs_zone = {point_list_vtxs_zone}")
    merge_pl_idx_zone, merge_pl_zone = py_utils.concatenate_point_list(point_list_vtxs_zone)
    LOG.info(f"merge_pl_idx_zone.shape[0] = {merge_pl_idx_zone.shape[0]}")
    vtx_bcs_ln_to_gn_zone = EX.extract_from_indices(vtx_ln_to_gn_zone, merge_pl_zone, 1, 1)
    vtx_bcs_ln_to_gns.append(vtx_bcs_ln_to_gn_zone)

    # Concatenate global numbering for all face presents in merged bc(s) for one partition
    LOG.info(f"face_ln_to_gn_zone [{face_ln_to_gn_zone.shape}] = {face_ln_to_gn_zone}")
    LOG.info(f"point_list_faces_zone = {point_list_faces_zone}")
    merge_pl_idx_zone, merge_pl_zone = py_utils.concatenate_point_list(point_list_faces_zone)
    LOG.info(f"merge_pl_idx_zone.shape[0] = {merge_pl_idx_zone.shape[0]}")
    face_bcs_ln_to_gn_zone = EX.extract_from_indices(face_ln_to_gn_zone, merge_pl_zone, 1, 1)
    face_bcs_ln_to_gns.append(face_bcs_ln_to_gn_zone)

  # Concatenate global numbering for all vertex presents in merged bc(s) for all partition
  LOG.info(f"vtx_bcs_ln_to_gns = {vtx_bcs_ln_to_gns}")
  vtx_bcs_ln_to_gn = py_utils.concatenate_numpy(vtx_bcs_ln_to_gns)
  LOG.info(f"vtx_bcs_ln_to_gn [{vtx_bcs_ln_to_gn.shape[0]}] = {np.sort(vtx_bcs_ln_to_gn)}")
  # Concatenate global numbering for all face presents in merged bc(s) for all partition
  LOG.info(f"face_bcs_ln_to_gns = {face_bcs_ln_to_gns}")
  face_bcs_ln_to_gn = py_utils.concatenate_numpy(face_bcs_ln_to_gns)
  LOG.info(f"face_bcs_ln_to_gn [{face_bcs_ln_to_gn.shape[0]}] = {np.sort(face_bcs_ln_to_gn)}")

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  LOG.info("extract_surf_from_bc: -> extract_surf_from_bc --> Compute the connectivity")
  LOG.info(f"extract_surf_from_bc: i_vtx_bcs         = {i_vtx_bcs}")
  LOG.info(f"extract_surf_from_bc: n_vtx_bcs_t       = {n_vtx_bcs_t}")
  LOG.info(f"extract_surf_from_bc: n_face_bcs_t      = {n_face_bcs_t}")
  LOG.info(f"extract_surf_from_bc: ibeg_face_vtx_idx = {ibeg_face_vtx_idx}")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  assert(i_vtx_bcs == n_vtx_bcs_t)
  assert(n_face_bcs_t == ibeg_face_vtx_idx)

  # 4. Compute the global numbering for Vertex/Face
  # ===============================================
  # shift_vtx_ln_to_gn = comm.scan(n_vtx_bcs_t , op=MPI.SUM) - n_vtx_bcs_t
  # # Shift to global
  # vtx_ln_to_gn = np.linspace(shift_vtx_ln_to_gn +1, shift_vtx_ln_to_gn +n_vtx_bcs_t , num=n_vtx_bcs_t , dtype=pdm_dtype)

  # shift_face_ln_to_gn = comm.scan(n_face_bcs_t, op=MPI.SUM) - n_face_bcs_t
  # # Shift to global
  # face_ln_to_gn = np.linspace(shift_face_ln_to_gn+1, shift_face_ln_to_gn+n_face_bcs_t, num=n_face_bcs_t, dtype=pdm_dtype)

  vtx_ln_to_gn  = compute_gnum_from_parent_gnum(vtx_bcs_ln_to_gn)
  face_ln_to_gn = compute_gnum_from_parent_gnum(face_bcs_ln_to_gn)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Panic verbose
  # LOG.info(f"extract_surf_from_bc: shift_face_ln_to_gn : {shift_face_ln_to_gn}")
  # LOG.info(f"extract_surf_from_bc: shift_vtx_ln_to_gn  : {shift_vtx_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: n_face_bcs_t                                : {n_face_bcs_t}")
  LOG.info(f"extract_surf_from_bc: n_vtx_bcs_t                                 : {n_vtx_bcs_t}")
  LOG.info(f"extract_surf_from_bc: face_ln_to_gn [{face_ln_to_gn.shape}]       : {face_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: sort(face_ln_to_gn)                         : {np.sort(face_ln_to_gn)}")
  LOG.info(f"extract_surf_from_bc: vtx_ln_to_gn  [{vtx_ln_to_gn.shape}]        : {vtx_ln_to_gn}")
  LOG.info(f"extract_surf_from_bc: sort(vtx_ln_to_gn)  [{vtx_ln_to_gn.shape}]  : {np.sort(vtx_ln_to_gn)}")
  LOG.info(f"extract_surf_from_bc: face_vtx_bcs  [{face_vtx_bcs.shape}]        : {face_vtx_bcs}")
  LOG.info(f"extract_surf_from_bc: face_vtx_bcs_idx [{face_vtx_bcs_idx.shape}] : {face_vtx_bcs_idx}")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  return face_vtx_bcs, face_vtx_bcs_idx, face_ln_to_gn, vtx_bcs, vtx_ln_to_gn

if __name__ == "__main__":
  # t = C.convertFile2PyTree("cubeS_join_bnd.hdf")
  # t = C.convertFile2PyTree("cubeU_join_bnd.hdf")
  # I._adaptNGon12NGon2(t)
  t = C.convertFile2PyTree("cubeU_join_bnd-new.hdf")
  I.printTree(t)

  families = ['Wall']

  extract_surf_from_bc(t, families)
