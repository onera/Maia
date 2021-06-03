from mpi4py import MPI
import Converter.Internal     as I
import maia.sids.Internal_ext as IE
import maia.sids.sids as SIDS

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

from maia.tree_exchange.part_to_dist         import discover    as disc
from maia.sids                               import conventions as conv
from maia.distribution.distribution_function import create_distribution_node_from_distrib
from maia.utils.parallel                     import utils          as par_utils
from maia.sids                               import elements_utils as EU

import maia.tree_exchange.utils              as te_utils

import maia.distribution.distribution_function as MID
import maia.sids.Internal_ext as IE

from etc.walldistance import Geometry
import cmaia.utils.extract_from_indices as EX

import numpy as np

import Pypdm.Pypdm as PDM

def rebuild_partial_dist_tree(part_base, comm):
  """
  Rebuild the a light dist_base in order to have all initiales zone AND initial BC_t
  """
  # >
  dist_tree = I.newCGNSTree()
  disc.discover_nodes_of_kind(dist_tree, [part_base], 'CGNSBase_t', comm, child_list=['Family_t'])
  disc.discover_nodes_of_kind(dist_tree, [part_base], 'CGNSBase_t/Zone_t', comm,
                              merge_rule=lambda zpath : '.'.join(zpath.split('.')[:-2]))
  # I.printTree(dist_tree)
  for dist_base, dist_zone in IE.getNodesWithParentsFromTypePath(dist_tree, 'CGNSBase_t/Zone_t'):

    part_zones = te_utils.get_partitioned_zones(part_base, I.getName(dist_base) + '/' + I.getName(dist_zone))

    # > BND and JNS
    bc_t_path = 'ZoneBC_t/BC_t'
    gc_t_path = 'ZoneGridConnectivity_t/GridConnectivity_t'

    # > Discover (skip GC created by partitioning)
    disc.discover_nodes_of_kind(dist_zone, part_zones, bc_t_path, comm,
          child_list=['FamilyName_t', 'GridLocation_t'], get_value='none')
    disc.discover_nodes_of_kind(dist_zone, part_zones, gc_t_path, comm,
          child_list=['GridLocation_t', 'GridConnectivityProperty_t', 'Ordinal', 'OrdinalOpp'],
          merge_rule= lambda path: conv.get_split_prefix(path),
          skip_rule = lambda node: conv.is_intra_gc(I.getName(node)))

  return dist_tree

# ------------------------------------------------------------------------
def get_zone_info(zone):
  """
  """
  # gridc = I.getNodeFromType1(zone, "GridCoordinates_t")
  # cx    = I.getNodeFromName1(gridc, "CoordinateX")[1]
  # cy    = I.getNodeFromName1(gridc, "CoordinateY")[1]
  # cz    = I.getNodeFromName1(gridc, "CoordinateZ")[1]

  # coords = np.concatenate([cx, cy, cz])
  # coords = coords.reshape( (3, cx.shape[0]))
  # coords = coords.transpose()
  # coords = coords.reshape( 3*cx.shape[0], order='C')

  pdm_nodes     = I.getNodeFromName1(zone, ":CGNS#Ppart")
  vtx_coords    = I.getNodeFromName1(pdm_nodes, "np_vtx_coord")[1]
  cell_face_idx = I.getNodeFromName1(pdm_nodes, "np_cell_face_idx")[1]
  cell_face     = I.getNodeFromName1(pdm_nodes, "np_cell_face")[1]
  cell_ln_to_gn = I.getNodeFromName1(pdm_nodes, "np_cell_ln_to_gn")[1]
  face_vtx_idx  = I.getNodeFromName1(pdm_nodes, "np_face_vtx_idx")[1]
  face_vtx      = I.getNodeFromName1(pdm_nodes, "np_face_vtx")[1]
  face_ln_to_gn = I.getNodeFromName1(pdm_nodes, "np_face_ln_to_gn")[1]
  vtx_ln_to_gn  = I.getNodeFromName1(pdm_nodes, "np_vtx_ln_to_gn")[1]
  return cell_face_idx, cell_face, cell_ln_to_gn, face_vtx_idx, face_vtx, face_ln_to_gn, vtx_coords, vtx_ln_to_gn


# ------------------------------------------------------------------------
def get_target_zone_info(zone, location='CellCenter'):
  """
  """
  gridc = I.getNodeFromType1(zone, "GridCoordinates_t")
  cx    = I.getNodeFromName1(gridc, "CoordinateX")[1]
  cy    = I.getNodeFromName1(gridc, "CoordinateY")[1]
  cz    = I.getNodeFromName1(gridc, "CoordinateZ")[1]

  # > Try to hook NGon
  found = False
  for elt in I.getNodesFromType1(zone, 'Elements_t'):
    if SIDS.ElementType(elt) == 22:
      found         = True
      face_vtx      = I.getNodeFromName1(elt, 'ElementConnectivity')[1]
      face_vtx_idx  = I.getNodeFromName1(elt, 'ElementStartOffset' )[1]
      ngon_pe       = I.getNodeFromName1(elt, 'ParentElements'     )[1]
  if not found :
    raise RuntimeError

  coords = np.concatenate([cx, cy, cz])
  coords = coords.reshape( (3, cx.shape[0]))
  coords = coords.transpose()
  coords = coords.reshape( 3*cx.shape[0], order='C')

  pdm_nodes     = I.getNodeFromName1(zone, ":CGNS#Ppart")
  vtx_coords    = I.getNodeFromName1(pdm_nodes, "np_vtx_coord")[1]
  cell_ln_to_gn = I.getNodeFromName1(pdm_nodes, "np_cell_ln_to_gn")[1]
  vtx_ln_to_gn  = I.getNodeFromName1(pdm_nodes, "np_vtx_ln_to_gn")[1]

  if(location == 'CellCenter'):
    n_cell = SIDS.zone_n_cell(zone)
    center_cell = np.empty(3*n_cell, dtype='double')
    Geometry.computeCellCenter__(center_cell,
                                 cx,
                                 cy,
                                 cz,
                                 ngon_pe,
                                 face_vtx,
                                 face_vtx_idx,
                                 n_cell,
                                 0)

    # > Keep alive
    I.newDataArray("cell_center", center_cell, parent=pdm_nodes)
    return center_cell, cell_ln_to_gn
  else:
    return vtx_coords, vtx_ln_to_gn


# ------------------------------------------------------------------------
def get_center_cell_cloud(zone):
  """
  """
  gridc = I.getNodeFromType1(zone, "GridCoordinates_t")
  cx    = I.getNodeFromName1(gridc, "CoordinateX")[1]
  cy    = I.getNodeFromName1(gridc, "CoordinateY")[1]
  cz    = I.getNodeFromName1(gridc, "CoordinateZ")[1]

  # > Try to hook NGon
  found = False
  for elt in I.getNodesFromType1(zone, 'Elements_t'):
    if SIDS.ElementType(elt) == 22:
      found         = True
      face_vtx      = I.getNodeFromName1(elt, 'ElementConnectivity')[1]
      face_vtx_idx  = I.getNodeFromName1(elt, 'ElementStartOffset' )[1]
      ngon_pe       = I.getNodeFromName1(elt, 'ParentElements'     )[1]
  if not found :
    raise RuntimeError

  coords = np.concatenate([cx, cy, cz])
  coords = coords.reshape( (3, cx.shape[0]))
  coords = coords.transpose()
  coords = coords.reshape( 3*cx.shape[0], order='C')

  pdm_nodes     = I.getNodeFromName1(zone, ":CGNS#Ppart")
  vtx_coords    = I.getNodeFromName1(pdm_nodes, "np_vtx_coord")[1]
  # cell_face_idx = I.getNodeFromName1(pdm_nodes, "np_cell_face_idx")[1]
  # cell_face     = I.getNodeFromName1(pdm_nodes, "np_cell_face")[1]
  cell_ln_to_gn = I.getNodeFromName1(pdm_nodes, "np_cell_ln_to_gn")[1]
  # face_vtx_idx  = I.getNodeFromName1(pdm_nodes, "np_face_vtx_idx")[1]
  # face_vtx      = I.getNodeFromName1(pdm_nodes, "np_face_vtx")[1]
  # face_ln_to_gn = I.getNodeFromName1(pdm_nodes, "np_face_ln_to_gn")[1]
  vtx_ln_to_gn  = I.getNodeFromName1(pdm_nodes, "np_vtx_ln_to_gn")[1]

  n_cell = SIDS.zone_n_cell(zone)
  center_cell = np.empty(3*n_cell, dtype='double')
  Geometry.computeCellCenter__(center_cell,
                               cx,
                               cy,
                               cz,
                               ngon_pe,
                               face_vtx,
                               face_vtx_idx,
                               n_cell,
                               0)

  # > Keep alive
  I.newDataArray("cell_center", center_cell, parent=pdm_nodes)
  # print("cell_center", center_cell)

  # return vtx_coords, vtx_ln_to_gn
  return center_cell, cell_ln_to_gn

# --------------------------------------------------------------------------
def locate_meshb_in_meshb():
  """
  """
  pass

def setup_src_mesh(mesh_loc, dist_tree_src, part_tree_src):
  """
  """
  n_domain_src = len(I.getZones(dist_tree_src))
  assert(n_domain_src == 1)

  n_part   = np.zeros(n_domain_src, dtype='int32')
  i_domain = 0
  for dist_base, dist_zone in IE.getNodesWithParentsFromTypePath(dist_tree_src, 'CGNSBase_t/Zone_t'):
    # > Get the list of all partition in this domain
    part_zones = te_utils.get_partitioned_zones(part_tree_src, I.getName(dist_base) + '/' + I.getName(dist_zone))

    mesh_loc.mesh_global_data_set(len(part_zones)) #  adapter si multi-domain
    assert(i_domain == 0)

    for i_part, part_zone in enumerate(part_zones):
      cell_face_idx, cell_face, cell_ln_to_gn, face_vtx_idx, face_vtx, face_ln_to_gn, vtx_coords, vtx_ln_to_gn = get_zone_info(part_zone)

      n_cell = cell_ln_to_gn.shape[0]
      n_face = face_ln_to_gn.shape[0]
      n_vtx  = vtx_ln_to_gn .shape[0]

      mesh_loc.part_set(i_part, n_cell, cell_face_idx, cell_face, cell_ln_to_gn,
                                n_face, face_vtx_idx, face_vtx, face_ln_to_gn,
                                n_vtx, vtx_coords, vtx_ln_to_gn)

    n_part[i_domain] = len(part_zones)
    i_domain =+ 1

# --------------------------------------------------------------------------
def setup_cloud_src_mesh(closest_point, dist_tree_src, part_tree_src):
  """
  """
  n_domain_src = len(I.getZones(dist_tree_src))
  assert(n_domain_src == 1)

  n_part   = np.zeros(n_domain_src, dtype='int32')
  i_domain = 0
  for dist_base, dist_zone in IE.getNodesWithParentsFromTypePath(dist_tree_src, 'CGNSBase_t/Zone_t'):
    # > Get the list of all partition in this domain
    part_zones = te_utils.get_partitioned_zones(part_tree_src, I.getName(dist_base) + '/' + I.getName(dist_zone))

    assert(i_domain == 0)

    for i_part, part_zone in enumerate(part_zones):
      center_cell, cell_ln_to_gn = get_center_cell_cloud(part_zone)
      closest_point.src_cloud_set(i_part, cell_ln_to_gn.shape[0], center_cell, cell_ln_to_gn)

    n_part[i_domain] = len(part_zones)
    i_domain =+ 1

# --------------------------------------------------------------------------
def setup_target_mesh(mesh_loc, dist_tree_target, part_tree_target, location='CellCenter'):
  """
  """
  n_domain_target = len(I.getZones(dist_tree_target))
  assert(n_domain_target == 1)

  i_domain = 0
  for dist_base, dist_zone in IE.getNodesWithParentsFromTypePath(dist_tree_target, 'CGNSBase_t/Zone_t'):
    # > Get the list of all partition in this domain
    part_zones = te_utils.get_partitioned_zones(part_tree_target, I.getName(dist_base) + '/' + I.getName(dist_zone))

    mesh_loc.n_part_cloud_set(0, len(part_zones)) # Pour l'instant 1 cloud et 1 partition
    assert(i_domain == 0)

    for i_part, part_zone in enumerate(part_zones):
      coords, ln_to_gn = get_target_zone_info(part_zone, location)
      n_points = ln_to_gn.shape[0]
      mesh_loc.cloud_set(0, i_part, n_points, coords, ln_to_gn)
    i_domain =+ 1

# --------------------------------------------------------------------------
def setup_gnum_for_unlocated(mesh_loc, closest_point, dist_tree_target, part_tree_target, comm, location='CellCenter'):
  """
  """
  n_domain_target = len(I.getZones(dist_tree_target))
  assert(n_domain_target == 1)

  n_unlocated = 0
  i_domain = 0
  for dist_base, dist_zone in IE.getNodesWithParentsFromTypePath(dist_tree_target, 'CGNSBase_t/Zone_t'):
    # > Get the list of all partition in this domain
    part_zones = te_utils.get_partitioned_zones(part_tree_target, I.getName(dist_base) + '/' + I.getName(dist_zone))


    gen_gnum = PDM.GlobalNumbering(3, # Dimension
                                   len(part_zones),
                                   0, # Merge
                                   0.,
                                   comm)
    assert(i_domain == 0)

    for i_part, part_zone in enumerate(part_zones):

      coords, ln_to_gn = get_target_zone_info(part_zone, location)
      results_located   = mesh_loc.located_get(0, i_part)  # Necessary for no leaks in ParaDiGM
      results_unlocated = mesh_loc.unlocated_get(0, i_part)

      n_unlocated += results_unlocated.shape[0]

      extract_coords   = EX.extract_from_indices(coords  , results_unlocated, 3, 1)
      extract_ln_to_gn = EX.extract_from_indices(ln_to_gn, results_unlocated, 1, 1)

      extract_n_points = extract_ln_to_gn.shape[0]
      gen_gnum.gnum_set_from_parent(i_part, extract_n_points, extract_ln_to_gn)

      pdm_nodes     = I.getNodeFromName1(part_zone, ":CGNS#Ppart")

      # > Keep alive
      I.newDataArray("unlocated_extract_coords"  , extract_coords  , parent=pdm_nodes)
      I.newDataArray("unlocated_extract_ln_to_gn", extract_ln_to_gn, parent=pdm_nodes)
      # I.newDataArray("unlocated_ln_to_gn", extract_ln_to_gn, parent=pdm_nodes)

    gen_gnum.gnum_compute()

    for i_part, part_zone in enumerate(part_zones):
      sub_ln_to_gn = gen_gnum.gnum_get(i_part)
      print("sub_ln_to_gn : ", sub_ln_to_gn)
      pdm_nodes     = I.getNodeFromName1(part_zone, ":CGNS#Ppart")
      I.newDataArray("unlocated_sub_ln_to_gn"  , sub_ln_to_gn["gnum"]  , parent=pdm_nodes)

      extract_coords = I.getNodeFromName1(pdm_nodes, "unlocated_extract_coords")
      closest_point.tgt_cloud_set(i_part, extract_n_points, I.getValue(extract_coords), sub_ln_to_gn["gnum"])

    i_domain =+ 1

# --------------------------------------------------------------------------
def post_and_set_closest_result(closest_point, interp_from_mesh_loc, dist_tree_target, part_tree_target, comm):
  """
  CETTE METHODE EST COMPLETEMENT FAUSSE : On attend des modif dans le pdm_mesh_location
  On prends l'arbre target en suposant que le source a exactment le même partitionnement !!!!!
  """
  n_domain_target = len(I.getZones(dist_tree_target))
  assert(n_domain_target == 1)

  n_unlocated = 0
  i_domain = 0
  for dist_base, dist_zone in IE.getNodesWithParentsFromTypePath(dist_tree_target, 'CGNSBase_t/Zone_t'):
    # > Get the list of all partition in this domain
    part_zones = te_utils.get_partitioned_zones(part_tree_target, I.getName(dist_base) + '/' + I.getName(dist_zone))

    assert(i_domain == 0)

    results_closest = closest_point.points_get(0)
    for i_part, part_zone in enumerate(part_zones):

      i_part_src = i_part # C'est FAUX
      results_tgt_in_src_closest = closest_point.tgt_in_src_get(i_part_src)

      pdm_nodes       = I.getNodeFromName1(part_zone, ":CGNS#Ppart")
      sub_ln_to_gn    = I.getNodeFromName1(pdm_nodes, "unlocated_sub_ln_to_gn")[1]
      parent_ln_to_gn = I.getNodeFromName1(pdm_nodes, "unlocated_extract_ln_to_gn")[1]

      # > Apply parent to result
      PDM.transform_to_parent_gnum(results_tgt_in_src_closest["tgt_in_src"], sub_ln_to_gn, parent_ln_to_gn, comm) # Inplace !!!!!

      I.newDataArray("tgt_in_src_idx", results_tgt_in_src_closest["tgt_in_src_idx"], parent=pdm_nodes)
      I.newDataArray("tgt_in_src"    , results_tgt_in_src_closest["tgt_in_src"]    , parent=pdm_nodes)

      interp_from_mesh_loc.points_in_elt_set(0, 0,
                                             results_tgt_in_src_closest["tgt_in_src_idx"],
                                             results_tgt_in_src_closest["tgt_in_src"],
                                             None,
                                             None,
                                             None,
                                             None,
                                             None,
                                             None)

    i_domain =+ 1



# --------------------------------------------------------------------------
def setup_unlocated_target_mesh(mesh_loc, closest_point, dist_tree_target, part_tree_target, location='CellCenter'):
  """
  """
  n_domain_target = len(I.getZones(dist_tree_target))
  assert(n_domain_target == 1)

  n_unlocated = 0
  i_domain = 0
  for dist_base, dist_zone in IE.getNodesWithParentsFromTypePath(dist_tree_target, 'CGNSBase_t/Zone_t'):
    # > Get the list of all partition in this domain
    part_zones = te_utils.get_partitioned_zones(part_tree_target, I.getName(dist_base) + '/' + I.getName(dist_zone))

    assert(i_domain == 0)

    for i_part, part_zone in enumerate(part_zones):

      coords, ln_to_gn = get_target_zone_info(part_zone, location)
      n_points = ln_to_gn.shape[0]
      results_located   = mesh_loc.located_get(0, i_part)  # Necessary for no leaks in ParaDiGM
      results_unlocated = mesh_loc.unlocated_get(0, i_part)

      n_unlocated += results_unlocated.shape[0]

      extract_coords   = EX.extract_from_indices(coords  , results_unlocated, 3, 1)
      extract_ln_to_gn = EX.extract_from_indices(ln_to_gn, results_unlocated, 1, 1)

      # extract_ln_to_gn = ln_to_gn[results_unlocated]
      extract_n_points = extract_ln_to_gn.shape[0]
      # print("extract_coords   : ", extract_coords  )
      # print("extract_ln_to_gn : ", extract_ln_to_gn)
      closest_point.tgt_cloud_set(i_part, extract_n_points, extract_coords, extract_ln_to_gn)

      pdm_nodes     = I.getNodeFromName1(part_zone, ":CGNS#Ppart")

      # > Keep alive
      I.newDataArray("extract_coords"  , extract_coords  , parent=pdm_nodes)
      I.newDataArray("extract_ln_to_gn", extract_ln_to_gn, parent=pdm_nodes)

    i_domain =+ 1

  return n_unlocated


# --------------------------------------------------------------------------
def mesha_to_meshb(part_tree_src,
                   part_tree_target,
                   comm,
                   location = 'CellCenter',
                   order = 0):
  """
    mesha is the src
    meshb is the target --> Correspond to the cloud of ParaDiGM
    The result of ParaDiGM is the list for each cell of meshb the list of point in target
  """
  dist_tree_src    = rebuild_partial_dist_tree(part_tree_src   , comm)
  dist_tree_target = rebuild_partial_dist_tree(part_tree_target, comm)
  # I.printTree(dist_tree_src)
  # I.printTree(dist_tree_target)

  n_domain_target = len(I.getZones(dist_tree_target))
  assert(n_domain_target == 1)

  # > Identify the number of cloud - for now, the number of partition of target (meshb)
  n_cloud = np.zeros(n_domain_target, dtype='int32')
  i_domain = 0
  for dist_base, dist_zone in IE.getNodesWithParentsFromTypePath(dist_tree_target, 'CGNSBase_t/Zone_t'):
    # > Get the list of all partition in this domain
    part_zones = te_utils.get_partitioned_zones(part_tree_target, I.getName(dist_base) + '/' + I.getName(dist_zone))
    n_cloud[i_domain] = len(part_zones)
    i_domain =+ 1

  mesh_nature = 1
  # mesh_loc = PDM.MeshLocation(mesh_nature, n_cloud[0], comm)
  # > Pour l'instant 1 seul cloud !!! On pourrait en faire plusieurs, par example les centres et les noeuds...
  mesh_loc = PDM.MeshLocation(mesh_nature, 1, comm)

  setup_src_mesh   (mesh_loc, dist_tree_src   , part_tree_src   )
  setup_target_mesh(mesh_loc, dist_tree_target, part_tree_target, location)

  mesh_loc.tolerance_set(1.e-6)
  # mesh_loc.tolerance_set(1.e-1)

  mesh_loc.compute()

  results = mesh_loc.location_get(0, 0)
  mesh_loc.dump_times()
  # for key, val in results.items():
  #   print(key, val)

  # print(results)
  results_pts = mesh_loc.points_in_elt_get(0, 0) # i_part, i_point_cloud
  # print(results_pts)
  # for key, val in results_pts.items():
  #   print(key, val)

  # > Il ne faut pas faire le get 2 fois :/ ATTENTION
  results_unlocated = mesh_loc.unlocated_get(0, 0)
  # print("Unlocated size = ", results_unlocated.shape[0])
  print(results_unlocated)

  n_unlocated = results_unlocated.shape[0]

  n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.MAX)
  print(" n_tot_unlocated = ", n_tot_unlocated )

  if( n_tot_unlocated > 0):
    n_closest = 1
    closest_point = PDM.ClosestPoints(comm, n_closest)
    closest_point.n_part_cloud_set(1, 1) # Pour l'instant 1 part src et 1 part target

    # > Create a global numbering for unalocated point (inside cgns ... )
    setup_gnum_for_unlocated(mesh_loc, closest_point, dist_tree_target, part_tree_target, comm)

    # setup_unlocated_target_mesh(mesh_loc, closest_point, dist_tree_target, part_tree_target)
    # print()

    # > Pour les target on doit faire que ceux qui n'ont pas été localisé (donc extractraction des précédement calculé )
    setup_cloud_src_mesh(closest_point, dist_tree_src, part_tree_src)

    closest_point.compute()

  # > Interpolation
  #    - For each part we have to compute a good interpolation for the point located
  #    - At the end, for each located point we have an interpolation
  interp_from_mesh_loc = PDM.InterpolateFromMeshLocation(1, comm)
  setup_src_mesh   (interp_from_mesh_loc, dist_tree_src   , part_tree_src   )
  setup_target_mesh(interp_from_mesh_loc, dist_tree_target, part_tree_target, location)

  interp_from_mesh_loc.points_in_elt_set(0, 0,
                                         results_pts["elt_pts_inside_idx"],
                                         results_pts["points_gnum"],
                                         results_pts["points_coords"],
                                         results_pts["points_uvw"],
                                         results_pts["points_weights_idx"],
                                         results_pts["points_weights"],
                                         results_pts["points_dist2"],
                                         results_pts["points_projected_coords"])

  # > Return result in "cloud"
  interp_from_mesh_loc.compute()

  fs_name = "Density"
  # fs_name = "TurbulentSANuTildeDensity"
  list_part_data_in = list()
  for zone in I.getZones(part_tree_src):

    fs = I.getNodeFromName1(zone, "FlowSolution#Init")
    da = I.getNodeFromName1(fs, fs_name)

    list_part_data_in.append(da[1])

  # print(len(list_part_data_in))
  results_interp = interp_from_mesh_loc.exch(0, list_part_data_in)
  # print("Avant  : ", results_interp[0])

  # > Recall the interpolation but with the unlocated part

  if( n_tot_unlocated > 0):
    # results_closest = closest_point.points_get(0)
    # for key, val in results_closest.items():
    #   print(key, val)

    # results_tgt_in_src_closest = closest_point.tgt_in_src_get(0)
    # for key, val in results_tgt_in_src_closest.items():
    #   print(key, val)

    # > Closest give the abolute numbering in the cloud numebering of the unloacated =! The true cloud with all point localted and unlocated
    post_and_set_closest_result(closest_point, interp_from_mesh_loc, dist_tree_target, part_tree_target, comm)
    interp_from_mesh_loc.exch_inplace(0, list_part_data_in, results_interp)

  list_part_data_in = list()
  for zone in I.getZones(part_tree_target):
    n_vtx  = SIDS.zone_n_vtx(zone)
    n_cell = SIDS.zone_n_cell(zone)
    if(location == 'CellCenter'):
      fs = I.newFlowSolution("FlowSolution#Init", gridLocation='CellCenter', parent=zone)
    else:
      fs = I.newFlowSolution("FlowSolution#Init", gridLocation='Vertex', parent=zone)

    da = I.newDataArray(fs_name, results_interp[0], parent=fs)
    # print(results_interp[0])

    # > Remove holder
    pdm_nodes     = I.getNodeFromName1(zone, ":CGNS#Ppart")
    I._rmNodesByName(pdm_nodes, "extract_coords" )
    I._rmNodesByName(pdm_nodes, "extract_ln_to_gn" )
    I._rmNodesByName(pdm_nodes, "cell_center" )
    I._rmNodesByName(pdm_nodes, "unlocated_extract_coords"   )
    I._rmNodesByName(pdm_nodes, "unlocated_extract_ln_to_gn" )
    I._rmNodesByName(pdm_nodes, "unlocated_sub_ln_to_gn"  )
    I._rmNodesByName(pdm_nodes, "tgt_in_src_idx"  )
    I._rmNodesByName(pdm_nodes, "tgt_in_src"      )

  for zone in I.getZones(part_tree_src):
    pdm_nodes     = I.getNodeFromName1(zone, ":CGNS#Ppart")
    I._rmNodesByName(pdm_nodes, "extract_coords" )
    I._rmNodesByName(pdm_nodes, "extract_ln_to_gn" )
    I._rmNodesByName(pdm_nodes, "cell_center" )
    I._rmNodesByName(pdm_nodes, "unlocated_extract_coords"   )
    I._rmNodesByName(pdm_nodes, "unlocated_extract_ln_to_gn" )
    I._rmNodesByName(pdm_nodes, "unlocated_sub_ln_to_gn"  )
    I._rmNodesByName(pdm_nodes, "tgt_in_src_idx"  )
    I._rmNodesByName(pdm_nodes, "tgt_in_src"      )
