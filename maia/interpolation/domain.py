from mpi4py import MPI
import numpy as np

import Converter.Internal     as I
import maia.sids.Internal_ext as IE

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

from maia.utils import py_utils
from maia.sids import sids

from cmaia.geometry.geometry import compute_center_cell_u
import cmaia.utils.extract_from_indices as EX

import Pypdm.Pypdm as PDM

# ------------------------------------------------------------------------
def get_target_zone_info(zone, location='CellCenter'):
  """
  """
  gridc = I.getNodeFromType1(zone, "GridCoordinates_t")
  cx    = I.getNodeFromName1(gridc, "CoordinateX")[1]
  cy    = I.getNodeFromName1(gridc, "CoordinateY")[1]
  cz    = I.getNodeFromName1(gridc, "CoordinateZ")[1]

  if(location == 'CellCenter'):
    ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
    assert len(ngons) == 1
    face_vtx      = I.getNodeFromName1(ngons[0], 'ElementConnectivity')[1]
    face_vtx_idx  = I.getNodeFromName1(ngons[0], 'ElementStartOffset' )[1]
    ngon_pe       = I.getNodeFromName1(ngons[0], 'ParentElements'     )[1]

    cell_ln_to_gn = IE.getGlobalNumbering(zone, 'Cell').astype(pdm_gnum_dtype)
    n_cell = sids.CellSize(zone)

    center_cell = compute_center_cell_u(n_cell,
                                        cx, cy, cz,
                                        face_vtx,
                                        face_vtx_idx,
                                        ngon_pe)
    return center_cell, cell_ln_to_gn

  else:
    vtx_coords = py_utils.interweave_arrays([cx,cy,cz])
    vtx_ln_to_gn  = IE.getGlobalNumbering(zone, 'Vertex').astype(pdm_gnum_dtype)

    return vtx_coords, vtx_ln_to_gn

# ------------------------------------------------------------------------
def get_center_cell_cloud(zone):
  """
  """
  gridc = I.getNodeFromType1(zone, "GridCoordinates_t")
  cx    = I.getNodeFromName1(gridc, "CoordinateX")[1]
  cy    = I.getNodeFromName1(gridc, "CoordinateY")[1]
  cz    = I.getNodeFromName1(gridc, "CoordinateZ")[1]

  ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
  assert len(ngons) == 1
  face_vtx      = I.getNodeFromName1(ngons[0], 'ElementConnectivity')[1]
  face_vtx_idx  = I.getNodeFromName1(ngons[0], 'ElementStartOffset' )[1]
  ngon_pe       = I.getNodeFromName1(ngons[0], 'ParentElements'     )[1]

  cell_ln_to_gn = IE.getGlobalNumbering(zone, 'Cell')
  n_cell = sids.zone_n_cell(zone)

  center_cell = compute_center_cell_u(n_cell,
                                      cx, cy, cz,
                                      face_vtx,
                                      face_vtx_idx,
                                      ngon_pe)

  return center_cell, cell_ln_to_gn

def setup_src_mesh(mesh_loc, parts_per_domain, keep_alive):
  """
  """
  n_domain_src = len(parts_per_domain)
  assert(n_domain_src == 1)

  for i_domain, part_zones in enumerate(parts_per_domain):
    mesh_loc.mesh_global_data_set(len(part_zones)) #  adapter si multi-domain

    for i_part, part_zone in enumerate(part_zones):

      gridc_n    = I.getNodeFromName1(part_zone, 'GridCoordinates')
      cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
      cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
      cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
      vtx_coords = py_utils.interweave_arrays([cx,cy,cz])
      keep_alive.append(vtx_coords)

      ngons  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
      nfaces = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NFACE_n']
      assert len(nfaces) == len(ngons) == 1

      cell_face_idx = I.getNodeFromName1(nfaces[0], "ElementStartOffset")[1]
      cell_face     = I.getNodeFromName1(nfaces[0], "ElementConnectivity")[1]
      face_vtx_idx  = I.getNodeFromName1(ngons[0],  "ElementStartOffset")[1]
      face_vtx      = I.getNodeFromName1(ngons[0],  "ElementConnectivity")[1]

      face_ln_to_gn = IE.getGlobalNumbering(ngons[0], 'Element').astype(pdm_gnum_dtype)
      cell_ln_to_gn = IE.getGlobalNumbering(part_zone, 'Cell').astype(pdm_gnum_dtype)
      vtx_ln_to_gn  = IE.getGlobalNumbering(part_zone, 'Vertex').astype(pdm_gnum_dtype)

      n_cell = cell_ln_to_gn.shape[0]
      n_face = face_ln_to_gn.shape[0]
      n_vtx  = vtx_ln_to_gn .shape[0]

      ngon_pe       = I.getNodeFromName1(ngons[0], 'ParentElements'     )[1]
      center_cell = compute_center_cell_u(n_cell,
                                          cx, cy, cz,
                                          face_vtx,
                                          face_vtx_idx,
                                          ngon_pe)
      keep_alive.append(cell_ln_to_gn)

      mesh_loc.part_set(i_part, n_cell, cell_face_idx, cell_face, cell_ln_to_gn,
                                n_face, face_vtx_idx, face_vtx, face_ln_to_gn,
                                n_vtx, vtx_coords, vtx_ln_to_gn)


# --------------------------------------------------------------------------
def setup_cloud_src_mesh(closest_point, parts_per_domain, keep_alive):
  """
  """
  n_domain_src = len(parts_per_domain)
  assert(n_domain_src == 1)

  for i_domain, part_zones in enumerate(parts_per_domain):
    for i_part, part_zone in enumerate(part_zones):
      center_cell, cell_ln_to_gn = get_center_cell_cloud(part_zone)
      keep_alive.append(center_cell)
      keep_alive.append(cell_ln_to_gn)
      closest_point.src_cloud_set(i_part, cell_ln_to_gn.shape[0], center_cell, cell_ln_to_gn)

# --------------------------------------------------------------------------
def setup_target_mesh(mesh_loc, parts_per_domain, location='CellCenter', keep_alive=None):
  """
  """
  n_domain_target = len(parts_per_domain)
  assert(n_domain_target == 1)

  for i_domain, part_zones in enumerate(parts_per_domain):
    # > Get the list of all partition in this domain

    mesh_loc.n_part_cloud_set(0, len(part_zones)) # Pour l'instant 1 cloud et 1 partition

    for i_part, part_zone in enumerate(part_zones):
      coords, ln_to_gn = get_target_zone_info(part_zone, location)
      keep_alive.append(coords)
      keep_alive.append(ln_to_gn)
      n_points = ln_to_gn.shape[0]
      mesh_loc.cloud_set(0, i_part, n_points, coords, ln_to_gn)

# --------------------------------------------------------------------------
def setup_gnum_for_unlocated(mesh_loc, closest_point, parts_per_domain, comm, location='CellCenter', keep_alive = None):
  """
  """
  n_domain_target = len(parts_per_domain)
  assert(n_domain_target == 1)

  unlocated_gnum = {"unlocated_extract_ln_to_gn" : [list() for dom in parts_per_domain],
                    "unlocated_sub_ln_to_gn" : [list() for dom in parts_per_domain] }

  n_unlocated = 0
  for i_domain, part_zones in enumerate(parts_per_domain):
    # > Get the list of all partition in this domain

    gen_gnum = PDM.GlobalNumbering(3, # Dimension
                                   len(part_zones),
                                   0, # Merge
                                   0.,
                                   comm)

    domain_coords = list()
    for i_part, part_zone in enumerate(part_zones):

      coords, ln_to_gn = get_target_zone_info(part_zone, location)
      keep_alive.append(coords)
      results_located   = mesh_loc.located_get(0, i_part)  # Necessary for no leaks in ParaDiGM
      results_unlocated = mesh_loc.unlocated_get(0, i_part)

      n_unlocated += results_unlocated.shape[0]

      extract_coords   = EX.extract_from_indices(coords  , results_unlocated, 3, 1)
      extract_ln_to_gn = EX.extract_from_indices(ln_to_gn, results_unlocated, 1, 1)

      extract_n_points = extract_ln_to_gn.shape[0]
      gen_gnum.gnum_set_from_parent(i_part, extract_n_points, extract_ln_to_gn)

      unlocated_gnum["unlocated_extract_ln_to_gn"][i_domain].append(extract_ln_to_gn)

      # > Keep alive
      domain_coords.append(extract_coords)
      keep_alive.append(extract_ln_to_gn)

    gen_gnum.gnum_compute()

    for i_part, part_zone in enumerate(part_zones):
      sub_ln_to_gn = gen_gnum.gnum_get(i_part)
      extract_n_points = sub_ln_to_gn["gnum"].shape[0]
      closest_point.tgt_cloud_set(i_part, extract_n_points, domain_coords[i_part], sub_ln_to_gn["gnum"])
      keep_alive.append(sub_ln_to_gn["gnum"])
      unlocated_gnum["unlocated_sub_ln_to_gn"][i_domain].append(sub_ln_to_gn["gnum"])
    keep_alive.append(domain_coords)

  return unlocated_gnum


# --------------------------------------------------------------------------
def post_and_set_closest_result(closest_point, interp_from_mesh_loc, n_src_part, tgt_parts_per_domain, unlocated_gnum, comm):
  """
  Appliquée sur target
  CETTE METHODE EST COMPLETEMENT FAUSSE : On attend des modif dans le pdm_mesh_location
  On prends l'arbre target en suposant que le source a exactment le même partitionnement !!!!!
  """
  n_domain_target = len(tgt_parts_per_domain)
  assert(n_domain_target == 1)

  for i_domain, tgt_part_zones in enumerate(tgt_parts_per_domain):
    # > Get the list of all partition in this domain

    #i_part_src = 0 #i_part # C'est FAUX
    #Pour chaque pt source, points cibles qui l'ont détecté comme le plus proche
    all_results_tgt_in_src_closest = [closest_point.tgt_in_src_get(i_part_src) for i_part_src in range(n_src_part)]
    # print("initial closest LEN", len(results_tgt_in_src_closest["tgt_in_src"]))
    # print("initial closest : ", results_tgt_in_src_closest)

    sub_ln_to_gn    = unlocated_gnum["unlocated_sub_ln_to_gn"][i_domain]
    parent_ln_to_gn = unlocated_gnum["unlocated_extract_ln_to_gn"][i_domain]
    PTB = PDM.PartToBlock(comm, sub_ln_to_gn, pWeight=None, partN=len(tgt_part_zones),
                          t_distrib=0, t_post=1, t_stride=0)

    dist_data = dict()
    part_data = {'parent_gnum' : parent_ln_to_gn}
    PTB.PartToBlock_Exchange(dist_data, part_data)

    pdm_distrib = PTB.getDistributionCopy()

    part_data = {'parent_gnum' : [np.empty(results_tgt_in_src_closest["tgt_in_src"].shape[0],
      dtype=dist_data['parent_gnum'].dtype) for results_tgt_in_src_closest in all_results_tgt_in_src_closest]}
    BTP = PDM.BlockToPart(pdm_distrib, comm, [results_tgt_in_src_closest["tgt_in_src"] for results_tgt_in_src_closest in all_results_tgt_in_src_closest], len(all_results_tgt_in_src_closest))
    BTP.BlockToPart_Exchange(dist_data, part_data)

    for i_part_src, results_tgt_in_src_closest in enumerate(all_results_tgt_in_src_closest):
      results_tgt_in_src_closest["tgt_in_src"] = part_data["parent_gnum"][i_part_src]
      print("updated closest : ", results_tgt_in_src_closest["tgt_in_src"])

    #Here partition is related to source
    for i_part_src, results_tgt_in_src_closest in enumerate(all_results_tgt_in_src_closest):
      interp_from_mesh_loc.points_in_elt_set(i_part_src, 0,
                                             results_tgt_in_src_closest["tgt_in_src_idx"],
                                             results_tgt_in_src_closest["tgt_in_src"],
                                             None,
                                             None,
                                             None,
                                             None,
                                             None,
                                             None)
    # I.newDataArray("tgt_in_src_idx", results_tgt_in_src_closest["tgt_in_src_idx"], parent=part_zones[0])
    # I.newDataArray("tgt_in_src"    , results_tgt_in_src_closest["tgt_in_src"]    , parent=part_zones[0])


# --------------------------------------------------------------------------
def mesha_to_meshb(part_tree_src,
                   part_tree_target,
                   comm,
                   location = 'CellCenter',
                   order = 0):
  """
    
    N single domain interface


    mesha is the src
    meshb is the target --> Correspond to the cloud of ParaDiGM
    The result of ParaDiGM is the list for each cell of meshb the list of point in target
  """

  mesh_loc = PDM.MeshLocation(mesh_nature=1, n_point_cloud=1, comm=comm)
  keep_alive = list()

  setup_src_mesh   (mesh_loc, [part_tree_src], keep_alive)
  setup_target_mesh(mesh_loc, [part_tree_target], location, keep_alive)

  n_part_src = len(part_tree_src)
  n_part_tgt = len(part_tree_target)

  mesh_loc.tolerance_set(1.e-6)
  # mesh_loc.tolerance_set(1.e-1)
  #print("SRC gnum", IE.getGlobalNumbering(I.getZones(part_tree_src)[0], 'Cell'))

  mesh_loc.compute()

  for i_tgt_part in range(n_part_tgt):
    results = mesh_loc.location_get(0, i_tgt_part) #Pour chaque pt target, elt source associé (i_pt_cloud, i_part)
    #mesh_loc.dump_times()
    # print(comm.Get_rank(), 'target gnum', results['g_num'])
    # print(comm.Get_rank(), 'src gnum', results['location'])
    # for key, val in results.items():
      # print(key, '\n',  val)

  # print(results)
  all_results_pts = []
  for i_src_part in range(n_part_src):
    results_pts = mesh_loc.points_in_elt_get(i_src_part, 0) #Pour chaque elt source, listes des points target localisés (i_part, i_point_cloud)
    #print(results_pts)
    # print(i_src_part, results_pts['points_gnum'])
    # print(i_src_part, results_pts['points_coords'][::3])
    # for key, val in results_pts.items():
      # print(key, '\n', val)
    all_results_pts.append(results_pts)
  # for key in results_pts:
    # print("test", key, (all_results_pts[0][key] == all_results_pts[2][key]).all())

  # > Il ne faut pas faire le get 2 fois :/ ATTENTION
  results_unlocated = mesh_loc.unlocated_get(0, 0) #Todo part
  #print("Unlocated size = ", results_unlocated.shape[0])
  # print(results_unlocated)

  n_unlocated = sum([mesh_loc.unlocated_get(0,i_part).shape[0] for i_part in range(n_part_tgt)])

  n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.SUM)
  if(comm.Get_rank() == 0):
    print(" n_tot_unlocated = ", n_tot_unlocated )

  #n_tot_unlocated = 0
  #Todo : allow other than double

  if( n_tot_unlocated > 0):
    n_closest = 1
    closest_point = PDM.ClosestPoints(comm, n_closest)
    closest_point.n_part_cloud_set(n_part_src, n_part_tgt) # Pour l'instant 1 part src et N part target

    # > Create a global numbering for unalocated point (inside cgns ... )
    unlocated_gnum = setup_gnum_for_unlocated(mesh_loc, closest_point, [part_tree_target], comm, 'CellCenter', keep_alive)

    # > Pour les target on doit faire que ceux qui n'ont pas été localisé (donc extractraction des précédement calculé )
    setup_cloud_src_mesh(closest_point, [part_tree_src], keep_alive)

    closest_point.compute()

  # > Interpolation
  #    - For each part we have to compute a good interpolation for the point located
  #    - At the end, for each located point we have an interpolation
  interp_from_mesh_loc = PDM.InterpolateFromMeshLocation(n_point_cloud=1, comm=comm)
  setup_src_mesh   (interp_from_mesh_loc, [part_tree_src], keep_alive   )
  setup_target_mesh(interp_from_mesh_loc, [part_tree_target], location, keep_alive)

  for i_part, results_pts in enumerate(all_results_pts):
    interp_from_mesh_loc.points_in_elt_set(i_part, 0,
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
  #Here source, we should loop on parts for all domain
  for part in part_tree_src:

    fs = I.getNodeFromName1(part, "FlowSolution#Init")
    da = I.getNodeFromName1(fs, fs_name)

    list_part_data_in.append(da[1])

  results_interp = interp_from_mesh_loc.exch(0, list_part_data_in)

  # > Recall the interpolation but with the unlocated part

  if( n_tot_unlocated > 0):

    # > Closest give the abolute numbering in the cloud numebering of the unloacated =! The true cloud with all point localted and unlocated
    post_and_set_closest_result(closest_point, interp_from_mesh_loc, n_part_src, [part_tree_target], unlocated_gnum, comm)
    interp_from_mesh_loc.exch_inplace(0, list_part_data_in, results_interp)

  for i_part, part in enumerate(part_tree_target):
    if(location == 'CellCenter'):
      fs = I.newFlowSolution("FlowSolution#Init", gridLocation='CellCenter', parent=part)
    else:
      fs = I.newFlowSolution("FlowSolution#Init", gridLocation='Vertex', parent=part)

    da = I.newDataArray(fs_name, results_interp[i_part], parent=fs)


def mesha_to_meshb_1(part_tree_src,
                   part_tree_target,
                   comm,
                   location = 'CellCenter',
                   order = 0):
  mesha_to_meshb(part_tree_src, part_tree_target, comm, location='CellCenter', order=0)
