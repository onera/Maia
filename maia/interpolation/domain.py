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
def get_point_cloud(zone, location='CellCenter'):
  """
  """
  gridc = I.getNodeFromType1(zone, "GridCoordinates_t")
  cx    = I.getNodeFromName1(gridc, "CoordinateX")[1]
  cy    = I.getNodeFromName1(gridc, "CoordinateY")[1]
  cz    = I.getNodeFromName1(gridc, "CoordinateZ")[1]

  if location == 'Vertex':
    vtx_coords   = py_utils.interweave_arrays([cx,cy,cz])
    vtx_ln_to_gn = IE.getGlobalNumbering(zone, 'Vertex').astype(pdm_gnum_dtype)
    return vtx_coords, vtx_ln_to_gn

  elif location == 'CellCenter':
    ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
    assert len(ngons) == 1
    face_vtx      = I.getNodeFromName1(ngons[0], 'ElementConnectivity')[1]
    face_vtx_idx  = I.getNodeFromName1(ngons[0], 'ElementStartOffset' )[1]
    ngon_pe       = I.getNodeFromName1(ngons[0], 'ParentElements'     )[1]

    cell_ln_to_gn = IE.getGlobalNumbering(zone, 'Cell').astype(pdm_gnum_dtype)
    n_cell = sids.zone_n_cell(zone)

    center_cell = compute_center_cell_u(n_cell,
                                        cx, cy, cz,
                                        face_vtx,
                                        face_vtx_idx,
                                        ngon_pe)
    return center_cell, cell_ln_to_gn

  raise RuntimeError("Unknow location")

# ------------------------------------------------------------------------

def setup_src_mesh(mesh_loc, src_parts_per_domain, keep_alive):
  """
  """
  n_domain_src = len(src_parts_per_domain)
  assert(n_domain_src == 1)

  for i_domain, src_part_zones in enumerate(src_parts_per_domain):
    mesh_loc.mesh_global_data_set(len(src_part_zones)) #  For now only on domain is supported

    for i_part, src_part in enumerate(src_part_zones):

      gridc_n    = I.getNodeFromName1(src_part, 'GridCoordinates')
      cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
      cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
      cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
      vtx_coords = py_utils.interweave_arrays([cx,cy,cz])
      keep_alive.append(vtx_coords)

      ngons  = [e for e in I.getNodesFromType1(src_part, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
      nfaces = [e for e in I.getNodesFromType1(src_part, 'Elements_t') if sids.ElementCGNSName(e) == 'NFACE_n']
      assert len(nfaces) == len(ngons) == 1

      cell_face_idx = I.getNodeFromName1(nfaces[0], "ElementStartOffset")[1]
      cell_face     = I.getNodeFromName1(nfaces[0], "ElementConnectivity")[1]
      face_vtx_idx  = I.getNodeFromName1(ngons[0],  "ElementStartOffset")[1]
      face_vtx      = I.getNodeFromName1(ngons[0],  "ElementConnectivity")[1]

      face_ln_to_gn = IE.getGlobalNumbering(ngons[0], 'Element').astype(pdm_gnum_dtype)
      cell_ln_to_gn = IE.getGlobalNumbering(src_part, 'Cell').astype(pdm_gnum_dtype)
      vtx_ln_to_gn  = IE.getGlobalNumbering(src_part, 'Vertex').astype(pdm_gnum_dtype)

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
      #For the closest point we always want cell center (?)
      center_cell, cell_ln_to_gn = get_point_cloud(part_zone, 'CellCenter')
      keep_alive.append(center_cell)
      keep_alive.append(cell_ln_to_gn)
      closest_point.src_cloud_set(i_part, cell_ln_to_gn.shape[0], center_cell, cell_ln_to_gn)

# --------------------------------------------------------------------------
def setup_target_mesh(mesh_loc, tgt_parts_per_domain, location='CellCenter', keep_alive=None):
  """
  """
  n_domain_target = len(tgt_parts_per_domain)
  assert(n_domain_target == 1)

  for i_domain, tgt_part_zones in enumerate(tgt_parts_per_domain):
    # > Get the list of all partition in this domain

    mesh_loc.n_part_cloud_set(0, len(tgt_part_zones)) # Pour l'instant 1 cloud et 1 partition

    for i_part, tgt_part in enumerate(tgt_part_zones):
      coords, ln_to_gn = get_point_cloud(tgt_part, location)
      keep_alive.append(coords)
      keep_alive.append(ln_to_gn)
      n_points = ln_to_gn.shape[0]
      mesh_loc.cloud_set(0, i_part, n_points, coords, ln_to_gn)

# --------------------------------------------------------------------------
def create_gnum_for_unlocated(mesh_loc, parts_per_domain, comm, location='CellCenter', keep_alive = None):
  """
  #Here domain matters! Because we are working on lngn
  """
  n_domain_target = len(parts_per_domain)
  assert(n_domain_target == 1)

  unlocated_gnum = {"unlocated_extract_ln_to_gn" : [list() for dom in parts_per_domain],
                    "unlocated_sub_ln_to_gn"     : [list() for dom in parts_per_domain],
                    "extracted_coords"           : [list() for dom in parts_per_domain]}

  for i_domain, part_zones in enumerate(parts_per_domain):
    # > Get the list of all partition in this domain

    gen_gnum = PDM.GlobalNumbering(3, # Dimension
                                   len(part_zones),
                                   0, # Merge
                                   0.,
                                   comm)

    domain_coords = list()
    for i_part, part_zone in enumerate(part_zones):

      #We could retrieve it isteead of recomputing !!
      coords, ln_to_gn = get_point_cloud(part_zone, location)
      keep_alive.append(coords)
      results_located   = mesh_loc.located_get(0, i_part)  # Necessary for no leaks in ParaDiGM
      results_unlocated = mesh_loc.unlocated_get(0, i_part)

      extract_coords   = EX.extract_from_indices(coords  , results_unlocated, 3, 1)
      extract_ln_to_gn = EX.extract_from_indices(ln_to_gn, results_unlocated, 1, 1)

      extract_n_points = extract_ln_to_gn.shape[0]
      gen_gnum.gnum_set_from_parent(i_part, extract_n_points, extract_ln_to_gn)

      unlocated_gnum["unlocated_extract_ln_to_gn"][i_domain].append(extract_ln_to_gn)
      unlocated_gnum["extracted_coords"][i_domain].append(extract_coords)

      # > Keep alive
      domain_coords.append(extract_coords)
      keep_alive.append(extract_ln_to_gn)

    gen_gnum.gnum_compute()

    for i_part, part_zone in enumerate(part_zones):
      sub_ln_to_gn = gen_gnum.gnum_get(i_part)
      #keep_alive.append(sub_ln_to_gn["gnum"])
      unlocated_gnum["unlocated_sub_ln_to_gn"][i_domain].append(sub_ln_to_gn["gnum"])
    keep_alive.append(domain_coords)

  return unlocated_gnum

# --------------------------------------------------------------------------
def mesha_to_meshb(src_parts_per_dom,
                   tgt_parts_per_dom,
                   comm,
                   location = 'CellCenter',
                   order = 0):
  """
    
    N single domain interface


    mesha is the src
    meshb is the target --> Correspond to the cloud of ParaDiGM
    The result of ParaDiGM is the list for each cell of meshb the list of point in target
  """
  n_dom_src = len(src_parts_per_dom)
  n_dom_tgt = len(tgt_parts_per_dom)

  assert n_dom_src == n_dom_tgt == 1
  n_part_src = len(src_parts_per_dom[0])
  n_part_tgt = len(tgt_parts_per_dom[0])

  keep_alive = list()

  #Phase 1 -- localisation
  mesh_loc = PDM.MeshLocation(mesh_nature=1, n_point_cloud=1, comm=comm)
  setup_src_mesh   (mesh_loc, src_parts_per_dom, keep_alive)
  setup_target_mesh(mesh_loc, tgt_parts_per_dom, location, keep_alive)

  mesh_loc.tolerance_set(1.e-6)
  mesh_loc.compute()

  #This is result from the target perspective -- not usefull here
  # for i_tgt_part in range(n_part_tgt):
    # #Pour chaque pt target, elt source associé (i_pt_cloud, i_part)
    # results = mesh_loc.location_get(0, i_tgt_part)

  #This is result from the source perspective : for each source part, list of tgt points
  # located in it (api : i_part,i_pt_cloud)
  all_located_inv = [mesh_loc.points_in_elt_get(i_src_part, 0) for i_src_part in range(n_part_src)]

  n_unlocated = sum([mesh_loc.unlocated_get(0,i_part).shape[0] for i_part in range(n_part_tgt)])

  n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.SUM)
  if(comm.Get_rank() == 0):
    print(" n_tot_unlocated = ", n_tot_unlocated )

  #n_tot_unlocated = 0
  #Todo : allow other than double

  if( n_tot_unlocated > 0):
    closest_point = PDM.ClosestPoints(comm, n_closest=1)
    closest_point.n_part_cloud_set(n_part_src, n_part_tgt)

    # > Create a global numbering for unalocated point (inside cgns ... )
    #unlocated_gnum = setup_gnum_for_unlocated(mesh_loc, closest_point, tgt_parts_per_dom, comm, 'CellCenter', keep_alive)
    unlocated_gnum = create_gnum_for_unlocated(mesh_loc, tgt_parts_per_dom, comm, 'CellCenter', keep_alive)
    for i_domain, part_zones in enumerate(tgt_parts_per_dom):
      for i_part, part_zone in enumerate(part_zones):

        extract_coords   = unlocated_gnum["extracted_coords"][i_domain][i_part]
        sub_ln_to_gn     = unlocated_gnum["unlocated_sub_ln_to_gn"][i_domain][i_part]
        closest_point.tgt_cloud_set(i_part, extract_coords.shape[0]//3, extract_coords, sub_ln_to_gn)
        keep_alive.append(extract_coords)

    # > Pour les target on doit faire que ceux qui n'ont pas été localisé (donc extractraction des précédement calculé )
    setup_cloud_src_mesh(closest_point, src_parts_per_dom, keep_alive)

    closest_point.compute()

    #Pour chaque pt source, points cibles qui l'ont détecté comme le plus proche
    #only domain id is usefull here
    for i_domain, part_zones in enumerate(tgt_parts_per_dom):
      all_closest_inv = [closest_point.tgt_in_src_get(i_part_src) for i_part_src in range(n_part_src)]
      gnum_to_transform = [results["tgt_in_src"] for results in all_closest_inv]
      sub_ln_to_gn    = unlocated_gnum["unlocated_sub_ln_to_gn"][i_domain]
      parent_ln_to_gn = unlocated_gnum["unlocated_extract_ln_to_gn"][i_domain]

      #Inplace 
      PDM.transform_to_parent_gnum(gnum_to_transform, sub_ln_to_gn, parent_ln_to_gn, comm)


  # > Interpolation
  #    - For each part we have to compute a good interpolation for the point located
  #    - At the end, for each located point we have an interpolation
  interpolator = PDM.InterpolateFromMeshLocation(n_point_cloud=1, comm=comm)

  # setup_src_mesh   (interpolator, src_parts_per_dom, keep_alive   ) #Not mandarory ?
  setup_target_mesh(interpolator, tgt_parts_per_dom, location, keep_alive) #Mandatory  but should be avoided
  one_or_two = 2 if n_unlocated > 0 else 1
  interpolator.mesh_global_data_set(one_or_two*n_part_src) #  For now only on domain is supported

  for i_part, res_loc in enumerate(all_located_inv):
    interpolator.points_in_elt_set(one_or_two*i_part, 0, **res_loc)
  if n_unlocated > 0:
    for i_part, res_clo in enumerate(all_closest_inv):
      interpolator.points_in_elt_set(one_or_two*i_part+1, 0, res_clo["tgt_in_src_idx"], res_clo["tgt_in_src"],
          None, None, None, None, None, None)

  # > Return result in "cloud"
  #interpolator.compute()

  fs_name = "Density"
  # fs_name = "TurbulentSANuTildeDensity"
  list_part_data_in = list()
  #Here source, we should loop on parts for all domain
  for part in src_parts_per_dom[0]:

    fs = I.getNodeFromName1(part, "FlowSolution#Init")
    da = I.getNodeFromName1(fs, fs_name)

    list_part_data_in.append(da[1])
    if one_or_two == 2:
      list_part_data_in.append(da[1])

  results_interp = interpolator.exch(0, list_part_data_in)

  for i_part, part in enumerate(tgt_parts_per_dom[0]):
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
  mesha_to_meshb([part_tree_src], [part_tree_target], comm, location='CellCenter', order=0)
