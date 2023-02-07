import os
import mpi4py.MPI as MPI

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils import np_utils, par_utils, layouts

import Pypdm.Pypdm as PDM


# FEFLO
feflo_path = "/stck/jvanhare/wkdir/spiro/bin/feflo.a"






# ---------------------------------------------------------
def _add_sections_to_zone(dist_zone, section, shift_elmt, comm):
  """
  """
  if section == None: return shift_elmt
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  for i_section, section in enumerate(section["sections"]):
    cgns_elmt_type = MT.pdm_elts.pdm_elt_name_to_cgns_element_type(section["pdm_type"])

    elmt = PT.new_Elements(f"{cgns_elmt_type}.{i_section}", cgns_elmt_type,
                           erange = [shift_elmt, shift_elmt + section["np_distrib"][n_rank]-1], parent=dist_zone)
    PT.new_DataArray('ElementConnectivity', section["np_connec"], parent=elmt)

    shift_elmt += section["np_distrib"][n_rank]

    distrib   = section["np_distrib"][[i_rank, i_rank+1, n_rank]]
    MT.newDistribution({'Element' : distrib}, parent=elmt)

  return shift_elmt


def dmesh_nodal_to_cgns(dmesh_nodal, comm, dicttag_to_bcinfo, families, out_files, isotrop):
  """
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()
  g_dims = dmesh_nodal.dmesh_nodal_get_g_dims()

  sections_vol   = None
  sections_surf  = None
  sections_ridge = None

  edge_groups    = None
  face_groups    = None

  if g_dims["n_cell_abs"] > 0:
    sections_vol  = dmesh_nodal.dmesh_nodal_get_sections(PDM._PDM_GEOMETRY_KIND_VOLUMIC, comm)
  sections_surf   = dmesh_nodal.dmesh_nodal_get_sections(PDM._PDM_GEOMETRY_KIND_SURFACIC, comm)
  sections_ridge  = dmesh_nodal.dmesh_nodal_get_sections(PDM._PDM_GEOMETRY_KIND_RIDGE   , comm)
  sections_corner = dmesh_nodal.dmesh_nodal_get_sections(PDM._PDM_GEOMETRY_KIND_CORNER  , comm)
  vtx_groups      = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_CORNER)
  edge_groups     = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_RIDGE)
  face_groups     = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_SURFACIC)


  # > Generate dist_tree
  dim = 3
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase(parent=dist_tree)

  if g_dims["n_cell_abs"] > 0:
    dist_zone = PT.new_Zone(name='zone', size=[[g_dims["n_vtx_abs"], g_dims["n_cell_abs"], 0]],
                            type='Unstructured', parent=dist_base)
  else:
    dist_zone = PT.new_Zone(name='zone', size=[[g_dims["n_vtx_abs"], g_dims["n_face_abs"], 0]],
                            type='Unstructured', parent=dist_base)

  # > Grid coordinates
  vtx_data = dmesh_nodal.dmesh_nodal_get_vtx(comm)
  cx, cy, cz = layouts.interlaced_to_tuple_coords(vtx_data['np_vtx'])  
  grid_coord = PT.new_GridCoordinates(parent=dist_zone)
  PT.new_DataArray('CoordinateX', cx, parent=grid_coord)
  PT.new_DataArray('CoordinateY', cy, parent=grid_coord)
  PT.new_DataArray('CoordinateZ', cz, parent=grid_coord)
  # > Section implicitement range donc on maintiens un compteur
  shift_elmt        = 1
  shift_elmt_vol    = _add_sections_to_zone(dist_zone, sections_vol   , shift_elmt      , comm)
  shift_elmt_surf   = _add_sections_to_zone(dist_zone, sections_surf  , shift_elmt_vol  , comm)
  shift_elmt_ridge  = _add_sections_to_zone(dist_zone, sections_ridge , shift_elmt_surf , comm)
  shift_elmt_corner = _add_sections_to_zone(dist_zone, sections_corner, shift_elmt_ridge, comm)

  # > BCs
  shift_bc = shift_elmt_vol - 1 if sections_vol is not None else shift_elmt_surf - 1
  zone_bc = PT.new_ZoneBC(parent=dist_zone)

  if face_groups is not None:
    face_group_idx = face_groups['dgroup_elmt_idx']
    face_group = shift_bc + face_groups['dgroup_elmt']
    distri = np.empty(n_rank, dtype=face_group.dtype)
    n_face_group = face_group_idx.shape[0] - 1

    for i_bc in range(n_face_group):
      bc_n = PT.new_BC('dcube_bnd_{0}'.format(i_bc), type='BCWall', parent=zone_bc)
      PT.new_GridLocation('FaceCenter', parent=bc_n)
      start, end = face_group_idx[i_bc], face_group_idx[i_bc+1]
      dn_face_bnd = end - start
      PT.new_PointList(value=face_group[start:end].reshape(1,dn_face_bnd), parent=bc_n)

      bc_distrib = par_utils.gather_and_shift(dn_face_bnd, comm, pdm_gnum_dtype)
      distrib    = bc_distrib[[i_rank, i_rank+1, n_rank]]
      MT.newDistribution({'Index' : distrib}, parent=bc_n)

      # > On remet BC + FamilyName
      solbc, famname = dicttag_to_bcinfo[i_bc]
      PT.add_child(bc_n, solbc)
      PT.add_child(bc_n, famname)

  # if edge_groups is not None:
  #   shift_bc = shift_elmt_surf - 1 if sections_surf is not None else shift_elmt_ridge - 1
  #   edge_group_idx = edge_groups['dgroup_elmt_idx']
  #   edge_group = shift_bc + edge_groups['dgroup_elmt']
  #   distri = np.empty(n_rank, dtype=edge_group.dtype)
  #   n_edge_group = edge_group_idx.shape[0] - 1

  #   for i_bc in range(n_edge_group):
  #     bc_n = PT.new_BC('dcube_ridge_{0}'.format(i_bc), type='BCWall', parent=zone_bc)
  #     PT.new_GridLocation('EdgeCenter', parent=bc_n)
  #     start, end = edge_group_idx[i_bc], edge_group_idx[i_bc+1]
  #     dn_edge_bnd = end - start
  #     PT.new_PointList(value=edge_group[start:end].reshape(1,dn_edge_bnd), parent=bc_n)

  #     bc_distrib = par_utils.gather_and_shift(dn_edge_bnd, comm, pdm_gnum_dtype)
  #     distrib    = bc_distrib[[i_rank, i_rank+1, n_rank]]
  #     MT.newDistribution({'Index' : distrib}, parent=bc_n)

  if vtx_groups is not None:
    shift_bc = shift_elmt_ridge - 1 if sections_ridge is not None else shift_elmt_corner - 1
    vtx_group_idx = vtx_groups['dgroup_elmt_idx']
    vtx_group = shift_bc + vtx_groups['dgroup_elmt']
    distri = np.empty(n_rank, dtype=vtx_group.dtype)
    n_vtx_group = vtx_group_idx.shape[0] - 1

    for i_bc in range(n_vtx_group):
      bc_n = PT.new_BC('dcube_corner_{0}'.format(i_bc), type='BCWall', parent=zone_bc)
      PT.new_GridLocation('Vertex', parent=bc_n)
      start, end = vtx_group_idx[i_bc], vtx_group_idx[i_bc+1]
      dn_vtx_bnd = end - start
      PT.new_PointList(value=vtx_group[start:end].reshape(1,dn_vtx_bnd), parent=bc_n)

      bc_distrib = par_utils.gather_and_shift(dn_vtx_bnd, comm, pdm_gnum_dtype)
      distrib    = bc_distrib[[i_rank, i_rank+1, n_rank]]
      MT.newDistribution({'Index' : distrib}, parent=bc_n)

  # > Distributions
  np_distrib_cell = par_utils.uniform_distribution(g_dims["n_cell_abs"], comm)

  distri_vtx     = vtx_data['np_vtx_distrib']
  np_distrib_vtx = distri_vtx[[i_rank, i_rank+1, n_rank]]

  MT.newDistribution({'Cell' : np_distrib_cell, 'Vertex' : np_distrib_vtx}, parent=dist_zone)

  for family in families:
    PT.add_child(dist_base, family)


  if not isotrop:
    # > FlowSolution
    cons = -100*np.ones(g_dims["n_vtx_abs"] * 7, dtype=np.double)
    PDM.read_solb(bytes(out_files['fld'], 'utf-8'), g_dims["n_vtx_abs"], 7, cons)

    cons = cons.reshape((7, cons.shape[0]//7), order='F')
    cons = cons.transpose()

    fs = PT.new_FlowSolution("FlowSolution#Init", loc='Vertex', parent=dist_zone)
    PT.new_DataArray("Density"                , cons[np_distrib_vtx[0]:np_distrib_vtx[1],0], parent=fs)
    PT.new_DataArray("MomentumX"              , cons[np_distrib_vtx[0]:np_distrib_vtx[1],1], parent=fs)
    PT.new_DataArray("MomentumY"              , cons[np_distrib_vtx[0]:np_distrib_vtx[1],2], parent=fs)
    PT.new_DataArray("MomentumZ"              , cons[np_distrib_vtx[0]:np_distrib_vtx[1],3], parent=fs)
    PT.new_DataArray("EnergyStagnationDensity", cons[np_distrib_vtx[0]:np_distrib_vtx[1],4], parent=fs)
    PT.new_DataArray("Mach"                   , cons[np_distrib_vtx[0]:np_distrib_vtx[1],5], parent=fs)

  return dist_tree


def meshb_to_cgns(out_files, dicttag_to_bcinfo, families, comm, isotrop=True):
  '''
  Reading a meshb file and conversion to CGNS norm.

  Arguments :
    - meshb_file        (str) : meshb file name
    - dicttag_to_bcinfo (dict): informations coming from cgns to meshb conversion
    - families          (list): list of families from the previous cgns
    - comm              (MPI) : MPI Communicator
    - isotrop           (bool): isotrop adaptation or not (read flds or not)

  '''
  # meshb -> dmesh_nodal # meshb -> dmesh_nodal -> cgns
  dmesh_nodal = PDM.meshb_to_dmesh_nodal(bytes(out_files['mesh'], 'utf-8'), comm, 1, 1)
  dist_tree   = dmesh_nodal_to_cgns(dmesh_nodal, comm, dicttag_to_bcinfo, families, out_files, isotrop)

  return dist_tree









def cgns_to_meshb(dist_tree, files, criterion):
  '''
  Dist_tree conversion to meshb format and writing.
  Arguments :
    - dist_tree (CGNSTree) : dist_tree to convert
    - files     (dict)     : file names for meshb files
    - files     (str)      : descriptor of the adaptation criterion
  '''
  dicttag_to_bcinfo = {}
  for zone in PT.get_all_Zone_t(dist_tree):
    
    # Coordinates
    cx  = PT.get_node_from_name(zone, "CoordinateX")[1]
    cy  = PT.get_node_from_name(zone, "CoordinateY")[1]
    cz  = PT.get_node_from_name(zone, "CoordinateZ")[1]
    

    # Gathering elements by dimension
    sorted_elts_by_dim = PT.Zone.get_ordered_elements_per_dim(zone)

    elmt_by_dim = list()
    for elmts in sorted_elts_by_dim:
      elmt_ec = list()
      for elmt in elmts:
        ec = PT.get_node_from_name(elmt, "ElementConnectivity")
        elmt_ec.append(ec[1])

      if(len(elmt_ec) > 1):
        elmt_by_dim.append(np.concatenate(elmt_ec))
      else:
        if(elmts != []):
          elmt_by_dim.append(elmt_ec[0])
        else:
          elmt_by_dim.append(elmt_ec)

    n_vtx   = PT.Zone.n_vtx(zone)
    n_tetra = elmt_by_dim[3].shape[0]//4
    n_tri   = elmt_by_dim[2].shape[0]//3
    try:
      n_edge  = elmt_by_dim[1].shape[0]//2
    except AttributeError:
      n_edge = 0


    # PointList BC to BC tag
    elmt_tag2 = -np.ones(n_tri, dtype=np.int32)
    edge_tag2 =  np.ones(n_edge, dtype=np.int32)

    n_tag = 0

    zone_bc     = PT.get_node_from_label(zone, 'ZoneBC_t')
    bcs         = PT.get_nodes_from_label(zone_bc, 'BC_t')
    bcs_to_elmt = list()
    tags        = list()
    for bc in bcs:
      gl = PT.get_node_from_name(bc, 'GridLocation')

      pl = PT.get_node_from_name(bc, 'PointList')
      size = pl[1].shape[1]

      bcs_to_elmt.append(pl[1][0, :])

      tags.append(np.ones(size, dtype='int32')* (n_tag+1))

      elmt_tag2[pl[1][0, :]-n_tetra-1] = n_tag+1

      solbc   = PT.get_node_from_name(bc, ".Solver#BC")
      famname = PT.get_node_from_name(bc, "FamilyName")
      dicttag_to_bcinfo[n_tag] = (solbc, famname)

      n_tag = n_tag +1

    bc_to_elmt  = np.concatenate(bcs_to_elmt)
    min_elmt    = np.min(bc_to_elmt)
    bc_to_elmt -= min_elmt
    bc_tags     = np.concatenate(tags)

    elmt_tag = np.take(bc_tags, bc_to_elmt)

    xyz       = np_utils.interweave_arrays([cx,cy,cz])
    vtx_tag   = np.zeros(n_vtx, dtype=np.int32)
    tetra_tag = np.zeros(n_tetra, dtype=np.int32)

    PDM.write_meshb(bytes(files["mesh"], 'utf-8'),
                    n_vtx, n_tetra, n_tri, n_edge,
                    xyz,            vtx_tag,
                    elmt_by_dim[3], tetra_tag,
                    elmt_by_dim[2], elmt_tag2,
                    elmt_by_dim[1], edge_tag2)


    # Write criterion file
    if criterion!='isotrop':
      fs = PT.get_node_from_name(zone, "FSolution#Vertex#EndOfRun")

      if   criterion=='mach_fld':
        mach = PT.get_node_from_name(fs  , "Mach")[1]
        PDM.write_solb(bytes(files["sol"], 'utf-8'), n_vtx, 1, mach)

      elif criterion=='mach_hess':
        mxx = PT.get_node_from_name(fs, "extrap_on(sym_grad(extrap_on(#0")[1]
        mxy = PT.get_node_from_name(fs, "extrap_on(sym_grad(extrap_on(#1")[1]
        mxz = PT.get_node_from_name(fs, "extrap_on(sym_grad(extrap_on(#2")[1]
        myy = PT.get_node_from_name(fs, "extrap_on(sym_grad(extrap_on(#3")[1]
        myz = PT.get_node_from_name(fs, "extrap_on(sym_grad(extrap_on(#4")[1]
        mzz = PT.get_node_from_name(fs, "extrap_on(sym_grad(extrap_on(#5")[1]

        met = np_utils.interweave_arrays([mxx,mxy,myy,mxz,myz,mzz])

        PDM.write_matsym_solb(bytes(files["sol"], 'utf-8'), n_vtx, met)

      # Fields to interpolate
      density = PT.get_node_from_name(fs, "Density")[1]
      momx    = PT.get_node_from_name(fs, "MomentumX")[1]
      momy    = PT.get_node_from_name(fs, "MomentumY")[1]
      momz    = PT.get_node_from_name(fs, "MomentumZ")[1]
      roe     = PT.get_node_from_name(fs, "EnergyStagnationDensity")[1]
      mach    = PT.get_node_from_name(fs, "Mach")[1]
      try:
        ronu  = PT.get_node_from_name(fs, "TurbulentSANuTildeDensity")[1]
      except TypeError:
        ronu  = np.zeros(density.shape[0], np.double)
      cons    = np_utils.interweave_arrays([density, momx, momy, momz, roe, mach, ronu])

      PDM.write_solb(bytes(files["fld"], 'utf-8'), n_vtx, 7, cons)


  families = PT.get_nodes_from_label(dist_tree, 'Family_t')

  return dicttag_to_bcinfo, families



