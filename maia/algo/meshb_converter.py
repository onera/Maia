import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT
import maia.utils.logging as mlog

from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils import np_utils, par_utils, layouts

import numpy as np

import Pypdm.Pypdm as PDM



def _add_sections_to_zone(dist_zone, section, shift_elmt, comm):
  """
  Add distributed element node to a dist_zone.
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


def dmesh_nodal_to_cgns(dmesh_nodal, comm, tree_info, dicttag_to_bcinfo, families, out_files, isotrop):
  """
  Convert a dmesh_nodal mesh to CGNS format, according to initial dist_tree informations
  contained in ``tree_info``, ``dicttag_to_bc_info``, ``families``.
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
  dim = 3 if g_dims["n_cell_abs"]>0 else 2
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase(name=tree_info["Base"], cell_dim=dim, phy_dim=3, parent=dist_tree)

  if g_dims["n_cell_abs"] > 0:
    dist_zone = PT.new_Zone(name="Zone", size=[[g_dims["n_vtx_abs"], g_dims["n_cell_abs"], 0]],
                            type='Unstructured', parent=dist_base)
  else:
    dist_zone = PT.new_Zone(name="Zone", size=[[g_dims["n_vtx_abs"], g_dims["n_face_abs"], 0]],
                            type='Unstructured', parent=dist_base)


  # > Grid coordinates
  vtx_data   = dmesh_nodal.dmesh_nodal_get_vtx(comm)
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
  def groups_to_bcs(elt_groups, zone_bc, location, shift_bc, comm):
    elt_group_idx = elt_groups['dgroup_elmt_idx']
    elt_group     = elt_groups['dgroup_elmt'] + shift_bc
    distri = np.empty(n_rank, dtype=elt_group.dtype)
    n_elt_group = elt_group_idx.shape[0] - 1

    for i_group in range(n_elt_group):
      bc_name = dicttag_to_bcinfo[location][i_group+1]["BC"]
      famname = dicttag_to_bcinfo[location][i_group+1]["Family"]
      
      bc_n = PT.new_BC(bc_name, type='FamilySpecified', loc=location, parent=zone_bc)
      start, end  = elt_group_idx[i_group], elt_group_idx[i_group+1]
      dn_elt_bnd = end - start
      PT.new_PointList(value=elt_group[start:end].reshape(1,dn_elt_bnd), parent=bc_n)

      bc_distrib = par_utils.gather_and_shift(dn_elt_bnd, comm, pdm_gnum_dtype)
      distrib    = bc_distrib[[i_rank, i_rank+1, n_rank]]
      MT.newDistribution({'Index' : distrib}, parent=bc_n)

      PT.add_child(bc_n, famname)

  zone_bc = PT.new_ZoneBC(parent=dist_zone)

  if face_groups is not None:
    shift_bc = shift_elmt_vol - 1 if sections_vol is not None else shift_elmt_surf - 1
    groups_to_bcs(face_groups, zone_bc, "FaceCenter", shift_bc, comm)

  if edge_groups is not None:
    shift_bc = shift_elmt_surf - 1 if sections_surf is not None else shift_elmt_ridge - 1
    groups_to_bcs(edge_groups, zone_bc, "EdgeCenter", shift_bc, comm)

  if vtx_groups is not None:
    shift_bc = shift_elmt_ridge - 1 if sections_ridge is not None else shift_elmt_corner - 1
    groups_to_bcs(vtx_groups, zone_bc, "Vertex", shift_bc, comm)


  # > Distributions
  np_distrib_cell = par_utils.uniform_distribution(g_dims["n_cell_abs"], comm)

  distri_vtx     = vtx_data['np_vtx_distrib']
  np_distrib_vtx = distri_vtx[[i_rank, i_rank+1, n_rank]]

  MT.newDistribution({'Cell' : np_distrib_cell, 'Vertex' : np_distrib_vtx}, parent=dist_zone)

  for family in families:
    PT.add_child(dist_base, family)


  # > FlowSolution
  if not isotrop:
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


def meshb_to_cgns(out_files, tree_info, dicttag_to_bcinfo, families, comm, isotrop=True):
  '''
  Reading a meshb file and conversion to CGNS norm.

  Arguments :
    - out_files         (dict): meshb file names
    - tree_info         (dict): initial dist_tree informations (base and zone names)
    - dicttag_to_bcinfo (dict): bc informations coming from cgns to meshb conversion
    - families          (list): list of families from the previous cgns
    - comm              (MPI) : MPI Communicator
    - isotrop           (bool): isotrop adaptation or not (read flds or not)

  '''
  mlog.info(f"meshb to CGNS dist_tree conversion...")
  start = time.time()
  
  # meshb -> dmesh_nodal # meshb -> dmesh_nodal -> cgns
  dmesh_nodal = PDM.meshb_to_dmesh_nodal(bytes(out_files['mesh'], 'utf-8'), comm, 1, 1)
  dist_tree   = dmesh_nodal_to_cgns(dmesh_nodal, comm, tree_info, dicttag_to_bcinfo, families, out_files, isotrop)
  
  end = time.time()
  mlog.info(f"meshb to CGNS conversion completed ({end-start:.2f} s) --")

  return dist_tree






def cgns_to_meshb(dist_tree, files, metric):
  '''
  Dist_tree conversion to meshb format and writing.
  Arguments :
    - dist_tree (CGNSTree) : dist_tree to convert
    - files     (dict)     : file names for meshb files
    - metric    (str)      : descriptor of the adaptation metric
  '''

  mlog.info(f"CGNS to meshb dist_tree conversion...")
  start = time.time()

  dicttag_to_bcinfo = {"CellCenter": dict(),
                       "FaceCenter": dict(),
                       "EdgeCenter": dict(),
                       "Vertex"    : dict()}
  
  base_name = PT.get_name(PT.get_child_from_label(dist_tree, 'CGNSBase_t'))

  for zone in PT.get_all_Zone_t(dist_tree):
    zone_name = PT.get_name(zone) # Just one zone at this time

    # > Coordinates
    cx  = PT.get_node_from_name(zone, "CoordinateX")[1]
    cy  = PT.get_node_from_name(zone, "CoordinateY")[1]
    cz  = PT.get_node_from_name(zone, "CoordinateZ")[1]

    # > Gathering elements by dimension
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
          elmt_by_dim.append(np.empty(0,dtype=np.int32))

    n_vtx   = PT.Zone.n_vtx(zone)
    try:
      n_tetra = elmt_by_dim[3].shape[0]//4
    except AttributeError:
      n_tetra = 0
    n_tri   = elmt_by_dim[2].shape[0]//3
    try:
      n_edge  = elmt_by_dim[1].shape[0]//2
    except AttributeError:
      n_edge = 0

    mlog.info(f" + n_vtx   = {n_vtx   }")
    mlog.info(f" + n_tetra = {n_tetra }")
    mlog.info(f" + n_tri   = {n_tri   }")
    mlog.info(f" + n_edge  = {n_edge  }")

    # > PointList BC to BC tag
    def bc_pl_to_bc_tag(list_of_bc, bc_tag, dicttag_to_bcinfo, offset):
      n_tag = 0
      for bc_n in list_of_bc:
        gl = PT.get_value(PT.get_node_from_name(bc_n, 'GridLocation'))
        pl = PT.get_value(PT.get_node_from_name(bc_n, 'PointList'))[0]

        bc_tag[pl-offset-1] = n_tag+1

        n_tag = n_tag +1
        bc_name = PT.get_name(bc_n)
        famname = PT.get_node_from_name(bc_n, "FamilyName")
        dicttag_to_bcinfo[n_tag] = {"BC":bc_name, "Family":famname}

    zone_bc     = PT.get_child_from_label(zone, 'ZoneBC_t')

    # > Face BC_t
    tri_tag    = -np.ones(n_tri, dtype=np.int32)
    is_face_bc = lambda n :PT.get_label(n)=='BC_t' and \
                           PT.get_value(PT.get_child_from_name(n, "GridLocation"))=="FaceCenter"
    face_bcs   = PT.get_children_from_predicate(zone_bc, is_face_bc)
    n_face_tag = bc_pl_to_bc_tag(face_bcs, tri_tag, dicttag_to_bcinfo["FaceCenter"], n_tetra)

    # > Edge BC_t
    edge_tag   = -np.ones(n_edge, dtype=np.int32)
    is_edge_bc = lambda n :PT.get_label(n)=='BC_t' and \
                           PT.get_value(PT.get_child_from_name(n, "GridLocation"))=="EdgeCenter"
    edge_bcs   = PT.get_children_from_predicate(zone_bc, is_edge_bc)
    n_edge_tag = bc_pl_to_bc_tag(edge_bcs, edge_tag, dicttag_to_bcinfo["EdgeCenter"], n_tetra+n_tri)
   
    # > Write meshb
    xyz       = np_utils.interweave_arrays([cx,cy,cz])
    vtx_tag   = np.zeros(n_vtx, dtype=np.int32)
    tetra_tag = np.zeros(n_tetra, dtype=np.int32)

    PDM.write_meshb(bytes(files["mesh"], 'utf-8'),
                    n_vtx, n_tetra, n_tri, n_edge,
                    xyz,              vtx_tag,
                    elmt_by_dim[3], tetra_tag,
                    elmt_by_dim[2],   tri_tag,
                    elmt_by_dim[1],  edge_tag)


    # Write metric file
    if metric!='isotrop':
      fs = PT.get_node_from_name(zone, "FSolution#Vertex#EndOfRun")

      if   metric=='mach_fld':
        mach = PT.get_node_from_name(fs  , "Mach")[1]
        PDM.write_solb(bytes(files["sol"], 'utf-8'), n_vtx, 1, mach)

      elif metric=='mach_hess':
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

  tree_info = {"Base":base_name, "Zone":zone_name}
  families  = PT.get_nodes_from_label(dist_tree, 'Family_t')

  end = time.time()
  mlog.info(f"CGNS to meshb conversion completed ({end-start:.2f} s) --")
  return tree_info, dicttag_to_bcinfo, families




