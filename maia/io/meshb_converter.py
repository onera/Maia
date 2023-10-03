import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT
import maia.utils.logging as mlog

from maia                         import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                   import np_utils, par_utils
from maia.factory.dcube_generator import _dmesh_nodal_to_cgns_zone

import numpy as np

import Pypdm.Pypdm as PDM


def get_tree_info(dist_tree, container_names):
  """
  Get tree informations such as bc_names and interpolated containers.
  """
  
  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone_n = zones[0]

  # > Get BCs infos
  bc_names = dict()

  for entity_name in ["EdgeCenter", "FaceCenter"]:
    is_entity_bc = lambda n :PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)==entity_name
    entity_bcs   = PT.get_children_from_predicates(zone_n, ['ZoneBC_t', is_entity_bc])
    bc_names[entity_name] = [PT.get_name(bc_n) for bc_n in entity_bcs]


  # > Container field names
  field_names = dict()
  for container_name in container_names:
    container = PT.get_node_from_name(zone_n, container_name)
    assert PT.Subset.GridLocation(container) == 'Vertex'
    field_names[container_name] = [PT.get_name(n) for n in PT.iter_children_from_label(container, 'DataArray_t')]

  return {"bc_names"    : bc_names,
          "field_names" : field_names}


def dmesh_nodal_to_cgns(dmesh_nodal, comm, tree_info, out_files):
  """
  Convert a dmesh_nodal mesh to CGNS format, according to initial dist_tree informations
  contained in ``tree_info``.
  """

  # > Generate dist_tree
  g_dims    = dmesh_nodal.dmesh_nodal_get_g_dims()
  cell_dim  = 3 if g_dims["n_cell_abs"]>0 else 2
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase(cell_dim=cell_dim, phy_dim=3, parent=dist_tree)
  dist_zone = _dmesh_nodal_to_cgns_zone(dmesh_nodal, comm)
  PT.add_child(dist_base, dist_zone)


  # > BCs
  vtx_groups  = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_CORNER)
  edge_groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_RIDGE)
  face_groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_SURFACIC)

  bc_names = tree_info['bc_names']
  def groups_to_bcs(elt_groups, zone_bc, location, shift_bc, comm):
    elt_group_idx = elt_groups['dgroup_elmt_idx']
    elt_group     = elt_groups['dgroup_elmt'] + shift_bc
    n_elt_group   = elt_group_idx.shape[0] - 1

    n_bc_init = len(bc_names[location]) if location in bc_names else 0
    n_new_bc  = n_elt_group - n_bc_init
    assert n_new_bc in [0,1], "Unknow tags in meshb file"

    for i_group in range(n_elt_group):
      if bc_names[location]:
        if i_group < n_new_bc:
          # name_bc   = {"Vertex":"vtx", "EdgeCenter":"edge", "FaceCenter":"face"}
          # bc_name = f"new_{name_bc[location]}_bc_{i_group+1}"
          continue # For now, skip BC detected in meshb but not provided in BC names
        else:
          bc_name = bc_names[location][i_group-n_new_bc]

        bc_n = PT.new_BC(bc_name, type='Null', loc=location, parent=zone_bc)
        start, end = elt_group_idx[i_group], elt_group_idx[i_group+1]
        dn_elt_bnd = end - start
        PT.new_PointList(value=elt_group[start:end].reshape((1,-1), order='F'), parent=bc_n)

        bc_distrib = par_utils.gather_and_shift(dn_elt_bnd, comm, pdm_gnum_dtype)
        MT.newDistribution({'Index' : par_utils.dn_to_distribution(dn_elt_bnd, comm)}, parent=bc_n)


  zone_bc = PT.new_ZoneBC(parent=dist_zone)
  range_per_dim = PT.Zone.get_elt_range_per_dim(dist_zone)

  if face_groups is not None:
    groups_to_bcs(face_groups, zone_bc, "FaceCenter", range_per_dim[3][1], comm)
  if edge_groups is not None:
    groups_to_bcs(edge_groups, zone_bc, "EdgeCenter", range_per_dim[2][1], comm)
  if vtx_groups  is not None:
    groups_to_bcs(vtx_groups,  zone_bc, "Vertex",     range_per_dim[1][1], comm)

  # > Add FlowSolution
  n_vtx = PT.Zone.n_vtx(dist_zone)
  distrib_vtx = PT.get_value(MT.getDistribution(dist_zone, "Vertex"))

  field_names = tree_info['field_names']
  n_itp_flds  = sum([len(fld_names) for fld_names in field_names.values()])
  if n_itp_flds!=0:
    all_fields = np.empty(n_vtx*n_itp_flds, dtype=np.double)
    PDM.read_solb(bytes(out_files['fld']), n_vtx, n_itp_flds, all_fields)

    i_fld = 0
    for container_name, fld_names in field_names.items():
      fs = PT.new_FlowSolution(container_name, loc='Vertex', parent=dist_zone)
      for fld_name in fld_names:
        # Deinterlace + select distributed section since everything has been read ...
        data = all_fields[i_fld::n_itp_flds][distrib_vtx[0]:distrib_vtx[1]] 
        PT.new_DataArray(fld_name, data, parent=fs) 
        i_fld += 1

  return dist_tree


def meshb_to_cgns(out_files, tree_info, comm):
  '''
  Reading a meshb file and conversion to CGNS norm.

  Arguments :
    - out_files         (dict): meshb file names
    - tree_info         (dict): initial dist_tree informations (bc_infos, interpolated field names)
    - comm              (MPI) : MPI Communicator
  '''
  mlog.info(f"Distributed read of meshb file...")
  start = time.time()
  
  # meshb -> dmesh_nodal -> cgns
  dmesh_nodal = PDM.meshb_to_dmesh_nodal(bytes(out_files['mesh']), comm, 1, 1)
  dist_tree   = dmesh_nodal_to_cgns(dmesh_nodal, comm, tree_info, out_files)
  
  end = time.time()
  dt_size     = sum(MT.metrics.dtree_nbytes(dist_tree))
  all_dt_size = comm.allreduce(dt_size, MPI.SUM)
  mlog.info(f"Read completed ({end-start:.2f} s) --"
            f" Size of dist_tree for current rank is {mlog.bsize_to_str(dt_size)}"
            f" (Σ={mlog.bsize_to_str(all_dt_size)})")

  return dist_tree






def cgns_to_meshb(dist_tree, files, metric_nodes, container_names):
  '''
  Dist_tree conversion to meshb format and writing.
  Arguments :
    - dist_tree       (CGNSTree) : dist_tree to convert
    - files           (dict)     : file names for meshb files
    - metric_nodes    (str)      : CGNS metric nodes
    - container_names (str)      : container_names to be interpolated
  '''

  dt_size = sum(MT.metrics.dtree_nbytes(dist_tree))
  mlog.info(f"Sequential write of a meshb file from a {mlog.bsize_to_str(dt_size)} dist_tree...")
  start = time.time()

  # > Monodomain only for now
  assert len(PT.get_all_Zone_t(dist_tree))==1

  for zone in PT.get_all_Zone_t(dist_tree):

    # > Coordinates
    cx, cy, cz = PT.Zone.coordinates(zone)

    # > Gathering elements by dimension
    elmt_by_dim = list()
    for elmts in PT.Zone.get_ordered_elements_per_dim(zone):
      elmt_ec = [np_utils.safe_int_cast(PT.get_node_from_name(elmt, "ElementConnectivity")[1], np.int32) for elmt in elmts]

      if(len(elmt_ec) > 1):
        elmt_by_dim.append(np.concatenate(elmt_ec))
      elif len(elmt_ec) == 1:
        elmt_by_dim.append(elmt_ec[0])
      else:
        elmt_by_dim.append(np.empty(0,dtype=np.int32))

    n_tetra = elmt_by_dim[3].size // 4
    n_tri   = elmt_by_dim[2].size // 3
    n_edge  = elmt_by_dim[1].size // 2
    n_vtx   = PT.Zone.n_vtx(zone)

    # > PointList BC to BC tag
    def bc_pl_to_bc_tag(list_of_bc, bc_tag, offset):
      for n_tag, bc_n in enumerate(list_of_bc):
        pl = PT.get_value(PT.get_node_from_name(bc_n, 'PointList'))[0]
        bc_tag[pl-offset-1] = n_tag + 1

    zone_bc = PT.get_child_from_label(zone, 'ZoneBC_t')

    tri_tag    = -np.ones(n_tri, dtype=np.int32)
    edge_tag   = -np.ones(n_edge, dtype=np.int32)
    if zone_bc is not None:
      # > Face BC_t
      is_face_bc = lambda n :PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n) == "FaceCenter"
      face_bcs   = PT.get_children_from_predicate(zone_bc, is_face_bc)
      n_face_tag = bc_pl_to_bc_tag(face_bcs, tri_tag, n_tetra)

      # > Edge BC_t
      is_edge_bc = lambda n :PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n) == "EdgeCenter"
      edge_bcs   = PT.get_children_from_predicate(zone_bc, is_edge_bc)
      n_edge_tag = bc_pl_to_bc_tag(edge_bcs, edge_tag, n_tetra+n_tri)

    is_3d = n_tetra!=0
    is_2d = n_tri  !=0
    if is_3d: 
      if (n_tri > 0 and (tri_tag < 0).any()) or (n_edge > 0 and (edge_tag < 0).any()):
        raise ValueError("Some Face or Edge elements do not belong to any BC")
    elif is_2d:
      tri_tag = np.zeros(n_tri, dtype=np.int32)
      is_face_bc = lambda n :PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n) == "FaceCenter"
      face_bcs   = PT.get_children_from_predicate(zone_bc, is_face_bc)
      n_face_tag = bc_pl_to_bc_tag(face_bcs, tri_tag, n_tetra)
      tri_tag +=1
      print(f'n_edge   = {n_edge}')
      print(f'edge_tag = {edge_tag}')
      if (n_edge > 0 and (edge_tag < 0).any()):
        raise ValueError("Some Face or Edge elements do not belong to any BC")
    else:
      raise ValueError("No tetrahedron or triangle Elements_t node could be found")


    # > Write meshb
    xyz       = np_utils.interweave_arrays([cx,cy,cz])
    vtx_tag   = np.zeros(n_vtx, dtype=np.int32)
    tetra_tag = np.zeros(n_tetra, dtype=np.int32)

    PDM.write_meshb(bytes(files["mesh"]),
                    n_vtx, n_tetra, n_tri, n_edge,
                    xyz,              vtx_tag,
                    elmt_by_dim[3], tetra_tag,
                    elmt_by_dim[2],   tri_tag,
                    elmt_by_dim[1],  edge_tag)



    n_metric_fld = len(metric_nodes)
    if n_metric_fld==1:
      metric_fld = PT.get_value(metric_nodes[0])
      PDM.write_solb(bytes(files["sol"]), n_vtx, 1, metric_fld)
    elif n_metric_fld==6:
      mxx = PT.get_value(metric_nodes[0])
      mxy = PT.get_value(metric_nodes[1])
      mxz = PT.get_value(metric_nodes[2])
      myy = PT.get_value(metric_nodes[3])
      myz = PT.get_value(metric_nodes[4])
      mzz = PT.get_value(metric_nodes[5])
      met = np_utils.interweave_arrays([mxx,mxy,myy,mxz,myz,mzz])
      PDM.write_matsym_solb(bytes(files["sol"]), n_vtx, met)


    # > Fields to interpolate
    fields_list = list()
    for container_name in container_names:
      container    = PT.get_node_from_name(zone, container_name)
      fields_list += [PT.get_value(n) for n in PT.get_children_from_label(container, 'DataArray_t')]
    if len(fields_list)>0:
      fields_array = np_utils.interweave_arrays(fields_list)
      PDM.write_solb(bytes(files["fld"]), n_vtx, len(fields_list), fields_array)


  end = time.time()
  mlog.info(f"Write of meshb file completed ({end-start:.2f} s)")



