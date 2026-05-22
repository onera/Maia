import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT
from   maia.pytree.maia   import metrics, pdm_elts
import maia.utils.logging as mlog

from maia                         import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                   import np_utils, par_utils
from maia.factory.dcube_generator import _dmesh_nodal_to_cgns_zone

import numpy as np

import Pypdm.Pypdm as PDM

from packaging.version import Version
from Pypdm.Pypdm import __version__ as _PDM_VERSION
PDM_VERSION = Version(_PDM_VERSION)
PDM_NEW_WRITER_API = PDM_VERSION >= Version('2.8')

def get_tree_info(dist_tree, containers_name):
  """
  Get tree informations such as bc_names and interpolated containers.
  """

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone_n = zones[0]

  # > Get BCs infos
  bc_names = dict()
  for entity_name in ["EdgeCenter", "FaceCenter", "CellCenter"]:
    is_entity_bc = PT.pred.is_bc_of_location(entity_name)
    entity_bcs   = PT.get_children_from_predicates(zone_n, ['ZoneBC_t', is_entity_bc])
    bc_names[entity_name] = [PT.get_name(bc_n) for bc_n in entity_bcs]

  # > Container field names
  field_names = dict()
  for container_name in containers_name:
    container = PT.find_node_from_name(zone_n, container_name)
    assert PT.Container.GridLocation(container) == 'Vertex'
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
  cell_groups = dmesh_nodal.dmesh_nodal_get_group(PDM._PDM_GEOMETRY_KIND_VOLUMIC)

  vtx_data = dmesh_nodal.dmesh_nodal_get_vtx_tag(comm)
  vtx_tag = vtx_data['np_vtx_tag']


  bc_names = tree_info['bc_names']
  def groups_to_bcs(elt_groups, zone_bc, location, shift_bc, comm):
    elt_group_idx = elt_groups['dgroup_elmt_idx']
    elt_group     = elt_groups['dgroup_elmt'] + shift_bc
    n_elt_group   = elt_group_idx.shape[0] - 1

    n_bc_init = len(bc_names[location]) if location in bc_names else 0
    n_new_bc  = n_elt_group - n_bc_init

    assert n_new_bc in [0,1], f"Unknow tags in meshb file ({location})"

    for i_group in range(n_elt_group):
      if bc_names[location] or edge_groups is not None:
        if i_group < n_new_bc:
          name_bc   = {"Vertex":"vtx", "EdgeCenter":"edge", "FaceCenter":"face", "CellCenter":"cell"}
          bc_name = f"feflo_{name_bc[location]}_bc_{i_group}"
          # continue # For now, skip BC detected in meshb but not provided in BC names
        else:
          bc_name = bc_names[location][i_group-n_new_bc]

        bc_n = PT.new_BC(bc_name, type='Null', loc=location, parent=zone_bc)
        start, end = elt_group_idx[i_group], elt_group_idx[i_group+1]
        dn_elt_bnd = end - start
        PT.new_IndexArray(value=elt_group[start:end].reshape((1,-1), order='F'), parent=bc_n)

        MT.new_Distribution({'Index' : par_utils.dn_to_distribution(dn_elt_bnd, comm)}, parent=bc_n)


  zone_bc = PT.new_ZoneBC(parent=dist_zone)
  range_per_dim = PT.Zone.get_elt_range_per_dim(dist_zone)

  locs = ['Vertex', 'EdgeCenter', 'FaceCenter', 'CellCenter']
  for i, group in enumerate([cell_groups, face_groups, edge_groups, vtx_groups]):
    dim = 3 - i
    if dim <= cell_dim and group is not None:
      loc = 'CellCenter' if dim == cell_dim else locs[dim]
      groups_to_bcs(group, zone_bc, loc, range_per_dim[dim][0]-1, comm)
   
  # > Add FlowSolution for vtx tag
  PT.new_DiscreteData('maia_topo', loc='Vertex', fields={'vtx_tag':vtx_tag}, parent=dist_zone)

  # > Add FlowSolution
  n_vtx = PT.Zone.n_vtx(dist_zone)
  distrib_vtx = MT.Zone.vtx_distribution(dist_zone)

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

  # > Add Metric
  """
  metric_names = tree_info['metric_names']
  n_itp_metric = sum([len(met_names) for met_names in metric_names.values()])

  if n_itp_metric != 0:
      all_metric = np.empty(n_vtx*n_itp_metric, dtype=np.double)
      PDM.read_solb(bytes(out_files['sol']), n_vtx, n_itp_metric, all_metric)

      i_fld = 0
      for container_name, met_names in metric_names.items():
          fs = PT.new_FlowSolution(container_name, loc='Vertex', parent=dist_zone)

          for met_name in met_names:
              # Deinterlace + select distributed section since everything has been read ...
              data = all_metric[i_fld::n_itp_metric][distrib_vtx[0]:distrib_vtx[1]]
              PT.new_DataArray(met_name, data, parent=fs)
              i_fld += 1
  """

  return dist_tree


def meshb_to_cgns(out_files, tree_info, comm, fix_orientation_2d=False, fix_orientation_3d=True):
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
  file_name = bytes(out_files["mesh"], 'utf-8') if isinstance(out_files["mesh"], str)\
         else bytes(out_files["mesh"])
  dmesh_nodal = PDM.meshb_to_dmesh_nodal(file_name, comm, fix_orientation_2d, fix_orientation_3d)
  dist_tree   = dmesh_nodal_to_cgns(dmesh_nodal, comm, tree_info, out_files)

  end = time.time()
  dt_size     = sum(metrics.dtree_nbytes(dist_tree))
  all_dt_size = comm.allreduce(dt_size, MPI.SUM)
  mlog.info(f"Read completed ({end-start:.2f} s) --"
            f" Size of dist_tree for current rank is {mlog.bsize_to_str(dt_size)}"
            f" (Σ={mlog.bsize_to_str(all_dt_size)})")

  return dist_tree




# > PointList BC to BC tag
def _bc_pl_to_bc_tag(list_of_bc, elts_tag, elts_ranges, elts_idx, constraint_bcs, constraint_tags):
  for n_tag, bc_n in enumerate(list_of_bc):
    pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0]

    for pdm_elt_t, (elt_idx, elt_ranges) in enumerate(zip(elts_idx, elts_ranges)):
      for i_elt, elt_range in enumerate(elt_ranges): # Some element type can a multiple node, so multiple elt_range
        if elt_range is not None and pdm_elt_t!=PDM._PDM_MESH_NODAL_POINT: # Beware of vtx range overlapping with element range
          pl_in_elmt = pl[np.logical_and( np.greater_equal(pl, elt_range[0]),
                                          np.less_equal   (pl, elt_range[1]))] - elt_range[0] + elt_idx[i_elt]
          elts_tag[pdm_elt_t][pl_in_elmt] = n_tag + 1

    bc_name = PT.get_name(bc_n)
    if constraint_bcs is not None and\
      PT.get_name(bc_n) not in constraint_bcs:
      constraint_tags.append(str(n_tag + 1))

def _bc_pl_to_bc_tag_vtx(list_of_bc, vtx_tag):
  for n_tag, bc_n in enumerate(list_of_bc):
    pl = PT.get_value(PT.get_node_from_name(bc_n, 'PointList'))[0]
    vtx_tag[pl-1] = pl


def cgns_to_meshb(dist_tree, files, metric_nodes, containers_name, constraints):
  '''
  Dist_tree conversion to meshb format and writing.
  Arguments :
    - dist_tree       (CGNSDistTree) : dist_tree to convert
    - files           (dict)         : file names for meshb files
    - metric_nodes    (str)          : CGNS metric nodes
    - containers_name (str)          : containers_name to be interpolated
  '''

  dt_size = sum(MT.metrics.dtree_nbytes(dist_tree))
  mlog.info(f"Sequential write of a meshb file from a {mlog.bsize_to_str(dt_size)} dist_tree...")
  start = time.time()

  # > Monodomain only for now
  assert len(PT.get_all_Zone_t(dist_tree))==1

  for zone in PT.get_all_Zone_t(dist_tree):

    is_3d = False
    is_2d = False
    phydim = PT.Zone.PhysicalDimension(zone)

    # > Coordinates
    cx, cy, cz = PT.Zone.coordinates(zone)
    if cz is None:
      cz = np.zeros_like(cx)

    # > Gathering elements by type
    #   For each element type, get info from element nodes of this type:
    #      - get number of element
    #      - connectivity
    #      - element range
    pdm_n_elmt     = [[]   for i_elmt in range(PDM._PDM_MESH_NODAL_N_ELEMENT_TYPES)]
    pdm_elmt_vtx   = [[]   for i_elmt in range(PDM._PDM_MESH_NODAL_N_ELEMENT_TYPES)]
    pdm_elmt_tag   = [None for i_elmt in range(PDM._PDM_MESH_NODAL_N_ELEMENT_TYPES)]
    pdm_elmt_range = [[]   for i_elmt in range(PDM._PDM_MESH_NODAL_N_ELEMENT_TYPES)]

    for elmts in PT.Zone.get_ordered_elements_per_dim(zone):
      for elmt_n in elmts:
        elmt_name  = PT.Element.Type(elmt_n)
        elmt_pdm_t = pdm_elts.cgns_elt_name_to_pdm_element_type(elmt_name)

        pdm_n_elmt    [elmt_pdm_t].append(PT.Element.Size(elmt_n))
        pdm_elmt_vtx  [elmt_pdm_t].append(np_utils.safe_int_cast(PT.get_node_from_name(elmt_n, "ElementConnectivity")[1], pdm_gnum_dtype))
        pdm_elmt_range[elmt_pdm_t].append(PT.Element.Range(elmt_n))

        if PT.Element.Size(elmt_n)!=0:
          is_3d = PT.Element.Dimension(elmt_n)==3 # Works cause elements are ordered by dim
          is_2d = PT.Element.Dimension(elmt_n)==2

    # > Reduce information and init tag
    #   For each element type, reduce info from element nodes of this type:
    #      - total number of elements
    #      - element number in node idx
    #      - concatenate node connectivities
    #      - initialize tag
    pdm_elmt_idx = [0 for i_elmt in range(PDM._PDM_MESH_NODAL_N_ELEMENT_TYPES)]
    for elmt_pdm_t in range(PDM._PDM_MESH_NODAL_N_ELEMENT_TYPES):
      pdm_elmt_idx[elmt_pdm_t] = np_utils.sizes_to_indices(pdm_n_elmt[elmt_pdm_t], dtype=np.int32)
      pdm_n_elmt  [elmt_pdm_t] = pdm_elmt_idx[elmt_pdm_t][-1]
      pdm_elmt_vtx[elmt_pdm_t] = np_utils.concatenate_np_arrays(pdm_elmt_vtx[elmt_pdm_t], dtype=pdm_gnum_dtype)[1]

      if is_3d:
        if elmt_pdm_t in [PDM._PDM_MESH_NODAL_BAR2, PDM._PDM_MESH_NODAL_TRIA3, PDM._PDM_MESH_NODAL_QUAD4]:
          pdm_elmt_tag[elmt_pdm_t] = -np.ones (pdm_n_elmt[elmt_pdm_t], dtype=np.int32)
        else:
          pdm_elmt_tag[elmt_pdm_t] =  np.zeros(pdm_n_elmt[elmt_pdm_t], dtype=np.int32)
      if is_2d:
        if elmt_pdm_t in [PDM._PDM_MESH_NODAL_BAR2]:
          pdm_elmt_tag[elmt_pdm_t] = -np.ones (pdm_n_elmt[elmt_pdm_t], dtype=np.int32)
        else:
          pdm_elmt_tag[elmt_pdm_t] =  np.ones(pdm_n_elmt[elmt_pdm_t], dtype=np.int32)



    pdm_n_elmt  [PDM._PDM_MESH_NODAL_POINT] = PT.Zone.n_vtx(zone)
    pdm_elmt_tag[PDM._PDM_MESH_NODAL_POINT] = np.zeros(PT.Zone.n_vtx(zone), dtype=np.int32)

    constraint_tags = {'CellCenter':[],
                       'FaceCenter':[],
                       'EdgeCenter':[]}

    # > Tags initialization
    zone_bc = PT.get_child_from_label(zone, 'ZoneBC_t')

    if zone_bc is not None:
      # > Cell BC_t
      is_cell_bc = PT.pred.is_bc_of_location('CellCenter')
      cell_bcs   = PT.get_children_from_predicate(zone_bc, is_cell_bc)
      n_cell_tag = _bc_pl_to_bc_tag(cell_bcs, pdm_elmt_tag, pdm_elmt_range, pdm_elmt_idx,
                                      constraints, constraint_tags["CellCenter"])

      # > Face BC_t
      is_face_bc = PT.pred.is_bc_of_location('FaceCenter')
      face_bcs   = PT.get_children_from_predicate(zone_bc, is_face_bc)
      n_face_tag = _bc_pl_to_bc_tag(face_bcs, pdm_elmt_tag, pdm_elmt_range, pdm_elmt_idx,
                                      constraints, constraint_tags["FaceCenter"])

      # > Edge BC_t
      is_edge_bc = PT.pred.is_bc_of_location('EdgeCenter')
      edge_bcs   = PT.get_children_from_predicate(zone_bc, is_edge_bc)
      n_edge_tag = _bc_pl_to_bc_tag(edge_bcs, pdm_elmt_tag, pdm_elmt_range, pdm_elmt_idx,
                                      constraints, constraint_tags["EdgeCenter"])

      # > Vertices BC_t
      is_vtx_bc  = PT.pred.is_bc_of_location('Vertex')
      vtx_bcs   = PT.get_children_from_predicate(zone_bc, is_vtx_bc)
      n_vtx_tag = _bc_pl_to_bc_tag_vtx(vtx_bcs, pdm_elmt_tag[PDM._PDM_MESH_NODAL_POINT])


    if is_3d:
      for pdm_elt_t in [PDM._PDM_MESH_NODAL_BAR2, PDM._PDM_MESH_NODAL_TRIA3, PDM._PDM_MESH_NODAL_QUAD4]:
        if (pdm_n_elmt[pdm_elt_t] > 0 and (pdm_elmt_tag[pdm_elt_t] < 0).any()):
          raise ValueError("Some Face or Edge elements do not belong to any BC")
    elif is_2d:
      if (pdm_n_elmt[PDM._PDM_MESH_NODAL_BAR2] > 0 and (pdm_elmt_tag[PDM._PDM_MESH_NODAL_BAR2] < 0).any()):
        raise ValueError("Some Face or Edge elements do not belong to any BC")
    else:
      raise ValueError("No tetrahedron or triangle Elements_t node could be found")


    # > Write meshb
    xyz       = np_utils.interweave_arrays([cx,cy,cz])
    file_name = bytes(files["mesh"], 'utf-8') if isinstance(files["mesh"], str)\
           else bytes(files["mesh"])

    dtype = pdm_gnum_dtype if PDM_NEW_WRITER_API else np.int32
    pdm_n_elmt = np.array(pdm_n_elmt, dtype=dtype)
    PDM.write_meshb(file_name,
                    pdm_n_elmt, pdm_elmt_tag,
                    pdm_elmt_vtx, xyz)

    n_metric_fld = len(metric_nodes)
    if n_metric_fld==1:
      metric_fld = PT.get_value(metric_nodes[0])
      if PDM_NEW_WRITER_API:
        PDM.write_solb(bytes(files["sol"]), phydim, pdm_n_elmt[PDM._PDM_MESH_NODAL_POINT], 1, metric_fld)
      else:
        PDM.write_solb(bytes(files["sol"]), pdm_n_elmt[PDM._PDM_MESH_NODAL_POINT], 1, metric_fld)
    elif n_metric_fld==6:
      mxx = PT.get_value(metric_nodes[0])
      mxy = PT.get_value(metric_nodes[1])
      mxz = PT.get_value(metric_nodes[2])
      myy = PT.get_value(metric_nodes[3])
      myz = PT.get_value(metric_nodes[4])
      mzz = PT.get_value(metric_nodes[5])
      met = np_utils.interweave_arrays([mxx,mxy,mxz,myy,myz,mzz])
      if PDM_NEW_WRITER_API:
        PDM.write_matsym_solb(bytes(files["sol"]), phydim, pdm_n_elmt[PDM._PDM_MESH_NODAL_POINT], met)
      else:
        PDM.write_matsym_solb(bytes(files["sol"]), pdm_n_elmt[PDM._PDM_MESH_NODAL_POINT], met)


    # > Fields to interpolate
    fields_list = list()
    for container_name in containers_name:
      container    = PT.get_node_from_name(zone, container_name)
      fields_list += [PT.get_value(n) for n in PT.get_children_from_label(container, 'DataArray_t')]
    if len(fields_list)>0:
      fields_array = np_utils.interweave_arrays(fields_list)
      if PDM_NEW_WRITER_API:
        PDM.write_solb(bytes(files["fld"]), phydim, pdm_n_elmt[PDM._PDM_MESH_NODAL_POINT], len(fields_list), fields_array)
      else:
        PDM.write_solb(bytes(files["fld"]), pdm_n_elmt[PDM._PDM_MESH_NODAL_POINT], len(fields_list), fields_array)


  end = time.time()
  mlog.info(f"Write of meshb file completed ({end-start:.2f} s)")

  return constraint_tags
