import numpy              as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                 import npy_pdm_gnum_dtype     as pdm_gnum_dtype
from maia.utils           import py_utils, s_numbering, pr_utils, par_utils
from maia.utils           import logging as mlog
from maia.utils.numbering import range_to_slab          as HFR2S
from maia.transfer.protocols import GlobalMultiIndexer

from .matching_jns_tools import gc_is_reference, add_joins_donor_name, copy_donor_subset
from .connectivity_utils import cell_vtx_connectivity_S
from .ngons_to_elements  import _collected_shifted_pl, _update_pl

def get_output_loc(request_dict, s_node):
  """Retrieve output location from the node if not provided in argument"""
  label_to_key = {'BC_t' : 'BC_t', 'GridConnectivity1to1_t' : 'GC_t', 'GridConnectivity_t': 'GC_t'}
  out_loc = request_dict.get(label_to_key[PT.get_label(s_node)], None)
  if out_loc is None:
    out_loc = PT.Subset.GridLocation(s_node)
    if 'FaceCenter' in out_loc: #Remove 'I', 'J', or 'K' for FaceCenter
      out_loc = 'FaceCenter'
    if 'EdgeCenter' in out_loc: #Remove 'I', 'J' for EdgeCenter
      out_loc = 'EdgeCenter'
  if isinstance(out_loc, str):
    out_loc = [out_loc]
  return out_loc

def _s_location(loc, bnd_axis):
  if loc == 'EdgeCenter':
    return ['I', 'J'][bnd_axis] + 'EdgeCenter'
  elif loc == 'FaceCenter':
    return ['I', 'J', 'K'][bnd_axis] + 'FaceCenter'
  else:
    return loc

###############################################################################
def n_face_per_dir(n_vtx):
  """ Compute the number of faces in each direction from the number of vtx in each direction """
  return (n_vtx[0]*(n_vtx[1]-1)*(n_vtx[2]-1),
          n_vtx[1]*(n_vtx[0]-1)*(n_vtx[2]-1),
          n_vtx[2]*(n_vtx[0]-1)*(n_vtx[1]-1))

def n_edge_per_dir(n_vtx):
  """ Compute the number of edges in each direction from the number of vtx in each direction """
  return (n_vtx[0]*(n_vtx[1]-1),
          n_vtx[1]*(n_vtx[0]-1))
###############################################################################

def generate_all_bnd_bcs(n_vtx, dtype):
  if len(n_vtx) == 2:
    nv0, nv1 = n_vtx
    point_range_list = [[[  1,  1], [  1,nv1]],
                        [[nv0,nv0], [  1,nv1]],
                        [[  1,nv0], [  1,  1]],
                        [[  1,nv0], [nv1,nv1]]]
  if len(n_vtx) == 3:
    nv0, nv1, nv2 = n_vtx
    point_range_list = [[[  1,  1], [  1,nv1], [  1,nv2]],
                        [[nv0,nv0], [  1,nv1], [  1,nv2]],
                        [[  1,nv0], [  1,  1], [  1,nv2]],
                        [[  1,nv0], [nv1,nv1], [  1,nv2]],
                        [[  1,nv0], [  1,nv1], [  1,  1]],
                        [[  1,nv0], [  1,nv1], [nv2,nv2]]]
    
  return [PT.new_BC(point_range=np.array(pr, dtype=dtype)) for pr in point_range_list]



###############################################################################
def bc_s_to_bc_u(bc_s, n_vtx_zone, output_loc, i_rank, n_rank):
  """
  Convert a structured bc into a non structured bc. This consist mainly in
  deducing a point_list array from the point_range. The support of the output
  point_list must be specified throught output_loc arg (which can be one of
  'Vertex', 'FaceCenter', 'CellCenter').
  """
  input_loc = PT.Subset.GridLocation(bc_s)
  point_range = PT.get_value(PT.get_child_from_name(bc_s, 'PointRange'))

  bnd_axis = PT.Subset.normal_axis(bc_s)
  #Compute slabs from attended location (better load balance)
  bc_size = pr_utils.transform_bnd_pr_size(point_range, input_loc, output_loc)
  bc_range = py_utils.uniform_distribution_at(bc_size.prod(), i_rank, n_rank)
  bc_slabs = HFR2S.compute_slabs(bc_size, bc_range)
  if len(n_vtx_zone) == 2:
    bc_slabs = [slab[:2] for slab in bc_slabs]

  shift = pr_utils.normal_index_shift(point_range, n_vtx_zone, bnd_axis, input_loc, output_loc)
  #Prepare sub pointRanges from slabs
  sub_pr_list = [np.asarray(slab, point_range.dtype) for slab in bc_slabs]
  for sub_pr in sub_pr_list:
    sub_pr[:,0] += point_range[:,0]
    sub_pr[:,1] += point_range[:,0] - 1
    sub_pr[bnd_axis,:] += shift

  _loc = _s_location(output_loc, bnd_axis)
  point_list = pr_utils.compute_pointList_from_pointRanges(sub_pr_list, n_vtx_zone, _loc, dtype=point_range.dtype)

  bc_u = PT.new_node(PT.get_name(bc_s), PT.get_label(bc_s), PT.get_value(bc_s))
  PT.new_GridLocation(output_loc, parent=bc_u)
  PT.new_IndexArray(value=point_list, parent=bc_u)

  # Manage datasets -- Data is already distributed, we just have to retrive the PointList
  # of the corresponding elements following same procedure than BCs
  for bcds in PT.iter_children_from_label(bc_s, 'BCDataSet_t'):
    ds_point_range = PT.get_child_from_name(bcds, 'PointRange')
    is_related = ds_point_range is None
    if not is_related: #BCDS has its own location / pr
      ds_distri = MT.getDistribution(bcds, 'Index')[1]
    if is_related: #BCDS has same location / pr than bc
      ds_point_range = PT.get_child_from_name(bc_s, 'PointRange')
      ds_distri = MT.getDistribution(bc_s, 'Index')[1]
    ds_loc = PT.BCDataSet.GridLocation(bcds, bc_s)
    ds_size = PT.PointRange.SizePerIndex(ds_point_range)
    ds_slabs = HFR2S.compute_slabs(ds_size, ds_distri[0:2])
    ds_sub_pr_list = [np.asarray(slab, ds_point_range[1].dtype) for slab in ds_slabs]
    if len(n_vtx_zone) == 2:
      ds_sub_pr_list = [slab[:2] for slab in ds_sub_pr_list]
    for sub_pr in ds_sub_pr_list:
      sub_pr[:,0] += ds_point_range[1][:,0]
      sub_pr[:,1] += ds_point_range[1][:,0] - 1
    ds_output_loc = ds_loc[1:] if ds_loc[0] in 'IJK' else ds_loc

    if not (is_related and ds_output_loc == output_loc): #Otherwise, point list has already been computed
      _loc = _s_location(ds_output_loc, bnd_axis)
      ds_pl = pr_utils.compute_pointList_from_pointRanges(ds_sub_pr_list, n_vtx_zone, _loc, dtype=point_range.dtype)
      PT.update_child(bcds, 'GridLocation', 'GridLocation_t', ds_output_loc)
      PT.new_IndexArray(value=ds_pl, parent=bcds)
      MT.newDistribution({'Index' : ds_distri}, parent=bcds)
    PT.rm_children_from_name(bcds, 'PointRange')
    PT.add_child(bc_u, bcds)


  MT.newDistribution({'Index' : np.array([*bc_range, bc_size.prod()], pdm_gnum_dtype)}, parent=bc_u)
  allowed_types = ['FamilyName_t', 'Descriptor_t', 'AdditionalFamilyName_t'] #Copy these nodes to bc_u
  for allowed_child in [c for c in PT.get_children(bc_s) if PT.get_label(c) in allowed_types]:
    PT.add_child(bc_u, allowed_child)
  return bc_u
###############################################################################

###############################################################################
def gc_s_to_gc_u(gc_s, zone_path, n_vtx_zone, n_vtx_zone_opp, output_loc, i_rank, n_rank):
  """
  Convert a structured gc into a non structured gc. This consist mainly in
  deducing a point_list array from the point_range. The support of the output
  point_list must be specified throught output_loc arg (which can be one of
  'Vertex', 'FaceCenter', 'CellCenter').
  Consistency between PL/PLDonor is preserved.
  """
  assert PT.Subset.GridLocation(gc_s) == 'Vertex'

  T = PT.GridConnectivity.Transform(gc_s, True)

  point_range     = PT.get_value(PT.get_child_from_name(gc_s, 'PointRange'))
  point_range_opp = PT.get_value(PT.get_child_from_name(gc_s, 'PointRangeDonor'))

  # One of the two connected zones is choosen to compute the slabs/sub_pointrange and to impose
  # it to the opposed zone.
  if gc_is_reference(gc_s, zone_path):
    point_range_loc, point_range_opp_loc = point_range, point_range_opp
    n_vtx_loc, n_vtx_opp_loc = n_vtx_zone, n_vtx_zone_opp
  else:
    point_range_loc, point_range_opp_loc = point_range_opp, point_range
    n_vtx_loc, n_vtx_opp_loc = n_vtx_zone_opp, n_vtx_zone
    T = T.transpose()
  loc_transform = T.sum(axis=0) * (np.where(T.T != 0)[1] + 1) #Recompute transform from matrix, who have been (potentially) transformed
  # Refence PR must be increasing, otherwise we have troubles with slabs->sub_point_range
  # When we swap the PR, we must swap the corresponding dim of the PRD as well
  dir_to_swap     = (point_range_loc[:,1] < point_range_loc[:,0])
  opp_dir_to_swap = np.empty_like(dir_to_swap)
  opp_dir_to_swap[abs(loc_transform) - 1] = dir_to_swap

  point_range_loc[dir_to_swap, 0], point_range_loc[dir_to_swap, 1] = \
          point_range_loc[dir_to_swap, 1], point_range_loc[dir_to_swap, 0]
  point_range_opp_loc[opp_dir_to_swap,0], point_range_opp_loc[opp_dir_to_swap,1] \
      = point_range_opp_loc[opp_dir_to_swap,1], point_range_opp_loc[opp_dir_to_swap,0]

  bnd_axis = PT.Subset.normal_axis(PT.new_GridConnectivity1to1(point_range=point_range_loc))
  bnd_axis_opp = PT.Subset.normal_axis(PT.new_GridConnectivity1to1(point_range=point_range_opp_loc))
  #Compute slabs from attended location (better load balance)
  gc_size = pr_utils.transform_bnd_pr_size(point_range_loc, "Vertex", output_loc)
  gc_range = py_utils.uniform_distribution_at(gc_size.prod(), i_rank, n_rank)
  gc_slabs = HFR2S.compute_slabs(gc_size, gc_range)

  sub_pr_list = [np.asarray(slab, point_range.dtype) for slab in gc_slabs]
  if len(n_vtx_zone) == 2:
    sub_pr_list = [slab[:2] for slab in sub_pr_list]
  #Compute sub pointranges from slab
  for sub_pr in sub_pr_list:
    sub_pr[:,0] += point_range_loc[:,0]
    sub_pr[:,1] += point_range_loc[:,0] - 1

  #Get opposed sub point ranges
  sub_pr_opp_list = []
  for sub_pr in sub_pr_list:
    sub_pr_opp = PT.utils._gc_transform_window(sub_pr, point_range_loc[:,0], point_range_opp_loc[:,0], T)
    sub_pr_opp_list.append(sub_pr_opp)

  #If output location is vertex, sub_point_range are ready. Otherwise, some corrections are required
  shift = pr_utils.normal_index_shift(point_range_loc, n_vtx_loc, bnd_axis, "Vertex", output_loc)
  shift_opp = pr_utils.normal_index_shift(point_range_opp_loc, n_vtx_opp_loc, bnd_axis_opp, "Vertex", output_loc)
  for i_pr in range(len(sub_pr_list)):
    sub_pr_list[i_pr][bnd_axis,:] += shift
    sub_pr_opp_list[i_pr][bnd_axis_opp,:] += shift_opp

  #When working on cell|face, extra care has to be taken if PR[:,1] < PR[:,0] : the cell|face id
  #is not given by the bottom left corner but by the top right. We can just shift to retrieve casual behaviour
  if 'Center' in output_loc:
    for sub_pr_opp in sub_pr_opp_list:
      reverted = np.sum(T, axis=1) < 0
      reverted[bnd_axis_opp] = False
      sub_pr_opp[reverted,:] -= 1

  # If the axes of opposite PointRange occurs in reverse order, vect. loop must be reverted thanks to order
  if len(n_vtx_zone) == 3:
    loc_transform_2d = np.abs(np.delete(loc_transform, bnd_axis))
    order = 'C' if loc_transform_2d[0] > loc_transform_2d[1] else 'F'
  else:
    # Not really relevant for 2D because we work on line of points; result will be the same
    order='F'

  _loc, _loc_opp = _s_location(output_loc, bnd_axis), _s_location(output_loc, bnd_axis_opp)
  dtype = point_range_loc.dtype
  point_list_loc     = pr_utils.compute_pointList_from_pointRanges(sub_pr_list, n_vtx_loc, _loc, dtype=dtype)
  point_list_opp_loc = pr_utils.compute_pointList_from_pointRanges(sub_pr_opp_list, n_vtx_opp_loc, _loc_opp, order, dtype=dtype)

  if gc_is_reference(gc_s, zone_path):
    point_list, point_list_opp = point_list_loc, point_list_opp_loc
  else:
    point_list, point_list_opp = point_list_opp_loc, point_list_loc

  gc_u = PT.new_GridConnectivity(PT.get_name(gc_s), PT.get_value(gc_s), type='Abutting1to1', loc=output_loc)
  PT.new_IndexArray('PointList'     , point_list,     parent=gc_u)
  PT.new_IndexArray('PointListDonor', point_list_opp, parent=gc_u)
  MT.newDistribution({'Index' : np.array([*gc_range, gc_size.prod()], pdm_gnum_dtype)}, parent=gc_u)
  #Copy these nodes to gc_u
  allowed_types = ['GridConnectivityProperty_t']
  allowed_names = ['GridConnectivityDonorName']
  for child in PT.get_children(gc_s):
    if PT.get_name(child) in allowed_names or PT.get_label(child) in allowed_types:
      PT.add_child(gc_u, child)
  return gc_u
###############################################################################

###############################################################################
def zonedims_to_ngon_3d(n_vtx_zone, comm, dtype=None):
  """
  Generates distributed NGonElement node from the number of
  vertices in the zone.
  """
  nface_per_dir = n_face_per_dir(n_vtx_zone)
  nf_i, nf_j, nf_k = nface_per_dir
  face_distri = par_utils.uniform_distribution(sum(nface_per_dir), comm)

  n_face_tot = face_distri[2]
  n_face_loc = face_distri[1] - face_distri[0]

  if dtype is None:
    dtype = np.int32 if n_face_tot < np.iinfo(np.int32).max else np.int64

  #Bounds stores for each proc [id of first iface, id of first jface, id of first kface, id of last kface]
  # assuming that face are globally ordered i, then j, then k
  bounds = np.empty(4, dtype=dtype)
  bounds[0] = face_distri[0]
  bounds[1] = bounds[0]
  if bounds[0] < nf_i:
    bounds[1] = min(face_distri[1], nf_i)
  bounds[2] = bounds[1]
  if nf_i <= bounds[1] and bounds[1] < nf_i + nf_j:
    bounds[2] = min(face_distri[1], nf_i + nf_j)
  bounds[3] = face_distri[1]

  assert bounds[3]-bounds[0] == n_face_loc

  face_vtx_idx = 4*np.arange(face_distri[0], face_distri[1]+1, dtype=dtype)
  face_vtx, face_pe = s_numbering.ngon_dconnectivity_from_gnum(bounds+1,n_vtx_zone, dtype)

  _erange = np.array([1, n_face_tot], dtype=dtype)
  ngon = PT.new_NGonElements('NGonElements', erange=_erange, eso=face_vtx_idx, ec=face_vtx, pe=face_pe)

  MT.newDistribution({'Element' : face_distri, 'ElementConnectivity' : 4*face_distri}, parent=ngon)

  return ngon
###############################################################################

###############################################################################
def zonedims_to_ngon_2d(n_vtx_zone, comm, dtype=None):

  nedge_per_dir = n_edge_per_dir(n_vtx_zone)
  ne_i, ne_j = nedge_per_dir
  edge_distri = par_utils.uniform_distribution(sum(nedge_per_dir), comm)

  n_edge_tot = edge_distri[2]
  n_edge_loc = edge_distri[1] - edge_distri[0]

  if dtype is None:
    dtype = np.int32 if n_edge_tot < np.iinfo(np.int32).max else np.int64

  #Bounds stores for each proc [id of first iedge, id of first jedge, id of last jedge]
  # assuming that edge are globally ordered i, then j
  bounds = np.empty(3, dtype=dtype)
  bounds[0] = edge_distri[0]
  bounds[1] = bounds[0]
  if bounds[0] < ne_i:
    bounds[1] = min(edge_distri[1], ne_i)
  bounds[2] = edge_distri[1]

  assert bounds[2]-bounds[0] == n_edge_loc

  edge_vtx, edge_pe = s_numbering.edge_dconnectivity_from_gnum(bounds+1, n_vtx_zone, dtype)

  _erange = np.array([1, n_edge_tot], dtype=dtype)
  edge = PT.new_Elements('EdgeElements', 'BAR_2', erange=_erange, econn=edge_vtx, pe=edge_pe)

  MT.newDistribution({'Element' : edge_distri}, parent=edge)
  return edge
###############################################################################

###############################################################################
def zonedims_to_ngon(n_vtx_zone, comm, dtype=None):
  if len(n_vtx_zone) == 3:
    return zonedims_to_ngon_3d(n_vtx_zone, comm, dtype)
  elif len(n_vtx_zone) == 2:
    return zonedims_to_ngon_2d(n_vtx_zone, comm, dtype)
###############################################################################

###############################################################################
def add_lowerdim_std_elements(zone, n_vtx, cell_dim, comm):
  zdtype = zone[1].dtype

  # Get the list of referenced bnd in mesh
  loc = {2 : 'EdgeCenter', 3 : 'FaceCenter'}[cell_dim]
  elt_kind = {2: 'BAR_2', 3 : 'QUAD_4'}[cell_dim]
  cnt_func = {2 : s_numbering.bar2_connectivity_of_selected_gid,
              3 : s_numbering.quad4_connectivity_of_selected_gid}[cell_dim]
  ini_pl = _collected_shifted_pl(zone, loc, 0)

  # Add some ids to be sure to generated every bnd elements, even
  # if they are not referenced
  bcs_l = generate_all_bnd_bcs(n_vtx, zdtype)
  bcu_l = [bc_s_to_bc_u(bcs, n_vtx, loc, comm.rank, comm.size) for bcs in bcs_l]
  extra_pl = np.concatenate([PT.get_child_from_name(bcu, 'PointList')[1][0] for bcu in bcu_l])
  ini_pl.append(extra_pl)

  # Distribution of all lowerdim elts (including non referenced)
  all_bnd_elt_distri_f = par_utils.distribution_from_gnum(ini_pl, comm, True, True)
  GI = GlobalMultiIndexer(all_bnd_elt_distri_f, [t-1 for t in ini_pl], comm)
  ref_bnd_elt_mask = GI.access_counts > 0
  # Distribution of referenced lowerdim elts only
  ref_bnd_elt_distri = par_utils.dn_to_distribution(ref_bnd_elt_mask.sum(), comm)

  # Extract gid of referenced lowerdim elts (arange + mask) and compute elt connectivity
  d_start = all_bnd_elt_distri_f[comm.rank]
  d_end   = all_bnd_elt_distri_f[comm.rank+1]
  ref_ids = np.arange(d_start+1, d_end+1, dtype=zdtype)[ref_bnd_elt_mask]
  elt_vtx = cnt_func(ref_ids, n_vtx)

  # Then "compress" to renumber ref elts and send this number to update PointLists
  counts = np.zeros(ref_bnd_elt_mask.size, np.int32)
  counts[ref_bnd_elt_mask] = 1
  values = np.arange(ref_bnd_elt_distri[0]+1, ref_bnd_elt_distri[1]+1, dtype=zdtype)
  new_pl = [vb[1] for vb in GI.Take_v((counts, values))]

  # Effective update of PL and create Elements_t node
  # Fake BC will be skipped by zip() inside _update_pl
  _update_pl(zone, loc, new_pl)
  erange = np.array([1, ref_bnd_elt_distri[-1]], elt_vtx.dtype)
  bar = PT.new_Elements(elt_kind, elt_kind, erange=erange, econn=elt_vtx, parent=zone)
  MT.new_distribution({'Element' : ref_bnd_elt_distri}, bar)

  # Lastly, update range of volumic elts to put it after
  PT.Element.Range(PT.get_child_from_label(zone, 'Elements_t'))[:] += erange[1]
###############################################################################

###############################################################################
def convert_s_to_u(dist_tree, connectivity, comm, subset_loc=dict()):
  """Performs the destructuration of the input distributed tree.

  The element connectivity can be generated:

  - as polyedric elements, if ``connectivity`` is set to ``Poly``; 
  - as standard elements, if ``connectivity`` is set to ``Standard``.

  In addition, the output location of the ``BC_t`` and ``GridConnectivity(1to1)_t`` subsets can be
  specified with the ``subset_loc`` dictionnary, using respectively the keys ``BC_t`` and ``GC_t``.
  If nothing is specified, these nodes keep their initial ``GridLocation_t`` value.

  Input tree is modified inplace.

  Note:
    Exists also as :func:`convert_s_to_ngon` with connectivity set to 
    ``Poly`` and subset location set to ``FaceCenter`` (or ``EdgeCenter`` for 2D meshes).

  Args:
    dist_tree (CGNSTree): Structured tree
    connectivity (str): Type of elements used to describe the connectivity.
      Admissible values are ``"Poly"`` and ``"Standard"``.
    comm       (MPIComm) : MPI communicator
    subset_loc (dict, optional):
        Expected output GridLocation for the following subset nodes: ``BC_t``, ``GC_t``.
        For each label, output location can be a single location value, a list
        of locations or None to preserve the input value. Defaults to ``None``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #convert_s_to_u@start
        :end-before: #convert_s_to_u@end
        :dedent: 2
  """
  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()

  add_joins_donor_name(dist_tree, comm)
  zone_path_to_vertex_size = {path: PT.Zone.VertexSize(PT.get_node_from_path(dist_tree, path))
                              for path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t')}

  PT.update_child(dist_tree, 'CGNSLibraryVersion', 'CGNSLibraryVersion_t', 4.2)
  for base in PT.iter_all_CGNSBase_t(dist_tree):
    for zone in PT.iter_all_Zone_t(base):
      if PT.Zone.Type(zone) == 'Unstructured': #Zone is already U
        continue

      elif PT.Zone.Type(zone) == 'Structured': #Zone is S -> convert it
        zone_dims_s = PT.get_value(zone)
        zone_dims_u = np.prod(zone_dims_s, axis=0, dtype=zone_dims_s.dtype).reshape(1,-1)
        n_vtx  = PT.Zone.VertexSize(zone)
        cell_dim = PT.Zone.CellDimension(zone)
        zdtype = zone_dims_s.dtype

        if connectivity == 'Standard':
          # Standard elements --> compute volumic (resp. surfacic) elts 1 ... n_cell
          # Lowerdim elts (eg. indexed by BCs) will be computed later BC conversion
          cell_vtx = cell_vtx_connectivity_S(zone, cell_dim)
          elt_type = {2: 'QUAD_4', 3: 'HEXA_8'}[cell_dim]
          erange = np.array([1, PT.Zone.n_cell(zone)], zdtype)
          elt = PT.new_Elements(elt_type, elt_type, erange=erange, econn=cell_vtx.values, parent=zone)
          MT.new_distribution({'Element' : MT.get_distribution(zone, 'Cell')[1].copy()}, elt)
        else:
          # Poly elements --> compute surfacic (resp. lineic) elts 1 ... n_face
          # Then FaceCenter (resp. EdgeCenter) BCs are ready
          PT.add_child(zone, zonedims_to_ngon(n_vtx, comm, zdtype))
      
        PT.update_child(zone, 'ZoneType', 'ZoneType_t', 'Unstructured')
        PT.set_value(zone, zone_dims_u)

        for flow_solution_s in PT.iter_children_from_label(zone, "FlowSolution_t"):
          patch = PT.get_child_from_predicate(flow_solution_s, lambda n: PT.get_name(n) in ['PointRange', 'PointList'])
          assert patch is None, f"Partial FlowSolution_t are not supported"

        loc_to_name = {'Vertex' : '#Vtx', 'FaceCenter': '#Face', 'EdgeCenter' : '#Edge', 'CellCenter': '#Cell'}
        zonebc_s = PT.get_child_from_label(zone, "ZoneBC_t")
        if zonebc_s is not None:
          bc_u_list = list()
          for bc_s in PT.iter_children_from_label(zonebc_s, "BC_t"):
            out_loc_l = get_output_loc(subset_loc, bc_s)
            for out_loc in out_loc_l:
              suffix = loc_to_name[out_loc] if len(out_loc_l) > 1 else ''
              bc_u = bc_s_to_bc_u(bc_s, n_vtx, out_loc, i_rank, n_rank)
              PT.set_name(bc_u, PT.get_name(bc_u) + suffix)
              bc_u_list.append(bc_u)
          # Replace bcs S -> bcs U
          PT.rm_children_from_label(zonebc_s, 'BC_t')
          PT.get_children(zonebc_s).extend(bc_u_list)

        zone_path = '/'.join([PT.get_name(base), PT.get_name(zone)])
        for zonegc_s in PT.iter_children_from_label(zone, "ZoneGridConnectivity_t"):
          gc_u_list = list()
          for gc_s in PT.iter_children_from_label(zonegc_s, "GridConnectivity1to1_t"):
            out_loc_l = get_output_loc(subset_loc, gc_s)
            opp_zone_path = PT.GridConnectivity.ZoneDonorPath(gc_s, PT.get_name(base))
            n_vtx_opp = zone_path_to_vertex_size[opp_zone_path]
            for out_loc in out_loc_l:
              suffix = loc_to_name[out_loc] if len(out_loc_l) > 1 else ''
              gc_u = gc_s_to_gc_u(gc_s, zone_path, n_vtx, n_vtx_opp, out_loc, i_rank, n_rank)
              PT.set_name(gc_u, PT.get_name(gc_u) + suffix)
              gc_u_list.append(gc_u)
          #Manage not 1to1 gcs as BCs
          is_abutt = lambda n : PT.get_label(n) == 'GridConnectivity_t' and PT.GridConnectivity.Type(n) == 'Abutting'
          for gc_s in PT.iter_children_from_predicate(zonegc_s, is_abutt):
            out_loc_l = get_output_loc(subset_loc, gc_s)
            for out_loc in out_loc_l:
              suffix = loc_to_name[out_loc] if len(out_loc_l) > 1 else ''
              gc_u = bc_s_to_bc_u(gc_s, n_vtx, out_loc, i_rank, n_rank)
              PT.set_name(gc_u, PT.get_name(gc_u) + suffix)
              PT.new_GridConnectivityType('Abutting', gc_u)
              gc_u_list.append(gc_u)

          # Hybrid joins should be here : we just have to translate the PL ijk into face index
          is_abutt1to1 = lambda n : PT.get_label(n) == 'GridConnectivity_t' and PT.GridConnectivity.Type(n) == 'Abutting1to1'
          for gc_s in PT.iter_children_from_predicate(zonegc_s, is_abutt1to1):
            opp_zone_path = PT.GridConnectivity.ZoneDonorPath(gc_s, PT.get_name(base))
            opp_zone = PT.get_node_from_path(dist_tree, opp_zone_path)
            if PT.Zone.Type(opp_zone) != 'Unstructured':
              continue
            loc = PT.Subset.GridLocation(gc_s)
            pl = PT.get_child_from_name(gc_s, 'PointList')[1]
            pl_idx = s_numbering.ijk_to_index_from_loc(*pl, loc, zone_path_to_vertex_size[zone_path])
            pl_idx = pl_idx.reshape((1,-1), order='F')
            if 'EdgeCenter' in loc: #IEdge or JEdge -> EdgeCenter
              PT.update_child(gc_s, 'GridLocation', value='EdgeCenter')
            if 'FaceCenter' in loc: #IFace, JFace or KFaceCenter -> FaceCenter
              PT.update_child(gc_s, 'GridLocation', value='FaceCenter')
            PT.update_child(gc_s, 'PointList', value=pl_idx)
            # Now update PointListDonor of the opposite (already U) join
            for opp_jn in PT.get_nodes_from_predicates(opp_zone, 'GridConnectivity_t'):
              opp_base_name = PT.utils.path_head(opp_zone_path,1)
              if PT.GridConnectivity.ZoneDonorPath(opp_jn, opp_base_name) == zone_path:
                pld_n = PT.get_child_from_name(opp_jn, 'PointListDonor')
                if pld_n is not None and np.array_equal(pld_n[1], pl):
                   PT.update_child(opp_jn, 'PointListDonor', value=pl_idx)
                   break
            else:
              raise RuntimeError(f"Opposite join of {PT.get_name(gc_s)} (zone {zone_path}) has not been found")
          
          # Replace jns S -> jns U
          PT.rm_children_from_predicate(zonegc_s, is_abutt)
          PT.rm_children_from_label(zonegc_s, "GridConnectivity1to1_t")
          PT.get_children(zonegc_s).extend(gc_u_list)

        # Create lowerdim elts now, because BC/groups have been translated into ids
        if connectivity == 'Standard':
          add_lowerdim_std_elements(zone, n_vtx, cell_dim, comm)

        # Update CellCenter subsets 
        if connectivity == 'Standard':
          elt = PT.get_child_from_name(zone, 'QUAD_4' if cell_dim == 3 else 'BAR_2')
        else:
          elt = PT.Zone.NGonNode(zone) if cell_dim == 3 else MT.Zone.EdgeNode(zone)
        cell_offset = PT.Element.Range(elt)[1]
        for subset in PT.iter_all_subsets(zone, 'CellCenter'):
          pl = PT.get_child_from_name(subset, 'PointList')
          pl[1][0] += cell_offset

        # Face or Edge distribution does not exist on U meshes
        distri = MT.getDistribution(zone)
        PT.rm_children_from_name(distri, 'Face')
        PT.rm_children_from_name(distri, 'Edge')

  copy_donor_subset(dist_tree)

###############################################################################
def convert_s_to_ngon(dist_tree, comm):
  """Shortcut to convert_s_to_u with NGon connectivity and FaceCenter subsets"""
  bases_dim = set(PT.get_value(base)[0] for base in PT.iter_all_CGNSBase_t(dist_tree))
  assert len(bases_dim) == 1, "Differents CellDimension in same tree are not allowed"
  _subset_loc = 'EdgeCenter' if list(bases_dim)[0] == 2 else 'FaceCenter'
  convert_s_to_u(dist_tree,
                 'NGON_n',
                 comm,
                 {'BC_t' : _subset_loc, 'GC_t' : _subset_loc})

def convert_s_to_poly(dist_tree, comm):
  """Same as convert_s_to_ngon, but also creates the NFace connectivity"""
  from maia.algo import pe_to_nface, edge_pe_to_ngon
  convert_s_to_ngon(dist_tree, comm)
  for z in PT.iter_all_Zone_t(dist_tree):
    if PT.Zone.CellDimension(z) == 2:
      edge_pe_to_ngon(z,comm)
    elif PT.Zone.CellDimension(z) == 3:
      pe_to_nface(z,comm)
