#coding:utf-8
import numpy              as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                 import npy_pdm_gnum_dtype     as pdm_gnum_dtype
from maia.utils           import py_utils, s_numbering
from maia.utils.numbering import range_to_slab          as HFR2S

def get_output_loc(request_dict, s_node):
  """Retrieve output location from the node if not provided in argument"""
  label_to_key = {'BC_t' : 'BC_t', 'GridConnectivity1to1_t' : 'GC_t', 'GridConnectivity_t': 'GC_t'}
  out_loc = request_dict.get(label_to_key[PT.get_label(s_node)], None)
  if out_loc is None:
    out_loc = PT.Subset.GridLocation(s_node)
    if 'FaceCenter' in out_loc: #Remove 'I', 'J', or 'K' for FaceCenter
      out_loc = 'FaceCenter'
  if isinstance(out_loc, str):
    out_loc = [out_loc]
  return out_loc

def gc_is_reference(gc_s, zone_path, zone_path_opp):
  """
  Check if a structured 1to1 GC is the reference of its pair or not
  """
  if zone_path < zone_path_opp:
    return True
  elif zone_path > zone_path_opp:
    return False
  else: #Same zone path
    pr  = PT.get_child_from_name(gc_s, "PointRange")[1]
    prd = PT.get_child_from_name(gc_s, "PointRangeDonor")[1]
    bnd_axis   = guess_bnd_normal_index(pr,  "Vertex")
    bnd_axis_d = guess_bnd_normal_index(prd, "Vertex")
    if bnd_axis < bnd_axis_d:
      return True
    elif bnd_axis > bnd_axis_d:
      return False
    else: #Same boundary axis
      bnd_axis_val = np.abs(pr[bnd_axis,0])
      bnd_axis_val_d = np.abs(prd[bnd_axis_d,0])
      if bnd_axis_val < bnd_axis_val_d:
        return True
      elif bnd_axis_val > bnd_axis_val_d:
        return False
      else: #Same position in boundary axis
        if np.sum(pr) < np.sum(prd):
          return True
        elif np.sum(pr) > np.sum(prd):
          return False
  raise ValueError("Unable to determine if node is reference")

###############################################################################
def n_face_per_dir(n_vtx, n_edge):
  """
  Compute the number of faces in each direction from the number of vtx/edge
  in each direction
  """
  return np.array([n_vtx[0]*n_edge[1]*n_edge[2],
                   n_vtx[1]*n_edge[0]*n_edge[2],
                   n_vtx[2]*n_edge[0]*n_edge[1]])
###############################################################################

###############################################################################
def vtx_slab_to_n_faces(vtx_slab, n_vtx):
  """
  Compute the number of faces to create for a zone by a proc with distributed info
  from a vertex slab
  """
  np_vtx_slab = np.asarray(vtx_slab)
  # Compute number of vertices and number of edges in each dir (exclude last
  # edge if slab is the end of the block
  n_vertices = np_vtx_slab[:,1] - np_vtx_slab[:,0]
  n_edges    = n_vertices - (np_vtx_slab[:,1] == n_vtx)

  return n_face_per_dir(n_vertices, n_edges)
###############################################################################

###############################################################################
def compute_pointList_from_pointRanges(sub_pr_list, n_vtx_S, output_loc, normal_index=None, order='F'):
  """
  Transform a list of pointRange in a concatenated pointList array in order. The sub_pr_list must
  describe entity of kind output_loc, which can take the values 'FaceCenter', 'Vertex' or 'CellCenter'
  and represent the output gridlocation of the pointlist array.
  Note that the pointRange intervals can be reverted (start > end) as it occurs in GC nodes.
  This function also require the normal_index parameter, (admissibles values : 0,1,2) which is mandatory
  to retrieve the indexing function when output_loc == 'FaceCenter'.
  """

  n_cell_S = [nv - 1 for nv in n_vtx_S]

  # The lambda func ijk_to_func redirect to the good indexing function depending
  # on the output grid location
  if output_loc == 'FaceCenter':
    ijk_to_face_index = [s_numbering.ijk_to_faceiIndex, s_numbering.ijk_to_facejIndex, s_numbering.ijk_to_facekIndex]
    ijk_to_func = lambda i,j,k : ijk_to_face_index[normal_index](i, j, k, n_cell_S, n_vtx_S)
  elif output_loc == 'Vertex':
    ijk_to_func = lambda i,j,k : s_numbering.ijk_to_index(i, j, k, n_vtx_S)
  elif output_loc == 'CellCenter':
    ijk_to_func = lambda i,j,k : s_numbering.ijk_to_index(i, j, k, n_cell_S)
  else:
    raise ValueError("Wrong output location : '{}'".format(output_loc))

  # The lambda func ijk_to_vect_func is a wrapping to ijk_to_func (and so to the good indexing func)
  # but with args expressed as numpy arrays : this allow vectorial call of indexing function as if we did an
  # imbricated loop
  if order == 'F':
    ijk_to_vect_func = lambda i_idx, j_idx, k_idx : ijk_to_func(i_idx, j_idx.reshape(-1,1), k_idx.reshape(-1,1,1))
  elif order == 'C':
    ijk_to_vect_func = lambda i_idx, j_idx, k_idx : ijk_to_func(i_idx.reshape(-1,1,1), j_idx.reshape(-1,1), k_idx)

  sub_range_sizes = [(np.abs(pr[:,1] - pr[:,0]) + 1).prod() for pr in sub_pr_list]
  point_list = np.empty((1, sum(sub_range_sizes)), order='F', dtype=n_vtx_S.dtype)
  counter = 0

  for ipr, pr in enumerate(sub_pr_list):
    inc = 2*(pr[:,0] <= pr[:,1]) - 1 #In each direction, 1 if pr[l,0] <= pr[l,1] else - 1

    # Here we build for each direction a looping array range(start, end+1) if pr is increasing
    # or range(start, end-1, -1) if pr is decreasing
    np_idx_arrays = []
    for l in range(pr.shape[0]):
      np_idx_arrays.append(np.arange(pr[l,0], pr[l,1] + inc[l], inc[l]))

    point_list[0][counter:counter+sub_range_sizes[ipr]] = ijk_to_vect_func(*np_idx_arrays).flatten()
    counter += sub_range_sizes[ipr]

  return point_list
###############################################################################

###############################################################################
def is_same_axis(x,y):
  """
  This function is the implementation of the 'del' function defined in the SIDS
  of CGNS (https://cgns.github.io/CGNS_docs_current/sids/cnct.html) as :
  del(x−y) ≡ +1 if |x| = |y|
  """
  return (np.abs(x) == np.abs(y)).astype(int)
###############################################################################

###############################################################################
def compute_transform_matrix(transform):
  """
  This function compute the matrix to convert current indices to opposite indices
  The definition of this matrix is given in the SIDS of CGNS 
  (https://cgns.github.io/CGNS_docs_current/sids/cnct.html)
  """
  transform_np = np.asarray(transform)
  del_matrix = is_same_axis(transform_np, np.array([[1],[2],[3]]))
  return np.sign(transform_np) * del_matrix
###############################################################################

###############################################################################
def apply_transform_matrix(index_1, start_1, start_2, T):
  """
  This function compute indices from current to oppposit or from opposite to current
  by using the transform matrix as defined in the SIDS of CGNS
  (https://cgns.github.io/CGNS_docs_current/sids/cnct.html)
  """
  return np.matmul(T, (index_1 - start_1)) + start_2
###############################################################################

###############################################################################
def guess_bnd_normal_index(point_range, grid_location):
  """
  From a point_range array and a grid_location value, try to predict the plane
  on which the boundary was created. Return the axis on which the boundary
  is constant (0:=x, 1:=y, 2:=z) eg return 1 if boundary belongs to x,y plane.
  """
  if grid_location in ['IFaceCenter', 'JFaceCenter', 'KFaceCenter']:
    normal_index = {'I':0, 'J':1, 'K':2}[grid_location[0]]
  elif sum(point_range[:,0] == point_range[:,1]) == 1: #Ambiguity can be resolved
    normal_index = np.nonzero(point_range[:,0] == point_range[:,1])[0][0]
  else:
    raise ValueError("Ambiguous input location")
  return normal_index
###############################################################################

###############################################################################
def normal_index_shift(point_range, n_vtx, bnd_axis, input_loc, output_loc):
  """
  Return the value that should be added to pr[normal_index,:] to account for cell <-> face|vtx transformation :
    +1 if we move from cell to face|vtx and if it was the last plane of cells
    -1 if we move from face|vtx to cell and if it was the last plane of face|vtx
     0 in other cases
  """
  in_loc_is_cell  = (input_loc == 'CellCenter')
  out_loc_is_cell = (output_loc == 'CellCenter')
  normal_index_is_last = point_range[bnd_axis,0] == (n_vtx[bnd_axis] - int(in_loc_is_cell))
  correction_sign = -int(out_loc_is_cell and not in_loc_is_cell) \
                    +int(not out_loc_is_cell and in_loc_is_cell)
  return int(normal_index_is_last) * correction_sign
###############################################################################

###############################################################################
def transform_bnd_pr_size(point_range, input_loc, output_loc):
  """
  Predict a point_range defined at an input_location if it were defined at an output_location
  """
  size = np.abs(point_range[:,1] - point_range[:,0]) + 1

  if input_loc == 'Vertex' and 'Center' in output_loc:
    size -= (size != 1)
  elif 'Center' in input_loc and output_loc == 'Vertex':
    bnd_axis = guess_bnd_normal_index(point_range, input_loc)
    mask = np.arange(point_range.shape[0]) == bnd_axis
    size += (~mask)
  return size
###############################################################################

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

  bnd_axis = guess_bnd_normal_index(point_range, input_loc)
  #Compute slabs from attended location (better load balance)
  bc_size = transform_bnd_pr_size(point_range, input_loc, output_loc)
  bc_range = py_utils.uniform_distribution_at(bc_size.prod(), i_rank, n_rank)
  bc_slabs = HFR2S.compute_slabs(bc_size, bc_range)

  shift = normal_index_shift(point_range, n_vtx_zone, bnd_axis, input_loc, output_loc)
  #Prepare sub pointRanges from slabs
  sub_pr_list = [np.asarray(slab) for slab in bc_slabs]
  for sub_pr in sub_pr_list:
    sub_pr[:,0] += point_range[:,0]
    sub_pr[:,1] += point_range[:,0] - 1
    sub_pr[bnd_axis,:] += shift

  point_list = compute_pointList_from_pointRanges(sub_pr_list, n_vtx_zone, output_loc, bnd_axis)

  bc_u = PT.new_node(PT.get_name(bc_s), PT.get_label(bc_s), PT.get_value(bc_s))
  PT.new_GridLocation(output_loc, parent=bc_u)
  PT.new_PointList(value=point_list, parent=bc_u)

  # Manage datasets -- Data is already distributed, we just have to retrive the PointList
  # of the corresponding elements following same procedure than BCs
  for bcds in PT.iter_children_from_label(bc_s, 'BCDataSet_t'):
    ds_point_range = PT.get_child_from_name(bcds, 'PointRange')
    is_related = ds_point_range is None
    if not is_related: #BCDS has its own location / pr
      ds_distri = MT.getDistribution(bcds, 'Index')[1]
      ds_loc   = PT.Subset.GridLocation(bcds)
    if is_related: #BCDS has same location / pr than bc
      ds_point_range = PT.get_child_from_name(bc_s, 'PointRange')
      ds_distri = MT.getDistribution(bc_s, 'Index')[1]
      ds_loc   = input_loc
    ds_size = PT.PointRange.SizePerIndex(ds_point_range)
    ds_slabs = HFR2S.compute_slabs(ds_size, ds_distri[0:2])
    ds_sub_pr_list = [np.asarray(slab) for slab in ds_slabs]
    for sub_pr in ds_sub_pr_list:
      sub_pr[:,0] += ds_point_range[1][:,0]
      sub_pr[:,1] += ds_point_range[1][:,0] - 1
    ds_output_loc = 'FaceCenter' if ds_loc in ['IFaceCenter', 'JFaceCenter', 'KFaceCenter'] else ds_loc

    if not (is_related and ds_output_loc == output_loc): #Otherwise, point list has already been computed
      ds_pl = compute_pointList_from_pointRanges(ds_sub_pr_list, n_vtx_zone, ds_output_loc, bnd_axis)
      PT.update_child(bcds, 'GridLocation', 'GridLocation_t', ds_output_loc)
      PT.new_PointList(value=ds_pl, parent=bcds)
      MT.newDistribution({'Index' : ds_distri}, parent=bcds)
    PT.rm_children_from_name(bcds, 'PointRange')
    PT.add_child(bc_u, bcds)


  MT.newDistribution({'Index' : np.array([*bc_range, bc_size.prod()], pdm_gnum_dtype)}, parent=bc_u)
  allowed_types = ['FamilyName_t'] #Copy these nodes to bc_u
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

  zone_path_opp = PT.get_value(gc_s)
  if not '/' in zone_path_opp:
    zone_path_opp = zone_path.split('/')[0] + '/' + zone_path_opp
  transform = PT.get_value(PT.get_child_from_name(gc_s, 'Transform'))
  T = compute_transform_matrix(transform)

  point_range     = PT.get_value(PT.get_child_from_name(gc_s, 'PointRange'))
  point_range_opp = PT.get_value(PT.get_child_from_name(gc_s, 'PointRangeDonor'))

  # One of the two connected zones is choosen to compute the slabs/sub_pointrange and to impose
  # it to the opposed zone.
  if gc_is_reference(gc_s, zone_path, zone_path_opp):
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

  bnd_axis = guess_bnd_normal_index(point_range_loc, "Vertex")
  bnd_axis_opp = guess_bnd_normal_index(point_range_opp_loc, "Vertex")
  #Compute slabs from attended location (better load balance)
  gc_size = transform_bnd_pr_size(point_range_loc, "Vertex", output_loc)
  gc_range = py_utils.uniform_distribution_at(gc_size.prod(), i_rank, n_rank)
  gc_slabs = HFR2S.compute_slabs(gc_size, gc_range)

  sub_pr_list = [np.asarray(slab) for slab in gc_slabs]
  #Compute sub pointranges from slab
  for sub_pr in sub_pr_list:
    sub_pr[:,0] += point_range_loc[:,0]
    sub_pr[:,1] += point_range_loc[:,0] - 1

  #Get opposed sub point ranges
  sub_pr_opp_list = []
  for sub_pr in sub_pr_list:
    sub_pr_opp = np.empty((3,2), dtype=sub_pr.dtype)
    sub_pr_opp[:,0] = apply_transform_matrix(sub_pr[:,0], point_range_loc[:,0], point_range_opp_loc[:,0], T)
    sub_pr_opp[:,1] = apply_transform_matrix(sub_pr[:,1], point_range_loc[:,0], point_range_opp_loc[:,0], T)
    sub_pr_opp_list.append(sub_pr_opp)

  #If output location is vertex, sub_point_range are ready. Otherwise, some corrections are required
  shift = normal_index_shift(point_range_loc, n_vtx_loc, bnd_axis, "Vertex", output_loc)
  shift_opp = normal_index_shift(point_range_opp_loc, n_vtx_opp_loc, bnd_axis_opp, "Vertex", output_loc)
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
  loc_transform_2d = np.abs(np.delete(loc_transform, bnd_axis))
  order = 'C' if loc_transform_2d[0] > loc_transform_2d[1] else 'F'

  point_list_loc     = compute_pointList_from_pointRanges(sub_pr_list, n_vtx_loc, output_loc, bnd_axis)
  point_list_opp_loc = compute_pointList_from_pointRanges(sub_pr_opp_list, n_vtx_opp_loc, output_loc, bnd_axis_opp, order)

  if gc_is_reference(gc_s, zone_path, zone_path_opp):
    point_list, point_list_opp = point_list_loc, point_list_opp_loc
  else:
    point_list, point_list_opp = point_list_opp_loc, point_list_loc

  gc_u = PT.new_GridConnectivity(PT.get_name(gc_s), PT.get_value(gc_s), type='Abutting1to1', loc=output_loc)
  PT.new_PointList('PointList'     , point_list,     parent=gc_u)
  PT.new_PointList('PointListDonor', point_list_opp, parent=gc_u)
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
def zonedims_to_ngon(n_vtx_zone, comm):
  """
  Generates distributed NGonElement node from the number of
  vertices in the zone.
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  n_cell_zone = n_vtx_zone - 1

  nf_i, nf_j, nf_k = n_face_per_dir(n_vtx_zone, n_cell_zone)
  n_face_tot = nf_i + nf_j + nf_k
  face_distri = py_utils.uniform_distribution_at(n_face_tot, i_rank, n_rank)

  n_face_loc = face_distri[1] - face_distri[0]
  #Bounds stores for each proc [id of first iface, id of first jface, id of first kface, id of last kface]
  # assuming that face are globally ordered i, then j, then k
  bounds = np.empty(4, dtype=n_cell_zone.dtype)
  bounds[0] = face_distri[0]
  bounds[1] = bounds[0]
  if bounds[0] < nf_i:
    bounds[1] = min(face_distri[1], nf_i)
  bounds[2] = bounds[1]
  if nf_i <= bounds[1] and bounds[1] < nf_i + nf_j:
    bounds[2] = min(face_distri[1], nf_i + nf_j)
  bounds[3] = face_distri[1]

  assert bounds[3]-bounds[0] == n_face_loc


  face_vtx_idx = 4*np.arange(face_distri[0], face_distri[1]+1, dtype=pdm_gnum_dtype)
  face_vtx, face_pe = s_numbering.ngon_dconnectivity_from_gnum(bounds+1,n_cell_zone, pdm_gnum_dtype)

  _erange = np.array([1, n_face_tot], dtype=pdm_gnum_dtype)
  ngon = PT.new_NGonElements('NGonElements', erange=_erange, eso=face_vtx_idx, ec=face_vtx, pe=face_pe)

  cg_face_distri = np.array([*face_distri, n_face_tot], dtype=pdm_gnum_dtype)
  MT.newDistribution({'Element' : cg_face_distri, 'ElementConnectivity' : 4*cg_face_distri}, parent=ngon)

  return ngon
###############################################################################

###############################################################################
def convert_s_to_u(disttree_s, connectivity, comm, subset_loc=dict()):
  """Performs the destructuration of the input ``dist_tree``.

  This function copies the ``GridCoordinate_t`` and (full) ``FlowSolution_t`` nodes,
  generate a NGon based connectivity and create a PointList for the following
  subset nodes: 
  BC_t, BCDataSet_t and GridConnectivity1to1_t.
  In addition, a PointListDonor node is generated for GridConnectivity_t nodes.
  
  Metadata nodes ("FamilyName_t", "ReferenceState_t", ...) at zone and base level
  are also reported on the unstructured tree.

  Note: 
    Exists also as :func:`convert_s_to_ngon()` with connectivity set to 
    NGON_n and subset_loc set to FaceCenter.

  Args:
    disttree_s (CGNSTree): Structured tree
    connectivity (str): Type of elements used to describe the connectivity.
      Admissible values are ``"NGON_n"`` and ``"HEXA"`` (not yet implemented).
    comm       (MPIComm) : MPI communicator
    subset_loc (dict, optional):
        Expected output GridLocation for the following subset nodes: BC_t, GC_t.
        For each label, output location can be a single location value, a list
        of locations or None to preserve the input value. Defaults to None.
  Returns:
    CGNSTree: Unstructured disttree

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #convert_s_to_u@start
        :end-before: #convert_s_to_u@end
        :dedent: 2
  """
  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()
  disttree_u = PT.new_CGNSTree()

  for base_s in PT.iter_all_CGNSBase_t(disttree_s):
    base_u = PT.new_node(PT.get_name(base_s), 'CGNSBase_t', PT.get_value(base_s), parent=disttree_u)
    for zone_s in PT.iter_all_Zone_t(base_s):
      if PT.Zone.Type(zone_s) == 'Unstructured': #Zone is already U
        PT.add_child(base_u, zone_s)

      elif PT.Zone.Type(zone_s) == 'Structured': #Zone is S -> convert it
        zone_dims_s = PT.get_value(zone_s)
        zone_dims_u = np.prod(zone_dims_s, axis=0, dtype=zone_dims_s.dtype).reshape(1,-1)
        n_vtx  = zone_dims_s[:,0]
      
        zone_u = PT.new_Zone(PT.get_name(zone_s), type='Unstructured', size=zone_dims_u, parent=base_u)

        grid_coord_s = PT.get_child_from_label(zone_s, "GridCoordinates_t")
        grid_coord_u = PT.new_GridCoordinates(parent=zone_u)
        for data in PT.iter_children_from_label(grid_coord_s, "DataArray_t"):
          PT.add_child(grid_coord_u, data)

        for flow_solution_s in PT.iter_children_from_label(zone_s, "FlowSolution_t"):
          flow_solution_u = PT.new_FlowSolution(PT.get_name(flow_solution_s), parent=zone_u,
              loc = PT.Subset.GridLocation(flow_solution_s))
          for data in PT.iter_children_from_label(flow_solution_s, "DataArray_t"):
            PT.add_child(flow_solution_u, data)

        PT.add_child(zone_u, zonedims_to_ngon(n_vtx, comm))

        loc_to_name = {'Vertex' : '#Vtx', 'FaceCenter': '#Face', 'CellCenter': '#Cell'}
        zonebc_s = PT.get_child_from_label(zone_s, "ZoneBC_t")
        if zonebc_s is not None:
          zonebc_u = PT.new_ZoneBC(zone_u)
          for bc_s in PT.iter_children_from_label(zonebc_s, "BC_t"):
            out_loc_l = get_output_loc(subset_loc, bc_s)
            for out_loc in out_loc_l:
              suffix = loc_to_name[out_loc] if len(out_loc_l) > 1 else ''
              bc_u = bc_s_to_bc_u(bc_s, n_vtx, out_loc, i_rank, n_rank)
              PT.set_name(bc_u, PT.get_name(bc_u) + suffix)
              PT.add_child(zonebc_u, bc_u)

        zone_path = '/'.join([PT.get_name(base_s), PT.get_name(zone_s)])
        for zonegc_s in PT.iter_children_from_label(zone_s, "ZoneGridConnectivity_t"):
          zonegc_u = PT.new_ZoneGridConnectivity(PT.get_name(zonegc_s), parent=zone_u)
          for gc_s in PT.iter_children_from_label(zonegc_s, "GridConnectivity1to1_t"):
            opp_name = PT.get_value(gc_s)
            out_loc_l = get_output_loc(subset_loc, gc_s)
            zone_opp_path = zone_opp_name if '/' in opp_name else PT.get_name(base_s)+'/'+opp_name
            n_vtx_opp = PT.get_value(PT.get_node_from_path(disttree_s, zone_opp_path))[:,0]
            for out_loc in out_loc_l:
              suffix = loc_to_name[out_loc] if len(out_loc_l) > 1 else ''
              gc_u = gc_s_to_gc_u(gc_s, zone_path, n_vtx, n_vtx_opp, out_loc, i_rank, n_rank)
              PT.set_name(gc_u, PT.get_name(gc_u) + suffix)
              PT.add_child(zonegc_u, gc_u)
          #Manage not 1to1 gcs as BCs
          is_abbut = lambda n : PT.get_label(n) == 'GridConnectivity_t' and PT.GridConnectivity.Type(n) == 'Abutting'
          for gc_s in PT.iter_children_from_predicate(zonegc_s, is_abbut):
            out_loc_l = get_output_loc(subset_loc, gc_s)
            for out_loc in out_loc_l:
              suffix = loc_to_name[out_loc] if len(out_loc_l) > 1 else ''
              gc_u = bc_s_to_bc_u(gc_s, n_vtx, out_loc, i_rank, n_rank)
              PT.set_name(gc_u, PT.get_name(gc_u) + suffix)
              PT.new_GridConnectivityType('Abutting', gc_u)
              PT.add_child(zonegc_u, gc_u)

        # Copy distribution of all Cell/Vtx, which is unchanged
        distri = PT.deep_copy(MT.getDistribution(zone_s))
        PT.rm_children_from_name(distri, 'Face')
        PT.add_child(zone_u, distri)

        # Top level nodes
        top_level_types = ["FamilyName_t", "AdditionalFamilyName_t", "Descriptor_t", \
            "FlowEquationSet_t", "ReferenceState_t", "ConvergenceHistory_t"]
        for top_level_type in top_level_types:
          for node in PT.iter_children_from_label(zone_s, top_level_type):
            PT.add_child(zone_u, node)

    # Top level nodes
    top_level_types = ["FlowEquationSet_t", "ReferenceState_t", "Family_t"]
    for top_level_type in top_level_types:
      for node in PT.iter_children_from_label(base_s, top_level_type):
        PT.add_child(base_u, node)

  return disttree_u
###############################################################################
def convert_s_to_ngon(disttree_s, comm):
  """Shortcut to convert_s_to_u with NGon connectivity and FaceCenter subsets"""
  return convert_s_to_u(disttree_s,
                        'NGON_n',
                        comm,
                        {'BC_t' : 'FaceCenter', 'GC_t' : 'FaceCenter'})

def convert_s_to_poly(disttree_s, comm):
  """Same as convert_s_to_ngon, but also creates the NFace connectivity"""
  disttree_u = convert_s_to_ngon(disttree_s, comm)
  for z in PT.iter_all_Zone_t(disttree_u):
    NGT.pe_to_nface(z,comm)
  return disttree_u
