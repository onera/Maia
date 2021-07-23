#coding:utf-8
import Converter.Internal as I
import numpy              as np

from maia.sids               import sids
from maia                    import npy_pdm_gnum_dtype     as pdm_gnum_dtype
from maia.distribution       import distribution_function  as MDIDF
from maia.distribution       import distribution_tree      as MDIDT
from maia.cgns_io.hdf_filter import range_to_slab          as HFR2S
from .                       import s_numbering_funcs      as s_numb

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
  n_edges    = n_vertices - (np_vtx_slab[:,1] == n_vtx).astype(int)

  return n_face_per_dir(n_vertices, n_edges)
###############################################################################

###############################################################################
def compute_pointList_from_pointRanges(sub_pr_list, n_vtx_S, output_loc, normal_index=None):
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
    ijk_to_face_index = [s_numb.ijk_to_faceiIndex, s_numb.ijk_to_facejIndex, s_numb.ijk_to_facekIndex]
    ijk_to_func = lambda i,j,k : ijk_to_face_index[normal_index](i, j, k, n_cell_S, n_vtx_S)
  elif output_loc == 'Vertex':
    ijk_to_func = lambda i,j,k : s_numb.ijk_to_index(i, j, k, n_vtx_S)
  elif output_loc == 'CellCenter':
    ijk_to_func = lambda i,j,k : s_numb.ijk_to_index(i, j, k, n_cell_S)
  else:
    raise ValueError("Wrong output location : '{}'".format(output_loc))

  # The lambda func ijk_to_vect_func is a wrapping to ijk_to_func (and so to the good indexing func)
  # but with args expressed as numpy arrays : this allow vectorial call of indexing function as if we did an
  # imbricated loop
  ijk_to_vect_func = lambda i_idx, j_idx, k_idx : ijk_to_func(i_idx, j_idx.reshape(-1,1), k_idx.reshape(-1,1,1))

  sub_range_sizes = [(np.abs(pr[:,1] - pr[:,0]) + 1).prod() for pr in sub_pr_list]
  point_list = np.empty((1, sum(sub_range_sizes)), order='F', dtype=pdm_gnum_dtype)
  counter = 0

  for ipr, pr in enumerate(sub_pr_list):
    inc = 2*(pr[:,0] <= pr[:,1]).astype(int) - 1 #In each direction, 1 if pr[l,0] <= pr[l,1] else - 1

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
def apply_transformation(index_1, start_1, start_2, T):
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
    size -= (size != 1).astype(int)
  elif 'Center' in input_loc and output_loc == 'Vertex':
    bnd_axis = guess_bnd_normal_index(point_range, input_loc)
    mask = np.arange(point_range.shape[0]) == bnd_axis
    size += (~mask).astype(int)
  return size
###############################################################################

###############################################################################
def bc_s_to_bc_u(bc_s, n_vtx_zone, output_loc, i_rank, n_rank):
  """
  Convert a structured bc into a non structured bc. This consist mainly in
  deducing a point_list array from the point_range. The support of the output
  point_list must be specified throught output_loc arg (which can be one of
  'Vertex', 'FaceCenter', 'CellCenter').
  For now, BCDataSet are not preserved.
  """
  input_loc = sids.GridLocation(bc_s)
  point_range = I.getValue(I.getNodeFromName1(bc_s, 'PointRange'))

  bnd_axis = guess_bnd_normal_index(point_range, input_loc)
  #Compute slabs from attended location (better load balance)
  bc_size = transform_bnd_pr_size(point_range, input_loc, output_loc)
  bc_range = MDIDF.uniform_distribution_at(bc_size.prod(), i_rank, n_rank)
  bc_slabs = HFR2S.compute_slabs(bc_size, bc_range)

  shift = normal_index_shift(point_range, n_vtx_zone, bnd_axis, input_loc, output_loc)
  #Prepare sub pointRanges from slabs
  sub_pr_list = [np.asarray(slab) for slab in bc_slabs]
  for sub_pr in sub_pr_list:
    sub_pr[:,0] += point_range[:,0]
    sub_pr[:,1] += point_range[:,0] - 1
    sub_pr[bnd_axis,:] += shift

  point_list = compute_pointList_from_pointRanges(sub_pr_list, n_vtx_zone, output_loc, bnd_axis)

  bc_u = I.newBC(I.getName(bc_s), btype=I.getValue(bc_s))
  I.newGridLocation(output_loc, parent=bc_u)
  I.newPointList(value=point_list, parent=bc_u)
  I.newIndexArray('PointList#Size', [1, bc_size.prod()], parent=bc_u)
  allowed_types = ['FamilyName_t'] #Copy these nodes to bc_u
  for allowed_child in [c for c in I.getChildren(bc_s) if I.getType(c) in allowed_types]:
    I.addChild(bc_u, allowed_child)
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
  assert sids.GridLocation(gc_s) == 'Vertex'

  zone_path_opp = I.getValue(gc_s)
  if not '/' in zone_path_opp:
    zone_path_opp = zone_path.split('/')[0] + '/' + zone_path_opp
  transform = I.getValue(I.getNodeFromName1(gc_s, 'Transform'))
  T = compute_transform_matrix(transform)

  point_range     = I.getValue(I.getNodeFromName1(gc_s, 'PointRange'))
  point_range_opp = I.getValue(I.getNodeFromName1(gc_s, 'PointRangeDonor'))

  # One of the two connected zones is choosen to compute the slabs/sub_pointrange and to impose
  # it to the opposed zone.
  if zone_path <= zone_path_opp:
    point_range_loc, point_range_opp_loc = point_range, point_range_opp
    n_vtx_loc, n_vtx_opp_loc = n_vtx_zone, n_vtx_zone_opp
  else:
    point_range_loc, point_range_opp_loc = point_range_opp, point_range
    n_vtx_loc, n_vtx_opp_loc = n_vtx_zone_opp, n_vtx_zone
    T = T.transpose()
  # Refence PR must be increasing, otherwise we have troubles with slabs->sub_point_range
  # When we swap the PR, we must swap the corresponding dim of the PRD as well
  dir_to_swap     = (point_range_loc[:,1] < point_range_loc[:,0])
  opp_dir_to_swap = dir_to_swap[abs(transform) - 1]
  point_range_loc[dir_to_swap, 0], point_range_loc[dir_to_swap, 1] = \
          point_range_loc[dir_to_swap, 1], point_range_loc[dir_to_swap, 0]
  point_range_opp_loc[opp_dir_to_swap,0], point_range_opp_loc[opp_dir_to_swap,1] \
      = point_range_opp_loc[opp_dir_to_swap,1], point_range_opp_loc[opp_dir_to_swap,0]

  bnd_axis = guess_bnd_normal_index(point_range_loc, "Vertex")
  bnd_axis_opp = guess_bnd_normal_index(point_range_opp_loc, "Vertex")
  #Compute slabs from attended location (better load balance)
  gc_size = transform_bnd_pr_size(point_range_loc, "Vertex", output_loc)
  gc_range = MDIDF.uniform_distribution_at(gc_size.prod(), i_rank, n_rank)
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
    sub_pr_opp[:,0] = apply_transformation(sub_pr[:,0], point_range_loc[:,0], point_range_opp_loc[:,0], T)
    sub_pr_opp[:,1] = apply_transformation(sub_pr[:,1], point_range_loc[:,0], point_range_opp_loc[:,0], T)
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
      reverted = np.sum(T, axis=0) < 0
      reverted[bnd_axis_opp] = False
      sub_pr_opp[reverted,:] -= 1

  point_list_loc     = compute_pointList_from_pointRanges(sub_pr_list, n_vtx_loc, output_loc, bnd_axis)
  point_list_opp_loc = compute_pointList_from_pointRanges(sub_pr_opp_list, n_vtx_opp_loc, output_loc, bnd_axis_opp)

  if zone_path <= zone_path_opp:
    point_list, point_list_opp = point_list_loc, point_list_opp_loc
  else:
    point_list, point_list_opp = point_list_opp_loc, point_list_loc

  gc_u = I.newGridConnectivity(I.getName(gc_s), I.getValue(gc_s), 'Abutting1to1')
  I.newGridLocation(output_loc, gc_u)
  I.newPointList('PointList'     , point_list,     parent=gc_u)
  I.newPointList('PointListDonor', point_list_opp, parent=gc_u)
  I.newIndexArray('PointList#Size', [1, gc_size.prod()], gc_u)
  #Copy these nodes to gc_u
  allowed_types = ['GridConnectivityProperty_t']
  allowed_names = ['Ordinal', 'OrdinalOpp']
  for child in I.getChildren(gc_s):
    if I.getName(child) in allowed_names or I.getType(child) in allowed_types:
      I.addChild(gc_u, child)
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
  face_distri = MDIDF.uniform_distribution_at(n_face_tot, i_rank, n_rank)

  n_face_loc = face_distri[1] - face_distri[0]
  #Bounds stores for each proc [id of first iface, id of first jface, id of first kface, id of last kface]
  # assuming that face are globally ordered i, then j, then k
  bounds = np.empty(4, dtype=np.int32)
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
  face_vtx, face_pe = s_numb.ngon_dconnectivity_from_gnum(bounds+1,n_cell_zone, pdm_gnum_dtype)

  ngon = I.newElements('NGonElements', 'NGON', face_vtx, [1, n_face_tot])
  I.newDataArray("ElementStartOffset", face_vtx_idx, parent=ngon)
  I.newIndexArray('ElementConnectivity#Size', [4*n_face_tot], parent=ngon)
  I.newParentElements(face_pe, parent=ngon)

  return ngon
###############################################################################

###############################################################################
def convert_s_to_u(disttree_s, comm, bc_output_loc="FaceCenter", gc_output_loc="FaceCenter"):
  """
  Convert a structured dist_tree into an unstructured dist_tree. This function
    - Copy the grid coordinates and flow solution nodes
    - Create the NGon connectivity
    - Convert the BCs (BCDataSet are not managed yet) and GCs
    - Copy the top level nodes (FlowEquationSet_t, ReferenceState_t, Family_t)
  """
  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()
  disttree_u = I.newCGNSTree()

  for base_s in I.getBases(disttree_s):
    base_u = I.createNode(I.getName(base_s), 'CGNSBase_t', I.getValue(base_s), parent=disttree_u)
    for zone_s in I.getZones(base_s):
      if I.getZoneType(zone_s) == 2: #Zone is already U
        I.addChild(base_u, zone_s)

      elif I.getZoneType(zone_s) == 1: #Zone is S -> convert it
        zone_dims_s = I.getValue(zone_s)
        zone_dims_u = np.prod(zone_dims_s, axis=0).reshape(1,-1)
        n_vtx  = zone_dims_s[:,0]
      
        zone_u = I.createNode(I.getName(zone_s), 'Zone_t', zone_dims_u, parent=base_u)
        I.createNode('ZoneType', 'ZoneType_t', 'Unstructured', parent=zone_u)

        grid_coord_s = I.getNodeFromType1(zone_s, "GridCoordinates_t")
        grid_coord_u = I.newGridCoordinates(parent=zone_u)
        for data in I.getNodesFromType1(grid_coord_s, "DataArray_t"):
          I.addChild(grid_coord_u, data)

        for flow_solution_s in I.getNodesFromType1(zone_s, "FlowSolution_t"):
          flow_solution_u = I.newFlowSolution(I.getName(flow_solution_s), parent=zone_u)
          grid_loc = I.getNodeFromType1(zone_s, "GridLocation_t")
          if grid_loc:
            I.addChild(flow_solution_u, grid_loc)
          for data in I.getNodesFromType1(flow_solution_s, "DataArray_t"):
            I.addChild(flow_solution_u, data)

        I.addChild(zone_u, zonedims_to_ngon(n_vtx, comm))

        zonebc_s = I.getNodeFromType1(zone_s, "ZoneBC_t")
        if zonebc_s is not None:
          zonebc_u = I.newZoneBC(zone_u)
          for bc_s in I.getNodesFromType1(zonebc_s,"BC_t"):
            I.addChild(zonebc_u, bc_s_to_bc_u(bc_s, n_vtx, bc_output_loc, i_rank, n_rank))

        zone_path = '/'.join([I.getName(base_s), I.getName(zone_s)])
        for zonegc_s in I.getNodesFromType1(zone_s, "ZoneGridConnectivity_t"):
          zonegc_u = I.newZoneGridConnectivity(I.getName(zonegc_s), parent=zone_u)
          for gc_s in I.getNodesFromType1(zonegc_s, "GridConnectivity1to1_t"):
            opp_name = I.getValue(gc_s)
            zone_opp_path = zone_opp_name if '/' in opp_name else I.getName(base_s)+'/'+opp_name
            n_vtx_opp = I.getValue(I.getNodeFromPath(disttree_s, zone_opp_path))[:,0]
            I.addChild(zonegc_u, gc_s_to_gc_u(gc_s, zone_path, n_vtx, n_vtx_opp, gc_output_loc, i_rank, n_rank))

    # Top level nodes
    top_level_types = ["FlowEquationSet_t", "ReferenceState_t", "Family_t"]
    for top_level_type in top_level_types:
      for node in I.getNodesFromType1(base_s, top_level_type):
        I.addChild(base_u, node)

  MDIDT.add_distribution_info(disttree_u, comm, distribution_policy='uniform')
  return disttree_u
###############################################################################
