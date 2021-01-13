#coding:utf-8
import Converter.Internal as I
import numpy              as np

import Pypdm.Pypdm as PDM

from maia.distribution       import distribution_function           as MDIDF
from maia.cgns_io.hdf_filter import range_to_slab                   as HFR2S
from .                       import s_numbering_funcs               as s_numb

###############################################################################
def n_face_per_dir(n_vtx):
  n_cell = n_vtx - 1
  return np.array([n_vtx[0]*n_cell[1]*n_cell[2],
                   n_vtx[1]*n_cell[0]*n_cell[2],
                   n_vtx[2]*n_cell[0]*n_cell[1]])

def vtx_slab_to_n_face(vtx_slab, n_vtx):
  """
  Compute the number of faces to create for a zone by a proc with distributed info
  from a vertex slab
  """
  np_vtx_slab = np.asarray(vtx_slab)

  # Number of vertices of the slab in each direction
  n_vertices  = np_vtx_slab[:,1] - np_vtx_slab[:,0]
  # Number of edges of the slab in each direction : exclude last edge if slab
  # is the end of the block
  n_edges = n_vertices - (np_vtx_slab[:,1] == n_vtx).astype(int)

  # In each direction, number of faces is n_vtx * n_edge1 * n_edge2
  n_faces_per_dir = np.array([ n_edges[1]*n_edges[2],
                               n_edges[0]*n_edges[2],
                               n_edges[0]*n_edges[1]])
  return (n_vertices * n_faces_per_dir).sum()

###############################################################################

###############################################################################
def compute_all_ngon_connectivity(vtx_slab_l, n_vtx, face_gnum, face_vtx, face_pe):
  """
  Compute the numerotation, the nodes and the cells linked to all face traited for
  zone by a proc and fill associated tabs :
  faceNumber refers to non sorted numerotation of each face
  face_vtx refers to non sorted NGonConnectivity
  faceLeftCell refers to non sorted ParentElement[:][0]
  faceRightCell refers to non sorted ParentElement[:][1]
  Remark : all tabs are defined in the same way i.e. for the fth face, information are
  located in faceNumber[f], face_vtx[4*f:4*(f+1)], faceLeftCell[f] and faceRightCell[f]
  WARNING : (i,j,k) begins at (1,1,1)
  """
  n_cell = n_vtx - 1
  counter = 0
  for vtx_slab in vtx_slab_l:
    iS,iE, jS,jE, kS,kE = [item+1 for bounds in vtx_slab for item in bounds]
    isup = iE - int(iE == n_vtx[0]+1)
    jsup = jE - int(jE == n_vtx[1]+1)
    ksup = kE - int(kE == n_vtx[2]+1)

    n_faces_i = (iE-iS)*(jsup-jS)*(ksup-kS)
    n_faces_j = (isup-iS)*(jE-jS)*(ksup-kS)
    n_faces_k = (isup-iS)*(jsup-jS)*(kE-kS)

    #Do 3 loops to remove if test
    start = counter
    end   = start + n_faces_i
    i_ar  = np.arange(iS,iE).reshape(-1,1,1)
    j_ar  = np.arange(jS,jsup).reshape(-1,1)
    k_ar  = np.arange(kS,ksup)

    face_gnum[start:end]    = s_numb.ijk_to_faceiIndex(i_ar,j_ar,k_ar,n_cell,n_vtx).flatten()
    face_pe[start:end]      = s_numb.compute_fi_PE_from_idx(face_gnum[start:end], n_cell, n_vtx)
    face_vtx[4*start:4*end] = s_numb.compute_fi_facevtx_from_idx(face_gnum[start:end], n_cell, n_vtx)
    counter += n_faces_i

    #Shift ifaces (shift is global for zone)
    shift = n_vtx[0]*(n_cell[1]*n_cell[2])
    start = counter
    end   = start + n_faces_j
    i_ar  = np.arange(iS,isup).reshape(-1,1,1)
    j_ar  = np.arange(jS,jE).reshape(-1,1)
    k_ar  = np.arange(kS,ksup)
    
    face_gnum[start:end]    = s_numb.ijk_to_facejIndex(i_ar,j_ar,k_ar,n_cell,n_vtx).flatten()
    face_pe[start:end]      = s_numb.compute_fj_PE_from_idx(face_gnum[start:end]-shift, n_cell, n_vtx)
    face_vtx[4*start:4*end] = s_numb.compute_fj_facevtx_from_idx(face_gnum[start:end]-shift, n_cell, n_vtx)
    counter += n_faces_j

    shift += n_vtx[1]*(n_cell[0]*n_cell[2])
    start = counter
    end   = start + n_faces_k
    i_ar  = np.arange(iS,isup).reshape(-1,1,1)
    j_ar  = np.arange(jS,jsup).reshape(-1,1)
    k_ar  = np.arange(kS,kE)

    face_gnum[start:end]    = s_numb.ijk_to_facekIndex(i_ar,j_ar,k_ar,n_cell,n_vtx).flatten()
    face_pe[start:end]      = s_numb.compute_fk_PE_from_idx(face_gnum[start:end]-shift, n_cell, n_vtx)
    face_vtx[4*start:4*end] = s_numb.compute_fk_facevtx_from_idx(face_gnum[start:end]-shift, n_cell, n_vtx)
    counter += n_faces_k

###############################################################################

###############################################################################
def compute_pointList_from_pointRanges(sub_pr_list, n_vtxS, output_loc, cst_axe=None):
  """
  Transform a list of pointRange in a concatenated pointList array in order. The sub_pr_list must
  describe entity of kind output_loc, which can take the values 'FaceCenter', 'Vertex' or 'CellCenter'
  and represent the output gridlocation of the pointlist array.
  Note that the pointRange intervals can be reverted (start > end) as it occurs in GC nodes.
  This function also require the cst_axe parameter, (admissibles values : 0,1,2) which is mandatory
  to retrieve the indexing function when output_loc == 'FaceCenter'.
  """

  nCellS = [nv - 1 for nv in n_vtxS]

  # The lambda func ijk_to_func redirect to the good indexing function depending
  # on the output grid location
  if output_loc == 'FaceCenter':
    ijk_to_faceIndex = [s_numb.ijk_to_faceiIndex, s_numb.ijk_to_facejIndex, s_numb.ijk_to_facekIndex]
    ijk_to_func = lambda i,j,k : ijk_to_faceIndex[cst_axe](i, j, k, nCellS, n_vtxS)
  elif output_loc == 'Vertex':
    ijk_to_func = lambda i,j,k : s_numb.ijk_to_index(i, j, k, n_vtxS)
  elif output_loc == 'CellCenter':
    ijk_to_func = lambda i,j,k : s_numb.ijk_to_index(i, j, k, nCellS)
  else:
    raise ValueError("Wrong output location : '{}'".format(output_loc))

  # The lambda func ijk_to_vect_func is a wrapping to ijk_to_func (and so to the good indexing func)
  # but with args expressed as numpy arrays : this allow vectorial call of indexing function as if we did an
  # imbricated loop
  ijk_to_vect_func = lambda i_idx, j_idx, k_idx : ijk_to_func(i_idx, j_idx.reshape(-1,1), k_idx.reshape(-1,1,1))

  sub_range_sizes = [(np.abs(pr[:,1] - pr[:,0]) + 1).prod() for pr in sub_pr_list]
  point_list = np.empty((1, sum(sub_range_sizes)), dtype=np.int32)
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
def isSameAxis(x,y):
  """
  This function is the implementation of the 'del' function defined in the SIDS
  of CGNS (https://cgns.github.io/CGNS_docs_current/sids/cnct.html) as :
  del(x−y) ≡ +1 if |x| = |y|
  """
  return (np.abs(x) == np.abs(y)).astype(int)
###############################################################################

###############################################################################
def compute_transformMatrix(transform):
  """
  This function compute the matrix to convert current indices to opposit indices
  The definition of this matrix is given in the SIDS of CGNS 
  (https://cgns.github.io/CGNS_docs_current/sids/cnct.html)
  """
  transform_np = np.asarray(transform)
  del_matrix = isSameAxis(transform_np, np.array([[1],[2],[3]]))
  return np.sign(transform_np) * del_matrix
###############################################################################

###############################################################################
def apply_transformation(index1, start1, start2, T):
  """
  This function compute indices from current to oppposit or from opposit to current
  by using the transform matrix.
  As defined in the SIDS of CGNS (https://cgns.github.io/CGNS_docs_current/sids/cnct.html) :
  Index2 = T.(Index1 - Begin1) + Begin2
  """
  return np.matmul(T, (index1 - start1)) + start2
###############################################################################

def guess_boundary_axis(point_range, grid_location):
  if grid_location in ['IFaceCenter', 'JFaceCenter', 'KFaceCenter']:
    cst_axe = {'I':0, 'J':1, 'K':2}[grid_location[0]]
  elif sum(point_range[:,0] == point_range[:,1]) == 1: #Ambiguity can be resolved
    cst_axe = np.nonzero(point_range[:,0] == point_range[:,1])[0][0]
  else:
    raise ValueError("Ambiguous input location")
  return cst_axe

def cst_axe_shift(point_range, n_vtx, bnd_axis, in_loc_is_cell, out_loc_is_cell):
  """
  Return the value that should be added to pr[cst_axe,:] to account for cell <-> face|vtx transformation :
    +1 if we move from cell to face|vtx and if it was the last plane of cells
    -1 if we move from face|vtx to cell and if it was the last plane of face|vtx
     0 in other cases
  """
  cst_axe_is_last = point_range[bnd_axis,0] == (n_vtx[bnd_axis] - int(in_loc_is_cell))
  correction_sign = -int(out_loc_is_cell and not in_loc_is_cell) + int(not out_loc_is_cell and in_loc_is_cell)
  return int(cst_axe_is_last) * correction_sign

def transform_bnd_pr_size(point_range, input_loc, output_loc):
  """
  Predict a point_range defined at an input_location if it were defined at an output_location
  """
  size = np.abs(point_range[:,1] - point_range[:,0]) + 1

  if input_loc == 'Vertex' and 'Center' in output_loc:
    size -= (size != 1).astype(int)
  elif 'Center' in input_loc and output_loc == 'Vertex':
    bnd_axis = guess_boundary_axis(point_range, input_loc)
    mask = np.arange(point_range.shape[0]) == bnd_axis
    size += (~mask).astype(int)
  return size

###############################################################################
def bc_s_to_bc_u(bc_s, n_vtx_zone, output_loc, i_rank, n_rank):
  input_loc_node = I.getNodeFromType1(bc_s, "GridLocation_t")
  input_loc = I.getValue(input_loc_node) if input_loc_node is not None else "Vertex"
  point_range = I.getValue(I.getNodeFromName1(bc_s, 'PointRange'))

  bnd_axis = guess_boundary_axis(point_range, input_loc)
  #Compute slabs from attended location (better load balance)
  bc_size = transform_bnd_pr_size(point_range, input_loc, output_loc)
  bc_range = MDIDF.uniform_distribution_at(bc_size.prod(), i_rank, n_rank)
  bc_slabs = HFR2S.compute_slabs(bc_size, bc_range)

  shift = cst_axe_shift(point_range, n_vtx_zone, bnd_axis,\
      input_loc=='CellCenter', output_loc=='CellCenter')
  #Prepare sub pointRanges from slabs
  sub_pr_list = [np.asarray(slab) for slab in bc_slabs]
  for sub_pr in sub_pr_list:
    sub_pr[:,0] += point_range[:,0]
    sub_pr[:,1] += point_range[:,0] - 1
    sub_pr[bnd_axis,:] += shift

  point_list = compute_pointList_from_pointRanges(sub_pr_list,n_vtx_zone,output_loc, bnd_axis)

  bc_u = I.newBC(I.getName(bc_s), btype=I.getValue(bc_s))
  I.newGridLocation(output_loc, parent=bc_u)
  I.newPointList(value=point_list, parent=bc_u)
  I.newIndexArray('PointList#Size', [1, bc_size.prod()], parent=bc_u)
  allowed_types = ['FamilyName_t'] #Copy these nodes to bc_u
  for allowed_child in [c for c in I.getChildren(bc_s) if I.getType(c) in allowed_types]:
    I.addChild(bc_u, allowed_child)
  return bc_u


def gc_s_to_gc_u(gc_s, zone_path, n_vtx_zone, n_vtx_zone_opp, output_loc, i_rank, n_rank):
  input_loc_node = I.getNodeFromType1(gc_s, "GridLocation_t")
  assert input_loc_node is None or I.getValue(input_loc_node) == "Vertex"

  zone_path_opp = I.getValue(gc_s)
  if not '/' in zone_path_opp:
    zone_path_opp = zone_path.split('/')[0] + '/' + zone_path_opp
  transform = I.getValue(I.getNodeFromName1(gc_s, 'Transform'))
  T = compute_transformMatrix(transform)

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

  bnd_axis = guess_boundary_axis(point_range_loc, "Vertex")
  bnd_axis_opp = guess_boundary_axis(point_range_opp_loc, "Vertex")
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
    sub_pr_opp = np.empty((3,2), dtype=np.int32)
    sub_pr_opp[:,0] = apply_transformation(sub_pr[:,0], point_range_loc[:,0], point_range_opp_loc[:,0], T)
    sub_pr_opp[:,1] = apply_transformation(sub_pr[:,1], point_range_loc[:,0], point_range_opp_loc[:,0], T)
    sub_pr_opp_list.append(sub_pr_opp)

  #If output location is vertex, sub_point_range are ready. Otherwise, some corrections are required
  shift = cst_axe_shift(point_range_loc, n_vtx_loc, bnd_axis, False, output_loc=='CellCenter')
  shift_opp = cst_axe_shift(point_range_opp_loc, n_vtx_opp_loc, bnd_axis_opp, False, output_loc=='CellCenter')
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

def zonedims_to_ngon(n_vtx_zone, comm):
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  n_face_tot = n_face_per_dir(n_vtx_zone).sum()
  #> with NgonElements
  #>> Definition en non structure des faces
  vtx_range  = MDIDF.uniform_distribution_at(n_vtx_zone.prod(), i_rank, n_rank)
  vtx_slabs  = HFR2S.compute_slabs(n_vtx_zone, vtx_range)
  n_face_slab = sum([vtx_slab_to_n_face(slab, n_vtx_zone) for slab in vtx_slabs])
  face_gnum     = np.empty(  n_face_slab, dtype=np.int32)
  face_vtx      = np.empty(4*n_face_slab, dtype=np.int32)
  face_pe       = np.empty((n_face_slab, 2), dtype=np.int32)
  compute_all_ngon_connectivity(vtx_slabs, n_vtx_zone, face_gnum, face_vtx, face_pe)

  #>> PartToBlock pour ordonner et equidistribuer les faces
  part_to_block = PDM.PartToBlock(comm, [face_gnum], None, partN=1, t_distrib=0, t_post=0, t_stride=1)
  #>>> Premier echange pour le ParentElements
  pfield_stride2 = {"NGonPE" : [face_pe.ravel()]}
  stride2 = [2*np.ones(n_face_slab, dtype='int32')]
  dfield_stride2 = dict()
  part_to_block.PartToBlock_Exchange(dfield_stride2, pfield_stride2, stride2)
  #>>> Deuxieme echange pour l'ElementConnectivity
  pfield_stride4 = {"NGonFaceVtx" : [face_vtx]}
  stride4 = [4*np.ones(n_face_slab,  dtype='int32')]
  dfield_stride4 = dict()
  part_to_block.PartToBlock_Exchange(dfield_stride4, pfield_stride4, stride4)

  face_pe  = dfield_stride2["NGonPE"].reshape(-1, 2)
  face_vtx = dfield_stride4["NGonFaceVtx"]
  n_face_loc = face_pe.shape[0]
  face_distri = part_to_block.getDistributionCopy()
  face_vtx_idx = 4*np.arange(face_distri[i_rank], face_distri[i_rank]+n_face_loc+1)

  ngon = I.newElements('NGonElements', 'NGON', face_vtx, [1, n_face_tot])
  I.newDataArray("ElementStartOffset", face_vtx_idx, parent=ngon)
  I.newIndexArray('ElementConnectivity#Size', [4*n_face_tot], parent=ngon)
  I.newParentElements(face_pe, parent=ngon)

  return ngon

def convert_s_to_u(disttree_s, comm, bc_output_loc="FaceCenter", gc_output_loc="FaceCenter"):

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

  return disttree_u
