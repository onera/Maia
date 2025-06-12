import numpy as np
from mpi4py import MPI

from maia.typing import *
import maia.pytree      as PT
import maia.pytree.pred as PTp
import maia.pytree.maia as MT

from maia           import npy_pdm_gnum_dtype  as pdm_dtype
from maia.algo.dist import ngon_tools
from maia.transfer  import protocols as EP
from maia.utils     import np_utils, par_utils, s_numbering
from maia.utils     import logging as mlog

IS_BAR  = PTp.is_elmt_of_type('BAR_2')
IS_TRI  = PTp.is_elmt_of_type('TRI_3')
IS_QUAD = PTp.is_elmt_of_type('QUAD_4')

def _extend_pr(pr_node:CGNSTree, val):
  """ Add a dimension to PR-like arrays with the specified values"""
  pr2d = PT.get_np_value(pr_node)
  assert pr2d.shape[0] == 2
  pr3d = np.append(pr2d, np.array(val, pr2d.dtype).reshape((1,-1), order='F'), axis=0)
  PT.set_value(pr_node, pr3d)

def _create_surfacic_bcs(zone, n_face2d, first_id, extrusion_vector, ksubset_as, comm):

  distrib_idx = par_utils.uniform_distribution(n_face2d, comm)
  pl_former   = np.arange(distrib_idx[0], distrib_idx[1], dtype=zone[1].dtype).reshape((1,-1), order='F') + first_id
  pl_extruded = np.arange(distrib_idx[0], distrib_idx[1], dtype=zone[1].dtype).reshape((1,-1), order='F') + first_id + n_face2d

  # Note : Subset are created as EdgeCenter right now, because they calling function convert it to FaceCenter after
  if ksubset_as == 'GC':
    # > Generate GridConnectivity between the two planes
    # Remark: former NGon is the first GridConnectivity and the duplicated one the second one
    zgc = PT.update_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
    gc1_name = 'InitialSurface'
    gc2_name = 'ExtrudedSurface'
    gc1 = PT.new_GridConnectivity(name=gc1_name, donor_name=PT.get_name(zone),
                                  type='Abutting1to1', loc='EdgeCenter',
                                  point_list=pl_former,
                                  point_list_donor=pl_extruded,
                                  parent=zgc)
    PT.new_GridConnectivityProperty({"translation": np.array(extrusion_vector, dtype=np.float64)}, parent=gc1)
    MT.new_Distribution({'Index' : distrib_idx}, parent=gc1)

    gc2 = PT.new_GridConnectivity(name=gc2_name, donor_name=PT.get_name(zone),
                                  type='Abutting1to1', loc='EdgeCenter',
                                  point_list=pl_extruded,
                                  point_list_donor=pl_former,
                                  parent=zgc)
    PT.new_GridConnectivityProperty({"translation": -np.array(extrusion_vector, dtype=np.float64)}, parent=gc2)
    MT.new_Distribution({'Index' : distrib_idx.copy()}, parent=gc2)

    PT.new_Descriptor("GridConnectivityDonorName", gc2_name, parent=gc1)
    PT.new_Descriptor("GridConnectivityDonorName", gc1_name, parent=gc2)

  elif ksubset_as == 'BC':
    zbc = PT.update_child(zone, 'ZoneBC', 'ZoneBC_t')
    bc1 = PT.new_BC(name='InitialSurface', type='FamilySpecified', point_list=pl_former,
                    loc='EdgeCenter', family='InitialSurface', parent=zbc)
    bc2 = PT.new_BC(name='ExtrudedSurface', type='FamilySpecified', point_list=pl_extruded,
                    loc='EdgeCenter', family='ExtrudedSuface', parent=zbc)
    MT.new_Distribution({'Index' : distrib_idx}, parent=bc1)
    MT.new_Distribution({'Index' : distrib_idx.copy()}, parent=bc2)


def _nodes_duplication(zone, extrusion_vector, comm, as_last=True):
  """
  Internal function used by _extrusion_2d_u_ngon, _extrusion_2d_u_elem and _extrusion_2d_s
  to create the duplicated nodes needed to generate the second plan
  """
  # > Define new distribution
  distrib_vtx_n    = MT.get_Distribution(zone, 'Vertex')
  distrib_vtx      = PT.get_value(distrib_vtx_n)
  new_distrib_vtx  = par_utils.uniform_distribution(2*distrib_vtx[2],  comm)
  # > Change value of Z or Theta coordinates in part_data
  coords = PT.Zone.coordinates(zone)
  initial = coords._asdict()
  extruded = {name: coord + extru for  extru, (name, coord) in zip(extrusion_vector, initial.items())}
  
  if not as_last: # For S meshes, change order depending of extrusion direction to keep i,j,k direct
    initial, extruded = extruded, initial

  # We can do two BlockToBlock, it is faster than a PartToBlock (we should do a function from this pattern)
  dist_data = {key: np.empty(new_distrib_vtx[1]-new_distrib_vtx[0]) for key in initial}
  distri_out = distrib_vtx.copy()
  n_vtx = distrib_vtx[2]
  distri_out[0] = max(min(new_distrib_vtx[0], n_vtx), 0) - 0
  distri_out[1] = max(min(new_distrib_vtx[1], n_vtx), 0) - 0
  stop = distri_out[1]-distri_out[0]
  btb = EP.BlockToBlock(distrib_vtx, distri_out, comm)
  for key in initial:
    btb.exchange_inplace(initial[key], dist_data[key][:stop])
  
  distri_out[0] = max(min(new_distrib_vtx[0], 2*n_vtx), n_vtx) - n_vtx
  distri_out[1] = max(min(new_distrib_vtx[1], 2*n_vtx), n_vtx) - n_vtx
  btb = EP.BlockToBlock(distrib_vtx, distri_out, comm)
  for key in initial:
    btb.exchange_inplace(extruded[key], dist_data[key][stop:])

  # > Update coordinates values
  for name, new_val in dist_data.items():
    coord_n = PT.get_child_from_predicates(zone, f'GridCoordinates_t/{name}')
    PT.set_value(coord_n, new_val)
  # > Update vertex distribution
  PT.set_value(distrib_vtx_n, new_distrib_vtx)
    

def _determine_mesh_orientation(zone, extrusion_vector, comm):
  # Remark: to determine orientation of faces, we need to test the sign of
  #         scalar product between extrusion vector and the first face in
  #         the current proc
  # > Get the third first nodes of the first face in current proc
  if PT.Zone.Type(zone) == 'Structured':
    nodes_of_first_face = s_numbering.ij_to_index(np.array([1, 2, 1]), np.array([1, 1, 2]), PT.Zone.VertexSize(zone))
  else:
    if PT.Zone.has_ngon_elements(zone): #U-NGon zone
      first_elem_2d = PT.Zone.NGonNode(zone)
    else: #U-Elem zone
      first_elem_2d = PT.Zone.get_ordered_elements_per_dim(zone)[2][0]
    first_elem_2d_ec = PT.get_value(PT.get_child_from_name(first_elem_2d, 'ElementConnectivity'))
    if len(first_elem_2d_ec) > 0:
      nodes_of_first_face = first_elem_2d_ec[0:3]
    else:
      nodes_of_first_face = np.empty(0, int)
  # > Get coordinates of nodes of the first face
  coords = PT.Zone.coordinates(zone)
  dist_coords_data = coords._asdict()
  distrib_vtx = MT.distribution_value(zone, 'Vertex')
  part_coords_data = EP.block_to_part(dist_coords_data, distrib_vtx, nodes_of_first_face-1, comm)
  align = 0
  # > Compute scalar product
  # **NB** Here we test only one face per rank. Maybe we should do it on all faces and check that is same everywhere ?
  #        We need compute_normals to be implemented for elts to do this
  if len(nodes_of_first_face) > 0:
    n1 = np.array([part_coords_data[name][0] for name in coords._fields])
    n2 = np.array([part_coords_data[name][1] for name in coords._fields])
    n3 = np.array([part_coords_data[name][2] for name in coords._fields])
    # normal vector = a^b with a = n1_n2 and b = n2_n3
    normal_vec = np.cross(n2-n1, n3-n2)
    # ps = c.extrusion_vector
    ps = np.dot(normal_vec, extrusion_vector)
    # >>> d. change order of one of two
    if ps>0: # Need to change initial faces
      align = 1
    elif ps<0: # Need to change duplicated faces
      align = -1
  if comm.allreduce(align >= 0, MPI.LAND):
    return True
  elif comm.allreduce(align <= 0, MPI.LAND):
    return False
  else:
    raise ValueError('Faces of 2D mesh are not uniformly oriented !')


def _reorder_ngon_ec(ngon_n):
  """
  Internal function used by _ngon_duplication to reorder the NGonNode connectivity
  to be exited normal
  """
  MT.Element.connectivity(ngon_n)._inner_flip()
    

def _ngon_duplication(zone, comm, align=True):
  """
  Internal function used by _extrusion_2d_u_ngon to create the duplicated NGonNode
  needed to generate the second plan
  
  Remark : no need to change the ElementStartOffset
  """
  # > Get number of 2D cells (face) and vertices
  n_cell_2d = PT.Zone.n_cell(zone)
  n_vtx_2d  = PT.Zone.n_vtx(zone)
  # > Add ParentElements to NGon node
  ngon_n = PT.Zone.NGonNode(zone)
  distrib_elem = MT.distribution_value(ngon_n, 'Element')
  er = PT.Element.Range(ngon_n)
  pe = np.zeros((distrib_elem[1]-distrib_elem[0],2), dtype=er.dtype)
  pe[:,0] = np.arange(distrib_elem[0], distrib_elem[1]) + er[0] + 2*n_cell_2d
  PT.new_child(ngon_n, 'ParentElements', 'DataArray_t', pe)
  # > Duplicate NGon
  ngon_bis_n = PT.deep_copy(ngon_n)
  PT.set_name(ngon_bis_n, f'{PT.get_name(ngon_n)}_bis')
  # > Update ElementRange and ElementConnectivity
  # on suppose que l'on a deja tous les elements 1D et 2D de définis dans le CGNS
  er = PT.get_child_from_name(ngon_bis_n, 'ElementRange')[1]
  ec = PT.get_child_from_name(ngon_bis_n, 'ElementConnectivity')[1]
  er += n_cell_2d
  ec += n_vtx_2d
  if align: # Need to change Ngon
    _reorder_ngon_ec(ngon_n)
  else: # Need to change duplicated Ngon
    _reorder_ngon_ec(ngon_bis_n)

  PT.add_child(zone, ngon_bis_n)


def _extrude_bar_to_ngon(bar, n_vtx, n_cell, align=True):
  """
  Internal function used by _extrusion_2d_u_ngon to create face by extrusion of BAR elements
  """
  distrib_elem = MT.distribution_value(bar, 'Element')
  ec_n = PT.get_child_from_name(bar, 'ElementConnectivity')

  # > Change value: 3 => 22
  PT.get_value(bar)[0] = 22
  # > Create ElementStartOffset
  eso = 4*np.arange(distrib_elem[0], distrib_elem[1]+1, dtype=ec_n[1].dtype)
  PT.new_child(bar, 'ElementStartOffset', 'DataArray_t', eso)
  # > Update ElementConnectivity
  first_nodes  = ec_n[1][0::2]
  second_nodes = ec_n[1][1::2]
  third_nodes  = second_nodes + n_vtx
  fourth_nodes = first_nodes  + n_vtx
  if align:
    new_ec = np_utils.interweave_arrays([second_nodes, first_nodes, fourth_nodes, third_nodes])
  else:
    new_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes])
  PT.set_value(ec_n, new_ec)
  # > Create ElementConnectivity distribution
  MT.new_Distribution({'ElementConnectivity' : 4*distrib_elem}, parent=bar)
  # > Update PE
  # Remark: new cells are the former faces because we keep the order
  #         so we just have to shift there values of 2*n_cell
  pe = PT.get_child_from_name(bar, 'ParentElements')[1]
  pe += 2*n_cell * (pe > 0)
    
    
def _merge_ngons(zone, comm):
  """ Internal function used by _extrusion_2d_u_ngon to create the merged NGonNode """
  part_ec = []
  part_pe0 = []
  part_pe1 = []
  ln_to_gn_elem_l = []
  ngon_nodes = PT.Zone.get_ordered_elements_per_dim(zone)[2]
  for ngon_n in ngon_nodes:
    assert PT.Element.CGNSName(ngon_n) == 'NGON_n'
    er  = PT.find_child_from_name(ngon_n, 'ElementRange')[1]
    ec  = PT.find_child_from_name(ngon_n, 'ElementConnectivity')[1]
    eso = PT.find_child_from_name(ngon_n, 'ElementStartOffset')[1]
    pe  = PT.find_child_from_name(ngon_n, 'ParentElements')[1]
    distrib_elem = MT.distribution_value(ngon_n, 'Element')
    part_ec.append((np.diff(eso).astype(np.int32), ec))
    part_pe0.append(pe[:,0])
    part_pe1.append(pe[:,1])
    ln_to_gn_elem_l.append(np.arange(distrib_elem[0]+er[0]-1, distrib_elem[1]+er[0]-1, dtype=pdm_dtype))
  # > Define new ElementRange
  # Warning : ne fonctionne pas si il y a des 'NODE' dans l'arbre !
  #           si tous les NODE sont avant, il faut faire démarrer l'ER au total des éléments de NODE
  #           si tous les NODE sont après, rien à changer
  #           si les NODE ne sont pas spécialement ordonnés, faire une réorganisation en entrée de 'extrusion_2d' ?
  n_faces = sum(PT.Element.Size(ng) for ng in ngon_nodes)
  new_er = np.array([1, n_faces], zone[1].dtype)
  # > Define new Element distribution
  new_distrib_elem = par_utils.uniform_distribution(n_faces,  comm)
  # > Exchange to define new ElementStartOffset, ElementConnectivity and ParentElements
  GI = EP.GlobalIndexer(new_distrib_elem, ln_to_gn_elem_l, comm)
  new_diff_eso, new_ec = GI.Put_v(part_ec)
  dn = new_distrib_elem[1] - new_distrib_elem[0]
  new_pe = np.empty((dn, 2), order='F', dtype=zone[1].dtype)
  GI.Put(part_pe0, new_pe[:,0])
  GI.Put(part_pe1, new_pe[:,1])
  # > Define new ElementConnectivity distribution
  new_distrib_ec = par_utils.dn_to_distribution(new_diff_eso.sum(), comm)
  new_eso = np_utils.sizes_to_indices(new_diff_eso) + new_distrib_ec[0]
  # > Delete old ngons
  PT.rm_children_from_predicate(zone, PTp.is_elmt_of_type('NGON_n'))
  # > Create new NGon node
  new_ngon_n = PT.new_NGonElements(erange=new_er, eso=new_eso, ec=new_ec, pe=new_pe, parent=zone)
  MT.new_Distribution({'Element' : new_distrib_elem, 'ElementConnectivity' : new_distrib_ec}, parent=new_ngon_n)
    
def _extrusion_2d_s(zone, extrusion_vector, comm, align, ksubset_as):
  """
  Internal function used by extrusion_2d to extrude a 2D structured mesh
  in the direction of the extrusion vector
  """
  
  # 0/ Global information
  n_vtx  = PT.Zone.n_vtx(zone)

  # 1/ Duplication of nodes to generate the second plan
  _nodes_duplication(zone, extrusion_vector, comm, align)
  
  # 2/ Manage K-plans
  pr_former = np.ones((3,2), order='F', dtype=zone[1].dtype)
  pr_former[0:2,1] = PT.Zone.VertexSize(zone)
  pr_extruded = pr_former.copy()
  pr_extruded[2,:] = 2
  if not align: # Swap former / extruded if needed
    pr_former, pr_extruded = pr_extruded, pr_former
  distrib_idx = par_utils.uniform_distribution(n_vtx, comm)
  if ksubset_as == 'GC':
    # > Generate GridConnectivity between the two planes
    zgc = PT.update_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
    gc1_name = 'InitialSurface'
    gc2_name = 'ExtrudedSurface'
    gc1 = PT.new_GridConnectivity1to1(name=gc1_name, donor_name=PT.get_name(zone),
                                      point_range=pr_former,
                                      point_range_donor=pr_extruded,
                                      transform=[1,2,3],
                                      parent=zgc)
    PT.new_GridConnectivityProperty({"translation": np.array(extrusion_vector, dtype=np.float64)}, parent=gc1)
    MT.new_Distribution({'Index' : distrib_idx}, parent=gc1)

    gc2 = PT.new_GridConnectivity1to1(name=gc2_name, donor_name=PT.get_name(zone),
                                      point_range=pr_extruded,
                                      point_range_donor=pr_former,
                                      transform=[1,2,3],
                                      parent=zgc)
    PT.new_GridConnectivityProperty({"translation": -np.array(extrusion_vector, dtype=np.float64)}, parent=gc2)
    MT.new_Distribution({'Index' : distrib_idx.copy()}, parent=gc2)

    PT.new_Descriptor("GridConnectivityDonorName", gc2_name, parent=gc1)
    PT.new_Descriptor("GridConnectivityDonorName", gc1_name, parent=gc2)

  elif ksubset_as == 'BC':
    zbc = PT.update_child(zone, 'ZoneBC', 'ZoneBC_t')
    bc1 = PT.new_BC(name='InitialSurface', type='FamilySpecified', point_range=pr_former, family='InitialSurface', parent=zbc)
    bc2 = PT.new_BC(name='ExtrudedSurface', type='FamilySpecified', point_range=pr_extruded, family='ExtrudedSuface', parent=zbc)
    MT.new_Distribution({'Index' : distrib_idx}, parent=bc1)
    MT.new_Distribution({'Index' : distrib_idx.copy()}, parent=bc2)



def _extrusion_2d_u_ngon(zone, extrusion_vector, comm, ksubset_as):
  """
  Internal function used by extrusion_2d to extrude a 2D unstructured mesh describe by edges
  in the direction of the extrusion vector in cartesian and cylindrical coordinates.
  """
  
  # 0/ Global information
  n_vtx  = PT.Zone.n_vtx(zone)
  n_cell = PT.Zone.n_cell(zone)
  n_edges = sum(PT.Element.Size(e) for e in PT.get_children_from_predicate(zone, IS_BAR))

  # 0bis / Ensure we have both EdgeElements/ParentElements and NGonElements
  if not PT.Zone.has_ngon_elements(zone):
    ngon_tools.edge_pe_to_ngon(zone, comm)
  if PT.get_node_from_predicates(zone, [IS_BAR, 'ParentElements']) is None:
    ngon_tools.ngon_to_edge_pe(zone, comm)
  
  # 1/ Duplication of nodes to generate the second plan
  _nodes_duplication(zone, extrusion_vector, comm)
  
  # 1bis/ Determine the mesh orientation
  align = _determine_mesh_orientation(zone, extrusion_vector, comm)
  
  # 2/ Create faces of the second plan
  _ngon_duplication(zone, comm, align)
  
  # 3/ Extrude Bar to NGon
  for bar in PT.get_nodes_from_predicate(zone, IS_BAR):
    _extrude_bar_to_ngon(bar, n_vtx, n_cell, align)
  
  # 4/ Merge all NGon nodes
  _merge_ngons(zone, comm)
  
  # 5/ Manage K-plans
  _create_surfacic_bcs(zone, n_cell, n_edges+1, extrusion_vector, ksubset_as, comm)



def _extrude_tri_to_prism_and_tris(tri, num, n_vtx, n_cell, er_max, align=True):
  """
  Internal function used by _extrusion_2d_u_elem to create face by extrusion of TRI elements
  
  Remark : no need to change the element distribution
  """
  # > Change value: 5 => 14
  PT.get_value(tri)[0] = 14
  # > Change name
  PT.set_name(tri, f'PENTA_6.{num}')
  # > Update ElementConnectivity
  ec_n = PT.get_child_from_name(tri, 'ElementConnectivity')
  first_nodes  = ec_n[1][0::3]
  second_nodes = ec_n[1][1::3]
  third_nodes  = ec_n[1][2::3]
  fourth_nodes = first_nodes  + n_vtx
  fifth_nodes  = second_nodes + n_vtx
  sixth_nodes  = third_nodes  + n_vtx
  ec_penta = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes, fifth_nodes, sixth_nodes])
  PT.set_value(ec_n, ec_penta)
  # > Treat new tri nodes
  # >>> Change ec of new TRIs with preservation of good orientation
  if align:
    new_tri1_ec = np_utils.interweave_arrays([first_nodes, third_nodes, second_nodes])
    new_tri2_ec = np_utils.interweave_arrays([fourth_nodes, fifth_nodes, sixth_nodes])
  else:
    new_tri1_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes])
    new_tri2_ec = np_utils.interweave_arrays([fourth_nodes, sixth_nodes, fifth_nodes])
  
  new_tri1_er = np.array([er_max+1, er_max+PT.Element.Size(tri)], new_tri1_ec.dtype)
  new_tri2_er = new_tri1_er + n_cell

  new_tri1 = PT.new_Elements(f'TRI_3.{num}a', 'TRI_3', erange=new_tri1_er, econn=new_tri1_ec)
  new_tri2 = PT.new_Elements(f'TRI_3.{num}b', 'TRI_3', erange=new_tri2_er, econn=new_tri2_ec)
  MT.new_Distribution({'Element' : MT.distribution_value(tri, 'Element').copy()}, new_tri1)
  MT.new_Distribution({'Element' : MT.distribution_value(tri, 'Element').copy()}, new_tri2)
  return (new_tri1, new_tri2)


def _extrude_quad_to_hexa_and_quads(quad, num, n_vtx, n_cell, er_max, align=True):
  """
  Internal function used by _extrusion_2d_u_elem to create face by extrusion of QUAD elements
  
  Remark : no need to change the element distribution
  """
  # > Change value: 7 => 17
  PT.get_value(quad)[0] = 17
  # > Change name
  PT.set_name(quad, f'HEXA_8.{num}') #ou on s'appuie sur le nom initial de l'élément ?
  # > Update ElementConnectivity
  ec_n = PT.get_child_from_name(quad, 'ElementConnectivity')
  first_nodes   = ec_n[1][0::4]
  second_nodes  = ec_n[1][1::4]
  third_nodes   = ec_n[1][2::4]
  fourth_nodes  = ec_n[1][3::4]
  fifth_nodes   = first_nodes  + n_vtx
  sixth_nodes   = second_nodes + n_vtx
  seventh_nodes = third_nodes  + n_vtx
  eighth_nodes  = fourth_nodes + n_vtx
  ec_hexa = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes, fifth_nodes, sixth_nodes, seventh_nodes, eighth_nodes])
  PT.set_value(ec_n, ec_hexa)
  # > Treat new quad nodes
  # >>> Change ec of new QUADs with preservation of good orientation
  if align:
    new_quad1_ec = np_utils.interweave_arrays([first_nodes, fourth_nodes, third_nodes, second_nodes])
    new_quad2_ec = np_utils.interweave_arrays([fifth_nodes, sixth_nodes, seventh_nodes, eighth_nodes])
  else:
    new_quad1_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes])
    new_quad2_ec = np_utils.interweave_arrays([fifth_nodes, eighth_nodes, seventh_nodes, sixth_nodes])

  new_quad1_er = np.array([er_max+1, er_max+PT.Element.Size(quad)], new_quad1_ec.dtype)
  new_quad2_er = new_quad1_er + n_cell

  new_quad1 = PT.new_Elements(f'QUAD_4.{num}a', 'QUAD_4', erange=new_quad1_er, econn=new_quad1_ec)
  new_quad2 = PT.new_Elements(f'QUAD_4.{num}b', 'QUAD_4', erange=new_quad2_er, econn=new_quad2_ec)
  MT.new_Distribution({'Element' : MT.distribution_value(quad, 'Element').copy()}, new_quad1)
  MT.new_Distribution({'Element' : MT.distribution_value(quad, 'Element').copy()}, new_quad2)
  return (new_quad1, new_quad2)

def _extrude_bar_to_quad(bar, num, n_vtx, align=True):
  """
  Internal function used by _extrusion_2d_u_elem to create face by extrusion of BAR elements
  
  Remark : no need to change the element distribution
  """
  # > Change value: 3 => 7
  PT.get_value(bar)[0] = 7
  # > Change name
  PT.set_name(bar, f'QUAD_4.{num}')
  # > Update ElementConnectivity
  ec_n = PT.get_child_from_name(bar, 'ElementConnectivity')
  first_nodes  = ec_n[1][0::2]
  second_nodes = ec_n[1][1::2]
  third_nodes  = second_nodes + n_vtx
  fourth_nodes = first_nodes  + n_vtx
  if align:
    new_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes])
  else:
    new_ec = np_utils.interweave_arrays([second_nodes, first_nodes, fourth_nodes, third_nodes])
  PT.set_value(ec_n, new_ec)


def _extrusion_2d_u_elem(zone, extrusion_vector, comm, ksubset_as):
  """
  Internal function used by extrusion_2d to extrude a 2D unstructured mesh describe by elements
  in the direction of the extrusion vector in cartesian and cylindrical coordinates.
  """
  
  # 0/ Global information
  n_vtx  = PT.Zone.n_vtx(zone)
  n_cell = PT.Zone.n_cell(zone)
  er_max = max(PT.Element.Range(e)[1] for e in PT.get_children_from_label(zone, 'Elements_t'))
  first_id = er_max + 1

  # 1/ Duplication of nodes to generate the second plan
  _nodes_duplication(zone, extrusion_vector, comm)
  
  # 1bis/ Determine the mesh orientation
  align = _determine_mesh_orientation(zone, extrusion_vector, comm)

  # Note : in step 2, we ensure to create all faces from initial plan, then all faces
  # from extruded plan, because it makes generation of FaceCenter BC/GC easier
  # 2/ Extrude Tri to Prism
  new_face_elts = []
  for num, tri in enumerate(PT.get_children_from_predicate(zone, IS_TRI)):
    new_tri1, new_tri2 = _extrude_tri_to_prism_and_tris(tri, num, n_vtx, n_cell, er_max, align=align)
    new_face_elts.extend([new_tri1, new_tri2])
    er_max += PT.Element.Size(tri)
  
  # 3/ Extrude Quad to Hexa
  for num, quad in enumerate(PT.get_children_from_predicate(zone, IS_QUAD)):
    new_quad1, new_quad2 = _extrude_quad_to_hexa_and_quads(quad, num, n_vtx, n_cell, er_max, align=align)
    new_face_elts.extend([new_quad1, new_quad2])
    er_max += PT.Element.Size(quad)

  for elt in sorted(new_face_elts, key=lambda e: PT.Element.Range(e)[0]):
    PT.add_child(zone, elt)
  
  # 4/ Extrude Bar to Quad
  for num, bar in enumerate(PT.get_nodes_from_predicate(zone, IS_BAR)):
    _extrude_bar_to_quad(bar, num, n_vtx, align=align)
  
  # 5/ Manage K-plans
  _create_surfacic_bcs(zone, n_cell, first_id, extrusion_vector, ksubset_as, comm)


def _pl_and_data_vtx_duplication(pl, distrib_idx, n_vtx_2d, data, comm):
  """
  Internal function used by _extrusion_2d to create the duplicated PointList 
  and associated datas needed to extented it to the second plan
  """
  new_distrib_idx  = par_utils.uniform_distribution(2*distrib_idx[2],  comm)
  # > Duplicate data in part_data
  part_data = {name: [value,value] for name, value in data.items()}
  if pl is not None:
    part_data['PointList'] = [pl[0], pl[0]+n_vtx_2d]

  dist_data = {key: np.empty(new_distrib_idx[1]-new_distrib_idx[0], val[0].dtype) for key,val in part_data.items()}
  distri_out = distrib_idx.copy()
  end = distrib_idx[2]
  distri_out[0] = max(min(new_distrib_idx[0], end), 0) - 0
  distri_out[1] = max(min(new_distrib_idx[1], end), 0) - 0
  stop = distri_out[1]-distri_out[0]
  btb = EP.BlockToBlock(distrib_idx, distri_out, comm)
  for key in part_data:
    btb.exchange_inplace(part_data[key][0], dist_data[key][:stop])

  distri_out[0] = max(min(new_distrib_idx[0], 2*end), end) - end
  distri_out[1] = max(min(new_distrib_idx[1], 2*end), end) - end
  btb = EP.BlockToBlock(distrib_idx, distri_out, comm)
  for key in part_data:
    btb.exchange_inplace(part_data[key][1], dist_data[key][stop:])
  
  if pl is None:
    dist_pl = None
  else:
    dist_pl = dist_data.pop('PointList').reshape((1,-1), order='F')
  # > Return
  return new_distrib_idx, dist_pl, dist_data
    

def extrude(dist_tree: CGNSDistTree,
            extrusion_vector: Sequence[float],
            comm: MPIComm,
            ksubset_as: Literal['GC', 'BC'] = 'GC',
            dupl_vtx_data: bool = False) -> None:
  """ Extrude a 2D mesh in the provided direction.

  The resulting mesh will be a 3D mesh with one layer of cells. Existing subsets and containers
  such as ``BC_t``, ``FlowSolution_t``, ``ZoneSubRegion_t``, etc. are updated following these rules:

  - EdgeCenter regions (lineic) become FaceCenter regions (surfacic),
  - CellCenter regions (surfacic) become CellCenter regions (volumic),
  - Vertex regions remain the same if ``dupl_vtx_data`` is False. Otherwise,
    they are extended with the corresponding extruded vertices, the fields values beeing simply
    duplicated. This choice does not apply to BC and GC vertex subsets, which are always extended.

  In addition, new FaceCenter subsets are created for the initial surface and its
  extruded counterpart. These subsets can be created as periodic ``GridConnectivity_t`` nodes
  (using ``ksubset_as == 'GC'``) or as ``BC_t`` nodes (using ``ksubset_as == 'BC'``).

  Input tree is modified inplace.

  Args:
    dist_tree (CGNSDistTree): Input 2D distributed tree
    extrusion_vector (array of 3 floats): extrusion axis, which can be any non zero vector
    comm      (MPIComm)     : MPI communicator
    ksubset_as (str): Set kind of surfacic subset created for the initial and extruded planes.
                          Default to ``GC``.
    dupl_vtx_data (bool)    : Enable duplication of vertex located fields (see above). Default to ``False``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #extrude@start
        :end-before: #extrude@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  if not ksubset_as in ['BC', 'GC']:
    raise ValueError(f"'ksubset_as' is {ksubset_as} but only 'GC' and 'BC' are allowed !")
  
  zone_to_distrib_vtx = dict()
  zone_to_align = dict()
  for zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    zone = PT.find_node_from_path(dist_tree, zone_path)
    if not PT.Zone.CellDimension(zone) == 2:
      raise ValueError("Only 2D zones are supported in this function")

    distrib_vtx_2d_n = MT.distribution_value(zone, 'Vertex').copy()
    zone_to_distrib_vtx[zone_path] = distrib_vtx_2d_n

    coord_n = PT.get_child_from_label(zone, 'GridCoordinates_t')
    coords = PT.Zone.coordinates(zone)
    for coord_name, coord_val in coords._asdict().items():
      if coord_val is None:
        PT.new_DataArray(coord_name, np.zeros_like(coords[0]), parent=coord_n)

    if PT.Zone.Type(zone) == 'Structured':
      zone_to_align[zone_path] = _determine_mesh_orientation(zone, extrusion_vector, comm)

  for base, zone in PT.get_children_from_labels(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
      
    zone_path = f'{base[0]}/{zone[0]}'
    distrib_vtx_2d = zone_to_distrib_vtx[zone_path]
    cell_offset_2d = PT.Zone.get_elt_range_per_dim(zone)[2][0]
    n_vtx_2d = distrib_vtx_2d[2]
    
    # Generate new vertices and Elements
    if PT.Zone.Type(zone) == 'Structured':
      _extrusion_2d_s(zone, extrusion_vector, comm, zone_to_align[zone_path], ksubset_as=ksubset_as)
    elif PT.Zone.Type(zone) == 'Unstructured':
      all_element_types = set([PT.Element.CGNSName(e) for e in PT.get_children_from_label(zone, 'Elements_t')])
      if all_element_types <= {'NODE', 'BAR_2', 'NGON_n'}:
        _extrusion_2d_u_ngon(zone, extrusion_vector, comm, ksubset_as=ksubset_as)
      elif all_element_types <= {'NODE', 'BAR_2', 'TRI_3', 'QUAD_4'}:
        _extrusion_2d_u_elem(zone, extrusion_vector, comm, ksubset_as=ksubset_as)
      else:
        raise ValueError(f'Zone {PT.get_name(zone)} is neither full NGON or composed only of TRI and QUAD elements !')
    else:
      raise ValueError(f'Zone {PT.get_name(zone)} is neither structured nor unstructured !')

    # Update zone dims
    # Remark: in extrusion, no need to change nb_cell because the new 3D cells are the 
    #         former 2D ones extruded
    if PT.Zone.Type(zone) == 'Unstructured':
      zone_dims = PT.get_np_value(zone)
      zone_dims[0][0] *= 2
    else:
      _extend_pr(zone, [2,1,0])

    if PT.Zone.has_ngon_elements(zone):
      cell_offset_3d = PT.Element.Range(PT.Zone.NGonNode(zone))[1] + 1
    else:
      cell_offset_3d = cell_offset_2d # For elt meshes, cell pl dont need to be updated
    
    # Update containers
    is_container = PTp.label_in(['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCDataSet_t'])
    is_subset    = PTp.label_in(['BC_t', 'GridConnectivity_t', 'GridConnectivity1to1_t'])
    has_pl = PTp.has_child_of_name('PointList')
    has_pr = PTp.has_child_of_name('PointRange')
    is_partial = has_pl | has_pr
    
    # > CellCenter -> Shift to refer cells ids
    is_container_cell = (is_container | is_subset) & is_partial & PTp.has_location('CellCenter')
    for container in PT.get_nodes_from_predicate(zone, is_container_cell, depth=3):
      if PT.Zone.Type(zone) == 'Unstructured':
        assert PT.get_child_from_label(container, 'IndexRange_t') is None, "PointRange not supported for U zones"
        assert PT.get_child_from_name(container, 'PointListDonor') is None, "CellCenter GC are not supported for U zones"
        pl = PT.get_np_value(PT.find_child_from_name(container, 'PointList'))
        pl += cell_offset_3d - cell_offset_2d
      else:
        assert PT.get_child_from_label(container, 'IndexArray_t') is None, "PointList not supported for S zones"
        for pr_n in PT.get_children_from_label(container, 'IndexRange_t'):
          _extend_pr(pr_n, [1,1])

    # > *FaceCenter -> should not exist on 2d mesh, remove it
    is_container_face = (is_container | is_subset) & PT.pred.has_location('*FaceCenter')
    container_face_l = PT.get_nodes_from_predicate(zone, is_container_face, depth=3, explore='deep')
    if len(container_face_l) > 0:
      cnt_names = [PT.get_name(n) for n in container_face_l]
      msg = f"The following subsets have been removed from the input 2D mesh, because GridLocation == *FaceCenter" \
            f" is not allowed by the CGNS norm on 2d meshes : {cnt_names}"
      mlog.error(msg)
      PT.rm_nodes_from_predicate(zone, is_container_face, depth=3)

    # > *EdgeCenter -> becomes *FaceCenter (no need to change their PointList, but PR must be extended)
    is_container_edge = (is_container | is_subset) & PT.pred.has_location('*EdgeCenter')
    for container in PT.get_nodes_from_predicate(zone, is_container_edge, depth=3, explore='deep'):
      if PT.Zone.Type(zone)  == 'Unstructured':
        PT.update_child(container, 'GridLocation', value='FaceCenter')
      else:
        cur_dir = PT.Subset.GridLocation(container)[0] 
        PT.update_child(container, 'GridLocation', value=f'{cur_dir}FaceCenter')
        for pr_n in PT.get_children_from_label(container, 'IndexRange_t'):
          _extend_pr(pr_n, [1,1])

    # > Vertex -> Subsets (BCs, GC) must be always extended, but containers depends on dupl_vtx_data
    # It seems easier to treat data first, because data can require the initial PL or Distribution
    is_vertex = PTp.has_location('Vertex')
    
    if dupl_vtx_data:
      # Duplicate data in Vertex containers. PL/PR must be extended if present in container. If containers
      # are full, we don't need to add PR/PL since they are still full after duplication.
      for container in PT.get_children_from_predicate(zone, is_container & is_vertex):
        if has_pl(container):
          maybe_pl = PT.find_child_from_name(container, 'PointList')[1]
          distrib_idx = MT.distribution_value(container, 'Index')
        elif PT.get_label(container) == 'ZoneSubRegion_t': # Related ZSR *or* PR defined ZSR
          maybe_pl = None
          zsr_extent = PT.Subset.ZSRExtent(container, zone)
          extent_node = PT.find_node_from_path(zone, zsr_extent)
          distrib_idx = MT.distribution_value(extent_node, 'Index')
        else: # Full containers
          maybe_pl = None
          distrib_idx = distrib_vtx_2d
        data = {PT.get_name(n) : PT.get_value(n) for n in PT.get_children_from_label(container, 'DataArray_t')}
        new_distrib_idx, new_pl, new_data = _pl_and_data_vtx_duplication(maybe_pl, distrib_idx, n_vtx_2d, data, comm)
        if has_pl(container): # Update PointList + Distribution
          PT.update_child(container, 'PointList', value=new_pl)
          MT.new_Distribution({'Index' : new_distrib_idx}, container)
        elif has_pr(container):
          _extend_pr(PT.find_child_from_name(container, 'PointRange'), [1, 2])
          MT.new_Distribution({'Index' : new_distrib_idx}, container)
        for name, value in new_data.items():
          PT.set_value(PT.find_child_from_name(container, name), value)
      
      # Specific treatment of BCDS (they are skipped above because of get_children).
      # Duplicate data and PL/PR if present in BCDS
      for _, bc, bcds in PT.get_children_from_predicates(zone, 'ZoneBC_t/BC_t/BCDataSet_t', ancestors=True):
        if PT.Subset.GridLocation(bcds) == 'Vertex':
          pl_ower = bcds if is_partial(bcds) else bc
          pl_n = PT.get_child_from_name(pl_ower, 'PointList')
          assert (pl_n is None) ^ (PT.Zone.Type(zone) == 'Unstructured'), "Required S zone + PR or U zone + PL"
          distrib_idx_n = MT.find_Distribution(pl_ower, 'Index')

          data = {path : PT.find_node_from_path(bcds, path)[1] for path in PT.predicates_to_paths(bcds, 'BCData_t/DataArray_t')}
          old_pl = pl_n[1] if pl_n is not None else None
          new_distrib_idx, new_pl, new_data = _pl_and_data_vtx_duplication(old_pl, distrib_idx_n[1], n_vtx_2d, data, comm)
          if has_pl(bcds):
            assert pl_n is not None
            PT.set_value(pl_n, new_pl)
            PT.set_value(distrib_idx_n, new_distrib_idx)
          elif has_pr(bcds):
            _extend_pr(PT.find_child_from_name(bcds, 'PointRange'), [1,2])
            PT.set_value(distrib_idx_n, new_distrib_idx)
          for path, value in new_data.items():
            PT.set_value(PT.find_node_from_path(bcds, path), value)
    else:
      # Do not add data in Vertex containers; consequently, we need to:
      # - add a PointList or PointRange in full containers
      # - update PointRange last direction if already existing (for PL, no update is needed)
      # - break link with BC/GC for ZSR, since vertices of BC/GC will be duplicated
      # (Full containers; )
      if PT.Zone.Type(zone) == 'Structured':
        zval = 1 if zone_to_align[zone_path] else 2
        vertex_size = PT.Zone.VertexSize(zone)
      for container in PT.get_children_from_predicate(zone, is_container & is_vertex & ~is_partial):
        if PT.get_label(container) == 'ZoneSubRegion_t': # Break ZSR link
          zsr_extent = PT.Subset.ZSRExtent(container, zone)
          extent_node = PT.find_node_from_path(zone, zsr_extent)
          PT.add_child(container, PT.deep_copy(PT.Subset.getPatch(extent_node)))
          PT.add_child(container, PT.deep_copy(MT.find_Distribution(extent_node)))
          PT.rm_children_from_name(container, '*RegionName')
          if PT.Zone.Type(zone) == 'Structured':
            pr_n = PT.find_child_from_name(container, 'PointRange')
            _extend_pr(pr_n, [zval,zval])
        else: # Add PR/PR in full containers
          ztype = PT.get_np_value(zone).dtype
          if PT.Zone.Type(zone) == 'Unstructured':
            pl = np.arange(distrib_vtx_2d[0]+1, distrib_vtx_2d[1]+1, dtype=ztype).reshape((1,-1), order='F')
            PT.new_IndexArray('PointList', value=pl, parent=container)
          else:
            pr = np.array([[1, vertex_size[0]], [1, vertex_size[1]], [zval, zval]], dtype=ztype, order='F')
            PT.new_IndexRange('PointRange', value=pr, parent=container)
          MT.new_Distribution({'Index': distrib_vtx_2d}, parent=container)
      # Specific treatment of BCDS
      for _, bc, bcds in PT.get_children_from_predicates(zone, 'ZoneBC_t/BC_t/BCDataSet_t', ancestors=True):
        if is_vertex(bcds) and not is_partial(bcds):
          assert is_vertex(bc)
          PT.add_child(bcds, PT.deep_copy(PT.Subset.getPatch(bc)))
          if PT.Zone.Type(zone) == 'Structured' and has_pr(bcds):
            _extend_pr(PT.Subset.getPatch(bcds), [zval,zval])
          PT.add_child(bcds, PT.deep_copy(MT.find_Distribution(bc)))
      # Update PR for structured zones
      for container in PT.get_children_from_predicate(zone, is_container & is_vertex & has_pr):
        assert PT.Zone.Type(zone) == 'Structured'
        _extend_pr(PT.Subset.getPatch(container), [zval,zval])
    
    # Now deal vertex subsets PL/PR, which are extended in all cases
    for subset in PT.get_nodes_from_predicate(zone, is_subset & is_vertex):
      if PT.Zone.Type(zone) == 'Structured' and PT.get_name(subset) not in ['InitialSurface', 'ExtrudedSurface'] :
        pr_n = PT.find_child_from_name(subset, 'PointRange')
        _extend_pr(pr_n, [1,2])
        MT.new_Distribution({'Index' : par_utils.uniform_distribution(PT.PointRange.n_elem(pr_n), comm)}, subset)
        if PT.get_label(subset) == 'GridConnectivity1to1_t':
          donor_path = PT.GridConnectivity.ZoneDonorPath(subset, PT.get_name(base))
          _extend_pr(PT.find_child_from_name(subset, 'PointRangeDonor'), [1,2])
          # Transform depend of align of zone and opp zone : -1 if different alignement
          sign = -1 if zone_to_align[zone_path] ^ zone_to_align[donor_path] else 1
          transform  = PT.find_child_from_name(subset, 'Transform') 
          PT.set_value(transform, np.append(PT.get_np_value(transform), np.array([sign*3], np.int32)))
          # In addition we need to swap one of the two PointRange
          if sign < 0:
            pr_n = PT.find_child_from_name(subset, 'PointRange' + (zone_path > donor_path)*'Donor')
            pr = PT.get_np_value(pr_n)
            pr[2,:] = [2,1]

      elif PT.Zone.Type(zone) == 'Unstructured':
        pl_n = PT.find_child_from_name(subset, 'PointList')
        distrib_idx_n = MT.find_Distribution(subset, 'Index')
        new_distrib_idx, new_pl, _ = _pl_and_data_vtx_duplication(pl_n[1], distrib_idx_n[1], n_vtx_2d, {}, comm)
        # Manage PointListDonor
        pld_n = PT.get_child_from_name(subset, 'PointListDonor')
        if pld_n is not None:
          opp_zone_path = PT.GridConnectivity.ZoneDonorPath(subset, base[0])
          distrib_vtx_2d_opp = zone_to_distrib_vtx[opp_zone_path]
          _, new_pld, _ = _pl_and_data_vtx_duplication(pld_n[1], distrib_idx_n[1], distrib_vtx_2d_opp[2], {}, comm)
          PT.set_value(pld_n, new_pld)
        # Update PointList and Distribution
        PT.set_value(pl_n, new_pl)
        PT.set_value(distrib_idx_n, new_distrib_idx)
  
  # Update base dimension
  for base in PT.get_all_CGNSBase_t(dist_tree):
    PT.set_value(base, [3, 3])
    if ksubset_as=='BC':
      PT.new_Family('InitialSurface',  family_bc='UserDefined', parent=base)
      PT.new_Family('ExtrudedSurface', family_bc='UserDefined', parent=base)
