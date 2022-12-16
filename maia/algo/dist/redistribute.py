import os
import mpi4py.MPI as MPI
import numpy      as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT
import maia.transfer.protocols as MTP

from   maia.utils import par_utils, np_utils

vtx_in_elt = {"TETRA_4": 4, 
              "TRI_3"  : 3, 
              "BAR_2"  : 2  
              }

# ---------------------------------------------------------------------------------------
def redistribute_pl_node(node, distribution, comm):
  """
  Redistribute a standard node having a PointList (and its childs) over several processes,
  using a given distribution function. Mainly useful for unit tests. Node must be know by 
  each process.
  """
  node_distrib = PT.get_node_from_path(node, ':CGNS#Distribution/Index')[1]
  n_elt        = node_distrib[2]
  PT.rm_node_from_path(node,  ":CGNS#Distribution")

  # New distribution
  new_distrib = distribution(n_elt, comm)
  MT.newDistribution({'Index' : new_distrib}, node)
  #PL and PLDonor
  for array_n in PT.get_children_from_predicate(node, 'IndexArray_t'):
    array_n[1] = MTP.block_to_block(array_n[1][0], node_distrib, new_distrib, comm).reshape(1,-1, order='F')

  #Data Arrays
  has_subset = lambda n : PT.get_child_from_name(n, 'PointList') is not None or PT.get_child_from_name(n, 'PointRange') is not None
  bcds_without_pl = lambda n : PT.get_label(n) == 'BCDataSet_t' and not has_subset(n)
  bcds_without_pl_query = [bcds_without_pl, 'BCData_t', 'DataArray_t']
  for array_path in ['DataArray_t', 'BCData_t/DataArray_t', bcds_without_pl_query]:
    for array_n in PT.iter_children_from_predicates(node, array_path):
      array_n[1] = MTP.block_to_block(array_n[1], node_distrib, new_distrib, comm)

  #Additionnal treatement for subnodes with PL (eg bcdataset)
  has_pl = lambda n : PT.get_name(n) not in ['PointList', 'PointRange'] and has_subset(n)
  for child in [node for node in PT.get_children(node) if has_pl(node)]:
    dist_child = redistribute_pl_node(child, comm)
    child[2] = dist_child[2]


  return node
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def redistribute_data_node(node, distri, new_distri, comm):
  """
  Distribute a standard node having arrays supported by allCells or allVertices over several processes,
  using given distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  assert PT.get_node_from_name(node, 'PointList') is None

  for array in PT.iter_children_from_label(node, 'DataArray_t'):
    array[1] = MTP.block_to_block(array[1], distri, new_distri, comm)

  return node
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def redistribute_elements_node(node, distribution, comm):

  assert PT.get_label(node) == 'Elements_t'
  assert PT.Element.CGNSName(node) != "MIXED", "Mixed elements are not supported"

  ngon_vision = False
  if PT.Element.CGNSName(node) in ['NGON_n', 'NFACE_n']   :  ngon_vision = True # If NGON or NFACE

  # print("DISTRIBUTION")
  # Get element distribution
  elt_distrib = PT.get_node_from_path(node, ":CGNS#Distribution/Element")[1]
  n_elt       = elt_distrib[2]

  # New element distribution
  new_elt_distrib = distribution(n_elt   , comm)
  new_distrib     = {'Element': new_elt_distrib}

  # print("ESO")
  # > ElementStartOffset
  if ngon_vision :
    eso_n = PT.get_child_from_name(node, 'ElementStartOffset')
    eso   = PT.get_value(eso_n)
    
    # Uniform
    if distribution==par_utils.uniform_distribution:
      i_rank = par_utils.get_gathering_rank(elt_distrib, comm)
      eso = comm.allgather(eso)#, root=i_rank)

      eso = np.concatenate(eso)
      if (eso.shape==new_elt_distrib[2]+1):
        eso_gather = eso[new_elt_distrib[0]:new_elt_distrib[1]+1]
      else :
        eso_gather = eso[new_elt_distrib[0]+1*comm.Get_rank():new_elt_distrib[1]+1+1*comm.Get_rank()]

    # Gathering
    else:
      if comm.Get_rank()==0 : eso = eso
      else                  : eso = eso[1:]
      eso_distrib        = par_utils.gather_and_shift(eso.shape[0], comm)
      new_eso_distrib    = np.full(comm.Get_size()+1, elt_distrib[-1]+1)
      i_rank = par_utils.get_gathering_rank(new_elt_distrib, comm)
      new_eso_distrib[:i_rank+1] = 0
      eso_gather = MTP.block_to_block(eso, eso_distrib, new_eso_distrib, comm)
    PT.set_value(eso_n, eso_gather)

  if ngon_vision :
    ec_distrib     = PT.get_node_from_path(node, ":CGNS#Distribution/ElementConnectivity")[1]

    if (new_elt_distrib[1]-new_elt_distrib[0]!=0):

      new_ec_distrib = np.zeros(3, dtype=np.int32)
      new_ec_distrib[:2] = eso_gather[[ new_elt_distrib[0]-new_elt_distrib[0],
                                        new_elt_distrib[1]-new_elt_distrib[0]]]
      new_ec_distrib[2]  = ec_distrib[2]
      
    else:
      if new_elt_distrib[1]==new_elt_distrib[2]:
        new_ec_distrib = np.full(3, ec_distrib[-1], dtype=np.int32)
      else :
        new_ec_distrib = np.array([0, 0, ec_distrib[-1]], dtype=np.int32)

    new_distrib['ElementConnectivity'] = new_ec_distrib
  else:
    ec_distrib     = elt_distrib*vtx_in_elt[PT.Element.CGNSName(node)]
    new_ec_distrib = new_elt_distrib*vtx_in_elt[PT.Element.CGNSName(node)]

  # > Set CGNS#Distribution node in node
  PT.rm_node_from_path(node,  ":CGNS#Distribution")
  MT.newDistribution(new_distrib, node)

  # > ElementConnectivity
  ec_n    = PT.get_child_from_name(node, 'ElementConnectivity')
  ec      = PT.get_value(ec_n)
  new_ec  = MTP.block_to_block(ec, ec_distrib, new_ec_distrib, comm)
  PT.set_value(ec_n, new_ec)

  # > ParentElement
  pe_n    = PT.get_child_from_name(node, 'ParentElements')
  if pe_n is not None :
    pe      = PT.get_value(pe_n)
    new_pe  = np.zeros((new_elt_distrib[1]-new_elt_distrib[0], pe.shape[1]), dtype=np.int32)
    for ip in range(pe.shape[1]):
      new_pe_tmp  = MTP.block_to_block(pe[:,ip], elt_distrib, new_elt_distrib, comm)
      new_pe[:,ip] = new_pe_tmp
    PT.set_value(pe_n, new_pe)

  return node # Pas top si y'a pas de deep copy ?
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def redistribute_zone(dist_zone, distribution, comm):
  zone = dist_zone # OR deep_copy ?? 

  # Get distribution
  vtx_distrib  = PT.get_node_from_path(zone, ":CGNS#Distribution/Vertex")[1]
  cell_distrib = PT.get_node_from_path(zone, ":CGNS#Distribution/Cell")[1]
  PT.rm_node_from_path(zone,  ":CGNS#Distribution")

  # New distribution
  new_vtx_distrib  = distribution(vtx_distrib[2] , comm)
  new_cell_distrib = distribution(cell_distrib[2], comm)
  new_distrib = {'Vertex' : new_vtx_distrib ,
                 'Cell'   : new_cell_distrib}
  MT.newDistribution(new_distrib, zone)
  

  # > Coords
  grid_coords = PT.get_children_from_label(zone, 'GridCoordinates_t')
  for grid_coord in grid_coords:
    for direction in ["CoordinateX", "CoordinateY", "CoordinateZ"]:
      coord_n = PT.get_node_from_name(grid_coord, direction)
      coord   = PT.get_node_from_name(grid_coord, direction)[1]
      new_coord = MTP.block_to_block(coord, vtx_distrib, new_vtx_distrib, comm)
      PT.set_value(coord_n, new_coord)


  # > Elements
  elts = PT.get_children_from_label(zone, 'Elements_t')
  for elt in elts:
    redistribute_elements_node(elt, distribution, comm)


  # > Flow Solutions
  sols = PT.get_children_from_label(zone, 'FlowSolution_t') + PT.get_children_from_label(zone, 'DiscreteData_t')
  for sol in sols:
    PT.rm_child(zone, sol)

    grid_location = PT.get_value(PT.get_child_from_label(sol, 'GridLocation_t'))
    if   grid_location == 'Vertex'      : distrib = vtx_distrib  ; new_distrib = new_vtx_distrib
    elif grid_location == 'CellCenter'  : distrib = cell_distrib ; new_distrib = new_cell_distrib
    else                                : raise Exception('Redistribute : Only Vertex and CellCenter FlowSolution are supported yet.')
    
    if PT.get_child_from_name(sol, 'PointList') is None:
      PT.add_child(zone, redistribute_data_node(sol, distrib, new_distrib, comm))
    else:
      PT.add_child(zone, redistribute_pl_node(sol, distribution, comm))


  # > BCs
  zonebcs = PT.get_children_from_label(zone, 'ZoneBC_t')
  for zonebc in zonebcs:
    PT.rm_child(zone, zonebc)
    dist_zonebc = PT.new_child(zone, PT.get_name(zonebc), 'ZoneBC_t')
    for bc in PT.iter_children_from_label(zonebc, 'BC_t'):
      PT.add_child(dist_zonebc, redistribute_pl_node(bc, distribution, comm))


  # > GCs
  zonegcs = PT.get_children_from_label(zone, 'ZoneGridConnectivity_t')
  for zonegc in zonegcs:
    PT.rm_child(zone, zonegc)
    dist_zonegc = PT.new_child(zone, PT.get_name(zonegc), 'ZoneGridConnectivity_t')
    for gc in PT.get_children_from_label(zonegc, 'GridConnectivity_t') + PT.get_children_from_label(zonegc, 'GridConnectivity1to1_t'):
      PT.add_child(dist_zonegc, redistribute_pl_node(gc, distribution, comm))


  # > ZoneSubRegion
  zone_subregions = PT.get_children_from_label(zone, 'ZoneSubRegion_t')
  for zone_subregion in zone_subregions:
    # Trick if related to an other node -> add pl
    matching_region_path = PT.getSubregionExtent(zone_subregion, zone)
    if matching_region_path != PT.get_name(zone_subregion):
      PT.add_child(zone_subregion, PT.get_node_from_path(zone, matching_region_path + '/PointList'))
      PT.add_child(zone_subregion, PT.get_node_from_path(zone, matching_region_path + '/PointRange'))
    dist_zone_subregion = redistribute_pl_node(zone_subregion, distribution, comm)
    if matching_region_path != PT.get_name(zone_subregion):
      PT.rm_children_from_name(dist_zone_subregion, 'PointList')
      PT.rm_children_from_name(dist_zone_subregion, 'PointRange')
      PT.rm_child(dist_zone_subregion, MT.getDistribution(dist_zone_subregion))

    PT.rm_child(zone, zone_subregion)
    PT.add_child(zone, dist_zone_subregion)
    
  return zone
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def redistribute_tree(dist_tree, comm, policy='uniform'):
  '''
  Redistribute a distribute tree following a rule :
    - uniform repartition of the data over each process
    - gather all data on one process (the tree structure remains on the other processes)

  Args :
    dist_tree (CGNSTree) : distributed tree that will be redistributed (in place)
    comm      (MPIComm)  : MPI communicator
    policy    (str)      : distribution policy (uniform or gather)

  Return :
    None or the CGNSTree ?? (eg deepcopy or not)

  Notes : Gathering can be controlled by adding the target processor in the policy argument
  (see in the example)

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #redistribute_dist_tree@start
      :end-before: #redistribute_dist_tree@end
      :dedent: 2
  '''
  policy_type = policy.split('.')[0]
  assert policy_type in ["uniform", "gather"]

  if   policy_type == "uniform" :
    distribution = par_utils.uniform_distribution

  elif policy_type == "gather"  :
    assert len(policy.split('.')) in [1, 2]
    if len(policy.split('.'))==2:
      i_rank = int(policy.split('.')[1])
      assert i_rank<comm.Get_size() 
      distribution = lambda n_elt, comm : par_utils.gathering_distribution(i_rank, n_elt, comm)
    else:
      distribution = lambda n_elt, comm : par_utils.gathering_distribution(0     , n_elt, comm)

  tree = dist_tree # OR deep_copy ??

  for zone in PT.iter_all_Zone_t(tree):
    zone = redistribute_zone(zone, distribution, comm)
  

  return tree

# ---------------------------------------------------------------------------------------
# =======================================================================================