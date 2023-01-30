import numpy as np
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import par_utils

def interpret_policy(policy, comm):
  if policy == 'gather':
    policy = 'gather.0'
  policy_split = policy.split('.')
  assert len(policy_split) in [1, 2]

  policy_type = policy_split[0]

  if policy_type == "uniform":
    distribution = par_utils.uniform_distribution
  elif policy_type == "gather":
    assert len(policy_split) == 2
    i_rank = int(policy_split[1])
    assert i_rank < comm.Get_size() 
    distribution = lambda n_elt, comm : par_utils.gathering_distribution(i_rank, n_elt, comm)
  else:
    raise ValueError("Unknown policy for distribution")

  return distribution

def compute_subset_distribution(node, comm, distri_func):
  """
  Compute the distribution for a given node using its PointList or PointRange child
  If a PointRange node is found, the total lenght is getted from the product
  of the differences for each direction (cgns convention (cgns convention :
  first and last are included).
  If a PointList node is found, the total lenght is getted from the product of
  PointList#Size arrays, which store the size of the PL in each direction.
  """

  pr_n = PT.get_child_from_name(node, 'PointRange')
  pl_n = PT.get_child_from_name(node, 'PointList')

  if(pr_n):
    assert pl_n is None
    pr_lenght = PT.PointRange.n_elem(pr_n)
    MT.newDistribution({'Index' : distri_func(pr_lenght, comm)}, parent=node)

  if(pl_n):
    assert pr_n is None
    pls_n   = PT.get_child_from_name(node, 'PointList#Size')
    pl_size = PT.get_value(pls_n)[1]
    MT.newDistribution({'Index' : distri_func(pl_size, comm)}, parent=node)

def compute_connectivity_distribution(node):
  """
  Once ESO is loaded, update element distribution with ElementConnectivity array
  """
  eso_n  = PT.get_child_from_name(node, 'ElementStartOffset')
  if eso_n is None:
    raise RuntimeError
  size_n = PT.get_child_from_name(node, 'ElementConnectivity#Size')

  beg  = eso_n[1][0]
  end  = eso_n[1][-1]
  size = size_n[1][0]

  distri_n = MT.getDistribution(node)
  dtype = PT.get_child_from_name(distri_n, 'Element')[1].dtype
  PT.new_DataArray("ElementConnectivity", value=np.array([beg,end,size], dtype), parent=distri_n)


def compute_elements_distribution(zone, comm, distri_func):
  """
  """
  for elt in PT.iter_children_from_label(zone, 'Elements_t'):
    MT.newDistribution({'Element' : distri_func(PT.Element.Size(elt), comm)}, parent=elt)
    eso_n = PT.get_child_from_name(elt, 'ElementStartOffset')
    if eso_n is not None and eso_n[1] is not None:
      compute_connectivity_distribution(elt)

def compute_zone_distribution(zone, comm, distri_func):
  """
  """
  zone_distri = {'Vertex' : distri_func(PT.Zone.n_vtx(zone), comm),
                 'Cell'   : distri_func(PT.Zone.n_cell(zone), comm)}
  if PT.Zone.Type(zone) == 'Structured':
    zone_distri['Face']  = distri_func(PT.Zone.n_face(zone), comm)

  MT.newDistribution(zone_distri, parent=zone)

  compute_elements_distribution(zone, comm, distri_func)

  predicate_list = [
      [lambda n : PT.get_label(n) in ['ZoneSubRegion_t', 'FlowSolution_t', 'DiscreteData_t']],
      'ZoneBC_t/BC_t',
      'ZoneBC_t/BC_t/BCDataSet_t',
      ['ZoneGridConnectivity_t', lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']]
      ]

  for predicate in predicate_list:
    for node in PT.iter_children_from_predicates(zone, predicate):
      compute_subset_distribution(node, comm, distri_func)


def add_distribution_info(dist_tree, comm, distribution_policy='uniform'):
  """
  """
  distri_func = interpret_policy(distribution_policy, comm)
  for zone in PT.iter_all_Zone_t(dist_tree):
    compute_zone_distribution(zone, comm, distri_func)

def clean_distribution_info(dist_tree):
  """
  Remove the node related to distribution info from the dist_tree
  """
  distri_name = ":CGNS#Distribution"
  is_dist = lambda n : PT.get_label(n) in ['Elements_t', 'ZoneSubRegion_t', 'FlowSolution_t']
  is_gc   = lambda n : PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  for zone in PT.iter_all_Zone_t(dist_tree):
    PT.rm_children_from_name(zone, distri_name)
    for node in PT.iter_nodes_from_predicate(zone, is_dist):
      PT.rm_children_from_name(node, distri_name)
    for bc in PT.iter_nodes_from_predicates(zone, 'ZoneBC_t/BC_t'):
      PT.rm_nodes_from_name(bc, distri_name, depth=2)
    for gc in PT.iter_nodes_from_predicates(zone, ['ZoneGridConnectivity_t', is_gc]):
      PT.rm_children_from_name(gc, distri_name)
