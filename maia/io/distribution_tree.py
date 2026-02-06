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
    pr_lenght = PT.Subset.n_elem(node)
    MT.new_Distribution({'Index' : distri_func(pr_lenght, comm)}, parent=node)

  if(pl_n):
    assert pr_n is None
    pls_n   = PT.find_child_from_name(node, 'PointList#Size')
    pl_size = PT.get_np_value(pls_n)[1]
    MT.new_Distribution({'Index' : distri_func(pl_size, comm)}, parent=node)

def compute_elements_distribution(zone, comm, distri_func):
  """
  """
  for elt in PT.iter_children_from_label(zone, 'Elements_t'):
    MT.new_Distribution({'Element' : distri_func(PT.Element.Size(elt), comm)}, parent=elt)

def compute_zone_distribution(zone, comm, distri_func):
  """
  """
  zone_distri = {'Vertex' : distri_func(PT.Zone.n_vtx(zone), comm),
                 'Cell'   : distri_func(PT.Zone.n_cell(zone), comm)}
  if PT.Zone.Type(zone) == 'Structured':
    if PT.Zone.IndexDimension(zone) == 3:
      zone_distri['Face']  = distri_func(PT.Zone.n_face(zone), comm)

  MT.new_Distribution(zone_distri, parent=zone)

  compute_elements_distribution(zone, comm, distri_func)

  predicate_list = [
      [PT.pred.label_in(['ZoneSubRegion_t', 'FlowSolution_t', 'DiscreteData_t'])],
      'ZoneBC_t/BC_t',
      'ZoneBC_t/BC_t/BCDataSet_t',
      ['ZoneGridConnectivity_t', PT.pred.IS_GC]
      ]

  for predicate in predicate_list:
    for node in PT.iter_children_from_predicates(zone, predicate):
      compute_subset_distribution(node, comm, distri_func)

  mark_global_bcds_arrays(zone)

def mark_global_bcds_arrays(zone):
  """ Create a Descriptor_t node to indicate the BCDataSet_t/BCData_t/DataArray_t that are
  global (no pointwise value). This descriptor is stored in the DataSet distribution (if existing)
  or in BC distribution otherwise.
  #Size and #Distribution nodes must exist
  """
  for bc in PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t'):
    bc_global_arrays = []
    
    for bcds in PT.get_children_from_label(bc, 'BCDataSet_t'):
      dataset_global_arrays = []
      for bcdata in PT.get_children_from_label(bcds, 'BCData_t'):
        is_global_data = lambda n : PT.get_label(n) == 'DataArray_t' \
                                    and not PT.get_name(n).endswith('#Size') \
                                    and PT.get_child_from_name(bcdata, PT.get_name(n)+'#Size') is None
        for data_array in PT.get_children_from_predicate(bcdata, is_global_data):
          dataset_global_arrays.append(f"{PT.get_name(bcdata)}/{PT.get_name(data_array)}")

      if len(dataset_global_arrays) > 0:
        distri_bcds_n = MT.get_Distribution(bcds)
        if distri_bcds_n is not None: # Register in BCDS/Distribution node
          PT.new_Descriptor('BCDataGlobal', '\n'.join(dataset_global_arrays), parent=distri_bcds_n)
        else: # Save for later registration in BC
          bc_global_arrays.extend([f'{PT.get_name(bcds)}/{path}' for path in dataset_global_arrays])

    if len(bc_global_arrays) > 0: # Register in BC/Distribution node
      PT.new_Descriptor('BCDataGlobal', '\n'.join(bc_global_arrays), parent=MT.get_Distribution(bc))



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
  is_dist = PT.pred.label_in(['Elements_t', 'ZoneSubRegion_t', 'FlowSolution_t'])
  for zone in PT.iter_all_Zone_t(dist_tree):
    PT.rm_children_from_name(zone, distri_name)
    for node in PT.iter_nodes_from_predicate(zone, is_dist):
      PT.rm_children_from_name(node, distri_name)
    for bc in PT.iter_nodes_from_predicates(zone, 'ZoneBC_t/BC_t'):
      PT.rm_nodes_from_name(bc, distri_name, depth=2)
    for gc in PT.iter_nodes_from_predicates(zone, ['ZoneGridConnectivity_t', PT.pred.IS_GC]):
      PT.rm_children_from_name(gc, distri_name)
