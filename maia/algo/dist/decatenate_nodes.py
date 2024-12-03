import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import par_utils
from maia.algo.apply_function_to_nodes import zones_iterator

import numpy as np


is_concat = lambda n: PT.get_child_from_name(n, ':maia#concatenate') is not None

def decatenate_subset_from_predicate(dist_tree, comm, predicate=is_concat):
  """
  Decatenate subset matching predicate using `OriginalBCId` data.

  Args:
    dist_tree (CGNSTree) : Distributed unstructured tree, starting at Zone_t level or higher.
    comm      (MPIComm)  : MPI communicator
    predicate (callable) : Conditions to select node to decatenate. Default to all nodes having a ':maia#concatenate' child. 

  Example:
    .. literalinclude:: snippets/test_algo.py
      :language: python
      :start-after: #decatenate_from_name@start
      :end-before:  #decatenate_from_name@end
      :dedent: 2

  """
  for dist_zone in zones_iterator(dist_tree):

    assert PT.Zone.Type(dist_zone)=="Unstructured"

    # > Merge bc nodes from a same family
    zone_bc_n = PT.get_child_from_label(dist_zone, "ZoneBC_t")
    for concat_bc_n in PT.get_nodes_from_predicate(dist_zone, predicate):

      # > For now only BCs are managed
      if PT.get_label(concat_bc_n)!='BC_t':
        raise NotImplementedError(f"decatenate_subset_from_predicate only works for BC_t nodes for now (predicate leads to {PT.get_label(concat_bc_n)} node)")

      # > Get concatenated BC node informations
      concat_bc_type = PT.get_value(concat_bc_n)
      concat_bc_loc  = PT.Subset.GridLocation(concat_bc_n)
      concat_bc_fam_n = PT.get_child_from_label(concat_bc_n, 'FamilyName_t')

      concat_bc_pl_n = PT.get_child_from_name(concat_bc_n, 'PointList')
      concat_bc_pl   = PT.get_value(concat_bc_pl_n)[0]

      concat_bc_id_n = PT.get_node_from_path(concat_bc_n, ':maia#concatenate/DirichletData/OriginalBCId')
      concat_bc_id   = PT.get_value(concat_bc_id_n)
      PT.rm_children_from_name(concat_bc_n, ':maia#concatenate')

      orig_bc_names = PT.get_value(PT.get_node_from_name_and_label(concat_bc_n, 'BCNames', 'Descriptor_t')).split('\n')
      orig_bc_ordin_n = PT.get_node_from_name_and_label(concat_bc_n, 'BCOrdinal', 'Descriptor_t')
      if orig_bc_ordin_n is not None:
        orig_bc_ordin = np.array(PT.get_value(orig_bc_ordin_n).split('\n'), dtype=np.int32)

      PT.rm_child(zone_bc_n, concat_bc_n)

      for bc_id, bc_name in enumerate(orig_bc_names):

        bc_pl_ids = np.where(concat_bc_id==bc_id)[0]
        bc_pl = concat_bc_pl[bc_pl_ids]
        bc_distrib = par_utils.dn_to_distribution(bc_pl.size, comm)

        bc_n = PT.new_BC(bc_name, concat_bc_type,
                         point_list=bc_pl.reshape((1,-1), order='F'),
                         loc=concat_bc_loc,
                         parent=zone_bc_n)
        if concat_bc_fam_n is not None:
          PT.new_FamilyName(PT.get_value(concat_bc_fam_n), parent=bc_n)
        PT.maia.newDistribution({'Index':bc_distrib}, parent=bc_n)
        if orig_bc_ordin_n is not None:
          PT.new_node('Ordinal', 'Ordinal_t', orig_bc_ordin[bc_id], parent=bc_n)

        # > Decatenate related BCDataSet children
        for nodes in PT.iter_children_from_predicates(concat_bc_n, 'BCDataSet_t/BCData_t', ancestors=True):
          bcds_n = nodes[0]
          bcd_n  = nodes[1]

          bcds_type  = PT.get_value(bcds_n)
          bcds_loc_n = PT.get_child_from_label(bcds_n, 'GridLocation_t')
          bcds_loc   = PT.BCDataSet.GridLocation(bcds_n, bc_n) if bcds_loc_n is not None else None

          bcds_pl_n = PT.get_child_from_name(bcds_n, 'PointList')
          if bcds_pl_n is not None:
            bcds_pl = PT.get_value(bcds_pl_n)[0]
            bcd_bc_id_n = PT.get_child_from_name_and_label(bcd_n, 'OriginalBCId', 'DataArray_t')
            bcd_bc_id   = PT.get_value(bcd_bc_id_n)
            bcds_pl_ids = np.where(bcd_bc_id==bc_id)[0]
            bcds_pl = bcds_pl[bcds_pl_ids]
            bcds_distrib = par_utils.dn_to_distribution(bcds_pl.size, comm)

            bc_bcds_n = PT.new_BCDataSet(PT.get_name(bcds_n), type=bcds_type,
                                         point_list=bcds_pl.reshape((1,-1), order='F'),
                                         loc=bcds_loc, parent=bc_n)
            PT.maia.newDistribution({'Index':bcds_distrib}, parent=bc_bcds_n)
            fields = {PT.get_name(data_array_n):PT.get_value(data_array_n)[bcds_pl_ids]
              for data_array_n in PT.get_children_from_label(bcd_n, 'DataArray_t')}
            bc_bcd_n = PT.new_BCData(PT.get_name(bcd_n), fields=fields, parent=bc_bcds_n)
            PT.rm_children_from_name(bc_bcd_n, 'OriginalBCId')
          else:
            bcds_pl_ids = bc_pl_ids
            bc_bcds_n = PT.new_BCDataSet(PT.get_name(bcds_n), type=bcds_type,
                                         loc=bcds_loc, parent=bc_n)
            fields = {PT.get_name(data_array_n):PT.get_value(data_array_n)[bcds_pl_ids]
              for data_array_n in PT.get_children_from_label(bcd_n, 'DataArray_t')}
            bc_bcd_n = PT.new_BCData(PT.get_name(bcd_n), fields=fields, parent=bc_bcds_n)
            PT.rm_children_from_name(bc_bcd_n, 'OriginalBCId')
