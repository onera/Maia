import maia.pytree      as PT
import maia.pytree.maia as MT
from   maia.algo.part.point_cloud_utils import create_sub_numbering
from   maia.factory import dist_from_part
from   maia import npy_pdm_gnum_dtype as pdm_dtype

import numpy as np


def deconcatenate_subset_from_family(part_zones, family, comm):
  # > Predicates to find family BCs over all procs
  is_bc_from_fam = lambda n: PT.get_label(n)=='BC_t' and PT.predicate.belongs_to_family(n, family)
  predicates = ['ZoneBC_t', is_bc_from_fam]
  dist_zone = ['MaskedZone', None, [], 'Zone_t']
  dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, predicates, comm,
    child_list=['FamilyName_t', 'GridLocation_t', 'Ordinal_t', 'BCNames', 'BCOrdinal'], get_value='leaf')

  concat_bc_paths = PT.predicates_to_paths(dist_zone, predicates)
  if len(concat_bc_paths)>1:
    raise ValueError(f"Family {family} leads to multiple BCs.")
  concat_bc_path = concat_bc_paths[0]

  bcds_paths = list()

  dist_bc_n = PT.get_node_from_path(dist_zone, concat_bc_path)
  orig_bc_names = PT.get_value(PT.get_child_from_name(dist_bc_n, 'BCNames')).split('\n')

  for i_part, part_zone in enumerate(part_zones):
    zone_bc_n = PT.get_node_from_label(part_zone, 'ZoneBC_t')
    concat_bc_n = PT.get_node_from_path(part_zone, concat_bc_path)
    if concat_bc_n is not None:

      # > For now only BCs are managed
      if PT.get_label(concat_bc_n)!='BC_t':
        raise NotImplementedError(f"deconcatenation service only works for BC_t nodes for now (predicate leads to {PT.get_label(concat_bc_n)} node)")

      # > Get concatenated BC node informations
      concat_bc_type = PT.get_value(concat_bc_n)
      concat_bc_loc  = PT.Subset.GridLocation(concat_bc_n)
      concat_bc_fam_n = PT.get_child_from_label(concat_bc_n, 'FamilyName_t')

      concat_bc_pl_n = PT.get_child_from_name(concat_bc_n, 'PointList')
      concat_bc_pl   = PT.get_value(concat_bc_pl_n)[0]

      PT.rm_child(zone_bc_n, concat_bc_n)
      if concat_bc_pl.size==0:
        # BC may be empty but not its BCDataSet. In this case,
        # we assume that BCDataSet values will be retrieved with other procs
        continue

      concat_bc_gn = MT.globalnumbering_value(concat_bc_n, 'Index')

      concat_bc_id_n = PT.get_node_from_path(concat_bc_n, ':maia#concatenate/DirichletData/OriginalBCId')
      concat_bc_id   = PT.get_value(concat_bc_id_n)
      PT.rm_children_from_name(concat_bc_n, ':maia#concatenate')

      orig_bc_ordin_n = PT.get_node_from_name_and_label(concat_bc_n, 'BCOrdinal', 'Descriptor_t')
      if orig_bc_ordin_n is not None:
        orig_bc_ordin = np.array(PT.get_value(orig_bc_ordin_n).split('\n'), dtype=np.int32)

      for bc_id, bc_name in enumerate(orig_bc_names):

        bc_pl_ids = np.where(concat_bc_id==bc_id)[0]
        if bc_pl_ids.size>0:
          bc_pl = concat_bc_pl[bc_pl_ids]
          bc_gn = concat_bc_gn[bc_pl_ids]

          bc_n = PT.new_BC(bc_name, concat_bc_type,
                           point_list=bc_pl.reshape((1,-1), order='F'),
                           loc=concat_bc_loc,
                           parent=zone_bc_n)

          if concat_bc_fam_n is not None:
            PT.new_FamilyName(PT.get_value(concat_bc_fam_n), parent=bc_n)
          if orig_bc_ordin_n is not None:
            PT.new_node('Ordinal', 'Ordinal_t', orig_bc_ordin[bc_id], parent=bc_n)
          MT.new_GlobalNumbering({'Index':bc_gn}, parent=bc_n)

          for nodes in PT.iter_children_from_predicates(concat_bc_n, 'BCDataSet_t/BCData_t', ancestors=True):
            bcds_n = nodes[0]
            bcd_n  = nodes[1]

            bcds_type  = PT.get_value(bcds_n)
            bcds_loc_n = PT.get_child_from_label(bcds_n, 'GridLocation_t')
            bcds_loc   = PT.BCDataSet.GridLocation(bcds_n, concat_bc_n) if bcds_loc_n is not None else None
            bcds_pl    = PT.get_node_from_name(bcds_n, 'PointList')

            bcds_path = '/'.join([PT.get_name(zone_bc_n), PT.get_name(bc_n), PT.get_name(bcds_n)])
            if bcds_path not in bcds_paths and bcds_pl is not None:
              bcds_paths.append(bcds_path)

            bcds_pl_n = PT.get_child_from_name(bcds_n, 'PointList')
            if bcds_pl_n is not None:
              bcds_pl = PT.get_value(bcds_pl_n)[0]

              bcds_gn_n = MT.find_GlobalNumbering(bcds_n, 'Index')
              bcds_gn   = PT.get_value(bcds_gn_n)

              bcd_bc_id_n = PT.get_child_from_name_and_label(bcd_n, 'OriginalBCId', 'DataArray_t')
              bcd_bc_id   = PT.get_value(bcd_bc_id_n)
              bcds_pl_ids = np.where(bcd_bc_id==bc_id)[0]
              bcds_pl = bcds_pl[bcds_pl_ids]
              bcds_gn = bcds_gn[bcds_pl_ids]

              bc_bcds_n = PT.new_BCDataSet(PT.get_name(bcds_n), type=bcds_type,
                                           point_list=bcds_pl.reshape((1,-1), order='F'),
                                           loc=bcds_loc, parent=bc_n)
              MT.new_GlobalNumbering({'Index':bcds_gn}, parent=bc_bcds_n)
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

  # > Generate gnum for deconcatenated BCs
  for bc_name in orig_bc_names:
    all_bc_gn = list()

    for part_zone in part_zones:
      bc_n = PT.get_child_from_predicates(part_zone, ['ZoneBC_t', bc_name])
      if bc_n is not None:
        all_bc_gn.append(MT.globalnumbering_value(bc_n, 'Index'))
      else:
        all_bc_gn.append(np.empty(0, dtype=pdm_dtype))

    all_bc_gn = create_sub_numbering(all_bc_gn, comm)

    for i_part, part_zone in enumerate(part_zones):
      bc_n = PT.get_child_from_predicates(part_zone, ['ZoneBC_t', bc_name])
      if bc_n is not None:
        bc_gn_n = MT.find_GlobalNumbering(bc_n, 'Index')
        PT.set_value(bc_gn_n, all_bc_gn[i_part])

  # > Generate gnum for deconcatenated BCDSs
  gather_bcds_paths = comm.allgather(bcds_paths)
  bcds_paths = list()
  for rank in range(comm.size):
    for bcds_path in gather_bcds_paths[rank]:
      if bcds_path not in bcds_paths:
        bcds_paths.append(bcds_path)

  for bcds_path in bcds_paths:
    all_bcds_gn = list()

    for part_zone in part_zones:
      bcds_n = PT.get_node_from_path(part_zone, bcds_path)
      if bcds_n is not None:
        all_bcds_gn.append(MT.globalnumbering_value(bcds_n, 'Index'))
      else:
        all_bcds_gn.append(np.empty(0, dtype=pdm_dtype))

    all_bcds_gn = create_sub_numbering(all_bcds_gn, comm)

    for i_part, part_zone in enumerate(part_zones):
      bcds_n = PT.get_node_from_path(part_zone, bcds_path)
      if bcds_n is not None:
        bcds_gn_n = MT.find_GlobalNumbering(bcds_n, 'Index')
        PT.set_value(bcds_gn_n, all_bcds_gn[i_part])


def deconcatenate_subsets_from_families(part_tree, comm, families='*'):
  """
  Deconcatenate BC from each family using `OriginalBCId` data.

  Warning:
    Each family from ``families`` argument must lead to unique BC.

  Args:
    part_tree (CGNSPartTree)          : Partitioned unstructured tree
    comm      (MPIComm)               : MPI communicator
    families  (str or list, optional) : Family names. Default to ``"*"``. 
  """

  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)
  for domain, part_zones in part_tree_per_dom.items():

    # > If all families, we need to discover them first
    if families=='*':

      predicates = ['ZoneBC_t', 'BC_t']
      dist_zone = ['MaskedZone', None, [], 'Zone_t']
      dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, predicates, comm,
        child_list=['FamilyName_t'], get_value='leaf')

      families = list()
      for n in PT.get_nodes_from_label(dist_zone, 'FamilyName_t'):
        if PT.get_value(n) not in families:
          families.append(PT.get_value(n))

    # > Deconcatenate BCs and associated BCDataSets from families
    for family in families:
      deconcatenate_subset_from_family(part_zones, family, comm)
