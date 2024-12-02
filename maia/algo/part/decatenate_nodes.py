import maia
import maia.pytree as PT
from   maia.utils   import par_utils
from   maia.factory import dist_from_part
from   maia.algo.part.point_cloud_utils  import create_sub_numbering
from   maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.factory.dist_from_part    import discover_nodes_from_matching

from mpi4py import MPI

import numpy as np

def split_zone_patch(part_zone, family, comm):

  # > Predicates to find family BC
  is_subset_container = lambda n: PT.get_label(n) in ['ZoneBC_t']
  is_subset = lambda n: PT.get_label(n) in ['BC_t'] and\
                        PT.predicate.belongs_to_family(n, family, True)

  zone_bc_n = PT.get_node_from_label(part_zone, "ZoneBC_t")
  
  bc_nodes = PT.get_nodes_from_predicates(part_zone, [is_subset_container, is_subset])

  orig_bc_names = list()
  if len(bc_nodes)>1:
    raise ValueError(f"Family {family} must have a unique related BC (got {len(bc_nodes)}: {[PT.get_name(n) for n in bc_nodes]})")
  if len(bc_nodes)==1:
    bc_n = bc_nodes[0]
    orig_bc_names_n = PT.get_child_from_name (bc_n, "OrdinalToBCNames")
    orig_bc_names = PT.get_value(orig_bc_names_n).split('\n')
    PT.rm_child(zone_bc_n, bc_n)
    bc_loc = PT.Subset.GridLocation(bc_n)
    merge_bc_pl   = PT.Subset.getPatch(bc_n)[1][0]
    merge_bc_gnum = PT.maia.getGlobalNumbering(bc_n, "Index")[1]
    orig_bc_ord = PT.get_node_from_path(bc_n, "BCDataSet/Ordinal/Ordinal")[1]
  else:
    orig_bc_ord = np.empty(0, dtype=np.int32)

  root = comm.allreduce(comm.rank if len(bc_nodes)==1 else -1, op=MPI.MAX)
  orig_bc_names = comm.bcast(orig_bc_names, root=root)
  ord_to_bc_name = {int(v.split(":")[0]):v.split(":")[1] for v in orig_bc_names}
  unique_ordinal = np.array(list(ord_to_bc_name.keys()), dtype=np.int32)
  for i, ordinal in enumerate(unique_ordinal):
    if len(bc_nodes)==1:
      ids     = np.where(orig_bc_ord==ordinal)[0]
      bc_pl   = merge_bc_pl[ids]
      bc_gnum = merge_bc_gnum[ids]
      bc_name = ord_to_bc_name[ordinal]
    else:
      bc_gnum = np.empty(0, dtype=pdm_dtype)
    bc_gnum = create_sub_numbering([bc_gnum], comm)

    if len(bc_nodes)==1:
      bc_n = PT.new_BC(name=bc_name, type="FamilySpecified",
                       point_list=bc_pl.reshape((1,-1), order='F'),
                       loc=bc_loc, parent=zone_bc_n)
      PT.new_node('Ordinal', label='Ordinal_t', value=np.array([ordinal], dtype=np.int32), parent=bc_n)
      PT.maia.newGlobalNumbering({"Index":bc_gnum[0]}, parent=bc_n)
      if family is not None:
        PT.new_FamilyName(family, parent=bc_n)


  # > Get family related ZSR
  is_bcd_zsr = lambda n: PT.get_label(n)=="ZoneSubRegion_t" and\
                         PT.get_name(n).split("#")[0]==family and\
                         PT.get_child_from_name_and_label(n, 'BCDataPath', 'Descriptor_t') is not None# and\
                         # PT.get_value(PT.get_child_from_name_and_label(n, 'BCDataPath', 'Descriptor_t'))==bcd_path
  dist_zone = PT.new_Zone('Zone')
  dist_from_part.discover_nodes_from_matching(dist_zone, [part_zone], [is_bcd_zsr], comm, get_value='leaf', child_list=['Descriptor_t'])

  zsr_paths = PT.predicates_to_paths(dist_zone, [is_bcd_zsr])
  for zsr_path in zsr_paths:
    zsr_n = PT.get_node_from_path(part_zone, zsr_path)
    if zsr_n is not None:
      bcd_path = PT.get_value(PT.get_child_from_name(zsr_n, 'BCDataPath'))
      zsr_pl_n = PT.get_child_from_name(zsr_n, 'PointList')
      ordinal  = PT.get_child_from_name(zsr_n, 'Ordinal')[1]
      zsr_gn_n = PT.maia.getGlobalNumbering(zsr_n, "Index")

      if zsr_pl_n is not None:
        zsr_data_idx_n = PT.get_child_from_name(zsr_n, 'DataStartOffset')
        zsr_data_idx   = PT.get_value(zsr_data_idx_n)
        zsr_data_strd  = np.diff(zsr_data_idx)
        zsr_pl = PT.get_value(zsr_pl_n)[0]
        zsr_gn = PT.get_value(zsr_gn_n)
        zsr_pl = np.repeat(zsr_pl, zsr_data_strd)
        zsr_gn = np.repeat(zsr_gn, zsr_data_strd)

      bcd_name  = PT.utils.path_tail(bcd_path)
      bcds_name = PT.utils.path_head(bcd_path)
      bcds_loc = None if zsr_pl_n is None else PT.Subset.GridLocation(zsr_n)

      # > Get data in zsr
      zsr_data = dict()
      is_data = lambda n: PT.get_label(n)=='DataArray_t' and\
                          PT.get_name(n) not in ['DataStartOffset','Ordinal']
      for data_array_n in PT.get_children_from_predicate(zsr_n, is_data):
        data_name  = PT.get_name (data_array_n)
        data_array = PT.get_value(data_array_n)
        zsr_data[data_name] = data_array

      # > Write BCDS in related BCs
      for i_bc, bc_n in enumerate(PT.get_nodes_from_predicates(part_zone, [is_subset_container, is_subset])):
        bc_ord_n = PT.get_child_from_name(bc_n, 'Ordinal_t')
        bc_ord = PT.get_value(bc_ord_n)[0] if bc_ord_n is not None else i_bc
        zsr_id = np.where(ordinal==bc_ord)[0]
        bcds_pl = zsr_pl[zsr_id] if zsr_pl_n is not None else None
        bcds_gn = zsr_gn[zsr_id] if zsr_pl_n is not None else None
        bcds_data = {name:data[zsr_id] for name,data in zsr_data.items()}

        bcds_n = PT.new_BCDataSet(bcds_name, type='FamilySpecified', 
                                  point_list=bcds_pl.reshape((1,-1), order='F'),
                                  loc=bcds_loc, parent=bc_n)
        PT.new_BCData(bcd_name, fields=bcds_data, parent=bcds_n)
    else:
      bcds_gn = np.empty(0, dtype=pdm_dtype)

    bcds_gn = create_sub_numbering([bcds_gn], comm)
    if zsr_n is not None:
      PT.maia.newGlobalNumbering({'Index':bcds_gn[0]}, parent=bcds_n)
      PT.rm_child(part_zone, zsr_n)


is_concat = lambda n: PT.get_child_from_name(n, ':maia#concatenate') is not None

def decatenate_nodes_from_predicate(part_tree, comm, predicates=['Zone_BC_t', is_concat]):
  """
  Decatenate subset matching predicate using `OriginalBCId` data.

  Args:
    part_tree (CGNSTree) : Partitioned unstructured tree
    comm      (MPIComm)  : MPI communicator
    predicates (list of callable) : Conditions to select node to decatenate. Default to all nodes having a ':maia#concatenate' child.
  """

  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)
  for domain, part_zones in part_tree_per_dom.items():

    
    dist_zone = ['MaskedZone', None, [], 'Zone_t']
    discover_nodes_from_matching(dist_zone, part_zones, predicates, comm,
      child_list=['FamilyName_t', 'GridLocation_t', 'Ordinal_t', 'BCNames', 'BCOrdinal'], get_value='leaf')

    concat_bc_paths = PT.predicates_to_paths(dist_zone, predicates)

    for concat_bc_path in concat_bc_paths:
      list_concat_bc_gn = list()
      decat_zone_bc_nodes = list()
      # bcds_paths = set()
      bcds_paths = list()

      dist_bc_n = PT.get_node_from_path(dist_zone, concat_bc_path)
      orig_bc_names = PT.get_value(PT.get_child_from_name(dist_bc_n, 'BCNames')).split('\n')

      for i_part, part_zone in enumerate(part_zones):
        zone_bc_n = PT.get_node_from_label(part_zone, 'ZoneBC_t')
        concat_bc_n = PT.get_node_from_path(part_zone, concat_bc_path)
        if concat_bc_n is not None:

          # > For now only BCs are managed
          if PT.get_label(concat_bc_n)!='BC_t':
            raise NotImplementedError(f"decatenate_subset_from_predicate only works for BC_t nodes for now (predicate leads to {PT.get_label(concat_bc_n)} node)")

          # > Get concatenated BC node informations
          concat_bc_type = PT.get_value(concat_bc_n)
          concat_bc_loc  = PT.Subset.GridLocation(concat_bc_n)
          concat_bc_fam_n = PT.get_child_from_label(concat_bc_n, 'FamilyName_t')

          concat_bc_pl_n = PT.get_child_from_name(concat_bc_n, 'PointList')
          concat_bc_pl   = PT.get_value(concat_bc_pl_n)[0]

          concat_bc_gn_n = PT.maia.getGlobalNumbering(concat_bc_n, 'Index')
          concat_bc_gn   = PT.get_value(concat_bc_gn_n)
          list_concat_bc_gn.append(concat_bc_gn)

          concat_bc_id_n = PT.get_node_from_path(concat_bc_n, ':maia#concatenate/DirichletData/OriginalBCId')
          concat_bc_id   = PT.get_value(concat_bc_id_n)
          PT.rm_children_from_name(concat_bc_n, ':maia#concatenate')

          orig_bc_ordin_n = PT.get_node_from_name_and_label(concat_bc_n, 'BCOrdinal', 'Descriptor_t')
          if orig_bc_ordin_n is not None:
            orig_bc_ordin = np.array(PT.get_value(orig_bc_ordin_n).split('\n'), dtype=np.int32)

          PT.rm_child(zone_bc_n, concat_bc_n)

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
              PT.maia.newGlobalNumbering({'Index':bc_gn}, parent=bc_n)

              for nodes in PT.iter_children_from_predicates(concat_bc_n, 'BCDataSet_t/BCData_t', ancestors=True):
                bcds_n = nodes[0]
                bcd_n  = nodes[1]

                bcds_type  = PT.get_value(bcds_n)
                bcds_loc_n = PT.get_child_from_label(bcds_n, 'GridLocation_t')
                bcds_loc   = PT.BCDataSet.GridLocation(bcds_n, concat_bc_n) if bcds_loc_n is not None else None

                bcds_path = '/'.join([PT.get_name(zone_bc_n), PT.get_name(bc_n), PT.get_name(bcds_n)])
                if bcds_path not in bcds_paths:
                  bcds_paths.append(bcds_path)


                bcds_pl_n = PT.get_child_from_name(bcds_n, 'PointList')
                if bcds_pl_n is not None:
                  bcds_pl = PT.get_value(bcds_pl_n)[0]

                  bcds_gn_n = PT.maia.getGlobalNumbering(bcds_n, 'Index')
                  bcds_gn   = PT.get_value(bcds_gn_n)

                  bcd_bc_id_n = PT.get_child_from_name_and_label(bcd_n, 'OriginalBCId', 'DataArray_t')
                  bcd_bc_id   = PT.get_value(bcd_bc_id_n)
                  bcds_pl_ids = np.where(bcd_bc_id==bc_id)[0]
                  bcds_pl = bcds_pl[bcds_pl_ids]
                  bcds_gn = bcds_gn[bcds_pl_ids]

                  bc_bcds_n = PT.new_BCDataSet(PT.get_name(bcds_n), type=bcds_type,
                                               point_list=bcds_pl.reshape((1,-1), order='F'),
                                               loc=bcds_loc, parent=bc_n)
                  PT.maia.newGlobalNumbering({'Index':bcds_gn}, parent=bc_bcds_n)
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

      # > Generate gnum for decatenated BCs
      for bc_name in orig_bc_names:
        all_bc_gn = list()

        for part_zone in part_zones:
          bc_n = PT.get_node_from_predicates(zone_bc_n, ['ZoneBC_t', bc_name])
          if bc_n is not None:
            all_bc_gn.append(PT.get_value(PT.maia.getGlobalNumbering(bc_n, 'Index')))
          else:
            all_bc_gn.append(np.empty(0, dtype=pdm_dtype))

        all_bc_gn = create_sub_numbering(all_bc_gn, comm)

        for i_part, part_zone in enumerate(part_zones):
          bc_n = PT.get_node_from_predicates(zone_bc_n, ['ZoneBC_t', bc_name])
          if bc_n is not None:
            bc_gn_n = PT.maia.getGlobalNumbering(bc_n, 'Index')
            PT.set_value(bc_gn_n, all_bc_gn[i_part])

      # > Generate gnum for decatenated BCDSs
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
            all_bcds_gn.append(PT.get_value(PT.maia.getGlobalNumbering(bcds_n, 'Index')))
          else:
            all_bcds_gn.append(np.empty(0, dtype=pdm_dtype))

        all_bcds_gn = create_sub_numbering(all_bcds_gn, comm)

        for i_part, part_zone in enumerate(part_zones):
          bcds_n = PT.get_node_from_path(part_zone, bcds_path)
          if bcds_n is not None:
            bcds_gn_n = PT.maia.getGlobalNumbering(bcds_n, 'Index')
            PT.set_value(bcds_gn_n, all_bcds_gn[i_part])
