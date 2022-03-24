import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

import maia.sids.sids     as SIDS
import maia.sids.Internal_ext as IE
from   maia.sids import pytree as PT
from maia.utils.parallel import utils    as par_utils
from maia.tree_exchange  import utils    as te_utils
from .discover           import discover_nodes_from_matching
from .                   import index_exchange as IPTB

def _discover_wrapper(dist_zone, part_zones, pl_path, data_path, comm):
  """
  Wrapper for discover_nodes_from_matching which add the node path in distree,
  but also recreate the distributed pointlist if needed
  """
  discover_nodes_from_matching(dist_zone, part_zones, pl_path,   comm, child_list=['GridLocation_t'])
  discover_nodes_from_matching(dist_zone, part_zones, data_path, comm)
  for nodes in IE.iterNodesWithParentsByMatching(dist_zone, pl_path):
    node_path   = '/'.join([I.getName(node) for node in nodes])
    if I.getNodeFromPath(nodes[-1], 'PointList') is None and \
       par_utils.exists_anywhere(part_zones, node_path+'/PointList', comm):
      # > Pointlist must be computed on dist node
      if not par_utils.exists_anywhere(part_zones, node_path+'/:CGNS#GlobalNumbering/Index', comm):
        # > GlobalNumbering is required to do that
        IPTB.create_part_pl_gnum(dist_zone, part_zones, node_path, comm)
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, node_path, comm)

def part_to_dist(partial_distri, part_data, ln_to_gn_list, comm):
  """
  Helper function calling PDM.PartToBlock
  """
  pdm_distrib = par_utils.partial_to_full_distribution(partial_distri, comm)

  PTB = PDM.PartToBlock(comm, ln_to_gn_list, pWeight=None, partN=len(ln_to_gn_list),
                        t_distrib=0, t_post=1, userDistribution=pdm_distrib)

  dist_data = dict()
  PTB.PartToBlock_Exchange(dist_data, part_data)
  return dist_data

def part_coords_to_dist_coords(dist_zone, part_zones, comm):

  distribution = te_utils.get_cgns_distribution(dist_zone, 'Vertex')
  lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')

  d_grid_co = I.getNodeFromType1(dist_zone, "GridCoordinates_t")
  part_data = dict()
  for coord in I.getNodesFromType1(d_grid_co, 'DataArray_t'):
    part_data[I.getName(coord)] = list()

  for part_zone in part_zones:
    p_grid_co = I.getNodesFromName1(part_zone, I.getName(d_grid_co))
    for coord in I.getNodesFromType1(p_grid_co, 'DataArray_t'):
      flat_data = coord[1].ravel(order='A') #Reshape structured arrays for PDM exchange
      part_data[I.getName(coord)].append(flat_data)

  # Exchange
  dist_data = part_to_dist(distribution, part_data, lntogn_list, comm)
  for coord, array in dist_data.items():
    dist_coord = I.getNodeFromName1(d_grid_co, coord)
    I.setValue(dist_coord, array)

def _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm):
  """
  Shared code for FlowSolution_t and DiscreteData_t
  """
  for mask_sol in I.getChildren(mask_tree):
    d_sol = I.getNodeFromName1(dist_zone, I.getName(mask_sol)) #True container

    if not par_utils.exists_everywhere(part_zones, I.getName(d_sol), comm):
      return #Skip FS that remains on dist_tree but are not present on part tree

    location = SIDS.GridLocation(d_sol)
    has_pl   = I.getNodeFromName1(d_sol, 'PointList') is not None

    if has_pl:
      distribution = te_utils.get_cgns_distribution(d_sol, 'Index')
      lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Index', I.getName(d_sol))
    else:
      assert location in ['Vertex', 'CellCenter']
      if location == 'Vertex':
        distribution = te_utils.get_cgns_distribution(dist_zone, 'Vertex')
        lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
      elif location == 'CellCenter':
        distribution = te_utils.get_cgns_distribution(dist_zone, 'Cell')
        lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Cell')

    #Discover data
    _exists_everywhere = lambda name : par_utils.exists_everywhere(part_zones, f'{I.getName(d_sol)}/{name}', comm)
    fields = [I.getName(n) for n in I.getChildren(mask_sol) if _exists_everywhere(I.getName(n))]
    part_data = {field : [] for field in fields}

    for part_zone in part_zones:
      p_sol = I.getNodesFromName1(part_zone, I.getName(d_sol))
      for field in fields:
        flat_data = I.getNodeFromName(p_sol, field)[1].ravel(order='A') #Reshape structured arrays for PDM exchange
        part_data[field].append(flat_data)

    # Exchange
    dist_data = part_to_dist(distribution, part_data, lntogn_list, comm)
    for field, array in dist_data.items():
      dist_field = I.getNodeFromName1(d_sol, field)
      I.setValue(dist_field, array)

def part_sol_to_dist_sol(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in FlowSolution_t nodes from partitioned
  zones to the distributed zone. Data created on (one or more) partitions and not present in dist_tree
  is also reported to the distributed zone.
  """
  # Complete distree with partitioned fields and exchange PL if needed
  _discover_wrapper(dist_zone, part_zones, 'FlowSolution_t', 'FlowSolution_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['FlowSolution_t', 'DataArray_t'], include, exclude)
  _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm)

def part_discdata_to_dist_discdata(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in DiscreteData_t from partitioned
  zones to the distributed zone. Data created on (one or more) partitions and not present in dist_tree
  is also reported to the distributed zone.
  """
  # Complete distree with partitioned fields and exchange PL if needed
  _discover_wrapper(dist_zone, part_zones, 'DiscreteData_t', 'DiscreteData_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['DiscreteData_t', 'DataArray_t'], include, exclude)
  _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm)

def part_subregion_to_dist_subregion(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in ZoneSubRegion_t nodes from the partitioned
  zones to the distributed zone.
  Zone subregions must exist on distzone
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['ZoneSubRegion_t', 'DataArray_t'], include, exclude)
  for mask_zsr in I.getChildren(mask_tree):
    d_zsr = I.getNodeFromName1(dist_zone, I.getName(mask_zsr)) #True ZSR
    # Search matching region
    matching_region_path = IE.getSubregionExtent(d_zsr, dist_zone)
    matching_region = I.getNodeFromPath(dist_zone, matching_region_path)
    assert matching_region is not None

    #Get distribution and lngn
    distribution = te_utils.get_cgns_distribution(matching_region, 'Index')
    lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', matching_region_path)

    #Discover data
    fields = [I.getName(n) for n in I.getChildren(mask_zsr)]
    part_data = {field : [] for field in fields}

    for part_zone in part_zones:
      p_zsr = I.getNodeFromPath(part_zone, I.getName(d_zsr))
      if p_zsr is not None:
        for field in fields:
          part_data[field].append(I.getNodeFromName(p_zsr, field)[1])

    #Partitions having no data must be removed from lngn list since they have no contribution
    empty_parts_ids = [ipart for ipart, part_zone in enumerate(part_zones)\
        if I.getNodeFromPath(part_zone, I.getName(d_zsr)) is None]
    for ipart in empty_parts_ids[::-1]:
      lngn_list.pop(ipart)

    # Exchange
    dist_data = part_to_dist(distribution, part_data, lngn_list, comm)
    for field, array in dist_data.items():
      dist_field = I.getNodeFromName1(d_zsr, field)
      I.setValue(dist_field, array)

def part_dataset_to_dist_dataset(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in BCDataSet_t/BCData_t nodes from partitioned
  zones to the distributed zone.
  """

  # Complete distree with partitioned fields and exchange PL if needed
  bc_ds_path = 'ZoneBC_t/BC_t/BCDataSet_t'
  _discover_wrapper(dist_zone, part_zones, bc_ds_path, bc_ds_path+'/BCData_t/DataArray_t', comm)

  for d_zbc in I.getNodesFromType1(dist_zone, "ZoneBC_t"):
    labels = ['BC_t', 'BCDataSet_t', 'BCData_t', 'DataArray_t']
    mask_tree = te_utils.create_mask_tree(d_zbc, labels, include, exclude)
    for mask_bc in I.getChildren(mask_tree):
      bc_path   = I.getName(d_zbc) + '/' + I.getName(mask_bc)
      d_bc = I.getNodeFromPath(dist_zone, bc_path) #True BC
      for mask_dataset in I.getChildren(mask_bc):
        ds_path = bc_path + '/' + I.getName(mask_dataset)
        d_dataset = I.getNodeFromPath(dist_zone, ds_path) #True DataSet
        #If dataset has its own PointList, we must override bc distribution and lngn
        if IE.getDistribution(d_dataset) is not None:
          distribution = te_utils.get_cgns_distribution(d_dataset, 'Index')
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', ds_path)
        else: #Fallback to bc distribution
          distribution = te_utils.get_cgns_distribution(d_bc, 'Index')
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', bc_path)

        #Discover data
        data_paths = PT.predicates_to_paths(mask_dataset, ['*', '*'])
        part_data = {path : [] for path in data_paths}

        for part_zone in part_zones:
          p_dataset = I.getNodeFromPath(part_zone, ds_path)
          if p_dataset is not None:
            for path in data_paths:
              part_data[path].append(I.getNodeFromPath(p_dataset, path)[1])

        #Partitions having no data must be removed from lngn list since they have no contribution
        empty_parts_ids = [ipart for ipart, part_zone in enumerate(part_zones)\
            if I.getNodeFromPath(part_zone, ds_path) is None]
        for ipart in empty_parts_ids[::-1]:
          lngn_list.pop(ipart)

        #Exchange
        dist_data = part_to_dist(distribution, part_data, lngn_list, comm)
        for field, array in dist_data.items():
          dist_field = I.getNodeFromPath(d_dataset, field)
          I.setValue(dist_field, array)

