import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

import maia.sids.sids     as SIDS
import maia.sids.Internal_ext as IE
from maia.utils.parallel import utils as par_utils
from maia.tree_exchange import utils as te_utils
from maia.sids import pytree as PT

def _one_level_filter(nodes, include, exclude):
  if len(include) * len(exclude) != 0:
    raise ValueError("Include and exclude args are mutally exclusive")

  is_array = lambda n : I.getType(n) == 'DataArray_t'
  if not (include or exclude):
    node_and_predicates = [(node, is_array) for node in nodes]
  else:
    queries = include if include else exclude
    splitted = [query.split('/') for query in queries]
    node_and_predicates = []
    for node in nodes:
      data_queries = [data_n for (container_n, data_n) in splitted if PT.match_name(node, container_n)]
      if include:
        # Using local variable names is mandatory to avoid shared reference https://is.gd/JMmMC2
        data_predicate = lambda n, names=data_queries : is_array(n) and any([PT.match_name(n, name) for name in names])
      elif exclude:
        data_predicate = lambda n, names=data_queries : is_array(n) and not any([PT.match_name(n, name) for name in names])
      node_and_predicates.append((node, data_predicate))
  return node_and_predicates

def dist_to_part(partial_distri, dist_data, ln_to_gn_list, comm):
  """
  Helper function calling PDM.BlockToPart
  """
  pdm_distrib = par_utils.partial_to_full_distribution(partial_distri, comm)

  part_data = dict()
  for data_name in dist_data:
    npy_type = dist_data[data_name].dtype
    part_data[data_name] = [np.empty(ln_to_gn.shape[0], dtype=npy_type) for ln_to_gn in ln_to_gn_list]

  BTP = PDM.BlockToPart(pdm_distrib, comm, ln_to_gn_list, len(ln_to_gn_list))
  BTP.BlockToPart_Exchange(dist_data, part_data)

  return part_data

def dist_coords_to_part_coords(dist_zone, part_zones, comm):
  """
  Transfert all the data included in GridCoordinates_t nodes from a distributed
  zone to the partitioned zones
  """
  #Get distribution
  distribution_vtx = te_utils.get_cgns_distribution(dist_zone, 'Vertex')

  #Get data
  dist_data = dict()
  dist_gc = I.getNodeFromType1(dist_zone, "GridCoordinates_t")
  for grid_co in I.getNodesFromType1(dist_gc, 'DataArray_t'):
    dist_data[I.getName(grid_co)] = grid_co[1] #Prevent np->scalar conversion


  vtx_lntogn_list = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
  part_data = dist_to_part(distribution_vtx, dist_data, vtx_lntogn_list, comm)

  for ipart, part_zone in enumerate(part_zones):
    part_gc = I.newGridCoordinates(parent=part_zone)
    for data_name, data in part_data.items():
      #F is mandatory to keep shared reference. Normally no copy is done
      shaped_data = data[ipart].reshape(SIDS.Zone.VertexSize(part_zone), order='F')
      I.newDataArray(data_name, shaped_data, parent=part_gc)



def _dist_to_part_sollike(dist_zone, part_zones, d_sol, data_predicate, comm):
  """
  Shared code for FlowSolution_t and DiscreteData_t
  """
  #Get distribution
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

  #Get data
  dist_data = {I.getName(field) : I.getVal(field) for field in PT.iter_children_from_predicate(d_sol, data_predicate)}

  #Exchange
  part_data = dist_to_part(distribution, dist_data, lntogn_list, comm)

  for ipart, part_zone in enumerate(part_zones):
    #Skip void flow solution (can occur with point lists)
    if lntogn_list[ipart].size > 0:
      if has_pl:
        p_sol = I.getNodeFromName1(part_zone, I.getName(d_sol))
        shape = I.getNodeFromName1(p_sol, 'PointList')[1].shape[1]
      else:
        p_sol = I.createChild(part_zone, I.getName(d_sol), I.getType(d_sol))
        I.newGridLocation(location, parent=p_sol)
        shape = SIDS.Zone.VertexSize(part_zone) if location == 'Vertex' else SIDS.Zone.CellSize(part_zone)
      for data_name, data in part_data.items():
        #F is mandatory to keep shared reference. Normally no copy is done
        shaped_data = data[ipart].reshape(shape, order='F')
        I.newDataArray(data_name, shaped_data, parent=p_sol)

def dist_sol_to_part_sol(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in FlowSolution_t nodes from a distributed
  zone to the partitioned zones
  """
  nodes = PT.get_children_from_label(dist_zone, 'FlowSolution_t')
  for node, predicates in _one_level_filter(nodes, include, exclude):
    _dist_to_part_sollike(dist_zone, part_zones, node, predicates, comm)

def dist_discdata_to_part_discdata(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in DiscreteData_t nodes from a distributed
  zone to the partitioned zones
  """
  nodes = PT.get_children_from_label(dist_zone, 'DiscreteData_t')
  for node, predicates in _one_level_filter(nodes, include, exclude):
    _dist_to_part_sollike(dist_zone, part_zones, node, predicates, comm)

def dist_dataset_to_part_dataset(dist_zone, part_zones, comm):
  """
  Transfert all the data included in BCDataSet_t/BCData_t nodes from a distributed
  zone to the partitioned zones
  """
  for d_zbc in I.getNodesFromType1(dist_zone, "ZoneBC_t"):
    for d_bc in I.getNodesFromType1(d_zbc, "BC_t"):
      bc_path   = I.getName(d_zbc) + '/' + I.getName(d_bc)
      #Get BC distribution and lngn
      distribution_bc = te_utils.get_cgns_distribution(d_bc, 'Index')
      lngn_list_bc    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', bc_path)
      for d_dataset in I.getNodesFromType1(d_bc, 'BCDataSet_t'):
        #If dataset has its own PointList, we must override bc distribution and lngn
        if IE.getDistribution(d_dataset) is not None:
          distribution = te_utils.get_cgns_distribution(d_dataset, 'Index')
          ds_path      = bc_path + '/' + I.getName(d_dataset)
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', ds_path)
        else: #Fallback to bc distribution
          distribution = distribution_bc
          lngn_list    = lngn_list_bc
        #Get data
        dist_data = dict()
        for bc_data, field in IE.iterNodesWithParentsByMatching(d_dataset, 'BCData_t/DataArray_t'):
          dist_data[I.getName(bc_data) + '/' + I.getName(field)] = field[1] #Prevent np->scalar conversion

        #Exchange
        part_data = dist_to_part(distribution, dist_data, lngn_list, comm)

        #Put part data in tree
        for ipart, part_zone in enumerate(part_zones):
          part_bc = I.getNodeFromPath(part_zone, bc_path)
          # Skip void bcs
          if lngn_list[ipart].size > 0:
            # Create dataset if no existing
            part_ds = I.createUniqueChild(part_bc, I.getName(d_dataset), I.getType(d_dataset), I.getValue(d_dataset))
            # Add data
            for data_name, data in part_data.items():
              container_name, field_name = data_name.split('/')
              p_container = I.createUniqueChild(part_ds, container_name, 'BCData_t')
              I.newDataArray(field_name, data[ipart], parent=p_container)


def dist_subregion_to_part_subregion(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in ZoneSubRegion_t nodes from a distributed
  zone to the partitioned zones
  """
  all_zsr = PT.get_children_from_label(dist_zone, 'ZoneSubRegion_t')
  for d_zsr, data_predicate in _one_level_filter(all_zsr, include, exclude):
    # Search matching region
    matching_region_path = IE.getSubregionExtent(d_zsr, dist_zone)
    matching_region = I.getNodeFromPath(dist_zone, matching_region_path)
    assert matching_region is not None

    #Get distribution and lngn
    distribution = te_utils.get_cgns_distribution(matching_region, 'Index')
    lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', matching_region_path)

    #Get Data
    dist_data = {I.getName(field) : I.getVal(field) for field in PT.iter_children_from_predicate(d_zsr, data_predicate)}

    #Exchange
    part_data = dist_to_part(distribution, dist_data, lngn_list, comm)

    #Put part data in tree
    for ipart, part_zone in enumerate(part_zones):
      # Skip void zsr
      if lngn_list[ipart].size > 0:
        # Create ZSR if not existing (eg was defined by bc/gc)
        p_zsr = I.createUniqueChild(part_zone, I.getName(d_zsr), I.getType(d_zsr), I.getValue(d_zsr))
        for field_name, data in part_data.items():
          I.newDataArray(field_name, data[ipart], parent=p_zsr)
