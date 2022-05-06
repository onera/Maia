import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

import maia.pytree      as PT
import maia.pytree.maia as MT
from maia.utils.parallel import utils as par_utils
from maia.transfer       import utils as te_utils

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
      shaped_data = data[ipart].reshape(PT.Zone.VertexSize(part_zone), order='F')
      I.newDataArray(data_name, shaped_data, parent=part_gc)



def _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm):
  """
  Shared code for FlowSolution_t and DiscreteData_t
  """
  #Get distribution
  for mask_sol in I.getChildren(mask_tree):
    d_sol = I.getNodeFromName1(dist_zone, I.getName(mask_sol)) #True container
    location = PT.Subset.GridLocation(d_sol)
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
    fields = [I.getName(n) for n in I.getChildren(mask_sol)]
    dist_data = {field : I.getNodeFromName1(d_sol, field)[1] for field in fields}

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
          shape = PT.Zone.VertexSize(part_zone) if location == 'Vertex' else PT.Zone.CellSize(part_zone)
        for data_name, data in part_data.items():
          #F is mandatory to keep shared reference. Normally no copy is done
          shaped_data = data[ipart].reshape(shape, order='F')
          I.newDataArray(data_name, shaped_data, parent=p_sol)

def dist_sol_to_part_sol(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in FlowSolution_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['FlowSolution_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def dist_discdata_to_part_discdata(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in DiscreteData_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['DiscreteData_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def dist_dataset_to_part_dataset(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in BCDataSet_t/BCData_t nodes from a distributed
  zone to the partitioned zones
  """
  for d_zbc in I.getNodesFromType1(dist_zone, "ZoneBC_t"):
    labels = ['BC_t', 'BCDataSet_t', 'BCData_t', 'DataArray_t']
    mask_tree = te_utils.create_mask_tree(d_zbc, labels, include, exclude)
    for mask_bc in I.getChildren(mask_tree):
      bc_path = I.getName(d_zbc) + '/' + I.getName(mask_bc)
      d_bc = I.getNodeFromPath(dist_zone, bc_path) #True BC
      for mask_dataset in I.getChildren(mask_bc):
        ds_path = bc_path + '/' + I.getName(mask_dataset)
        d_dataset = I.getNodeFromPath(dist_zone, ds_path) #True DataSet
        #If dataset has its own PointList, we must override bc distribution and lngn
        if MT.getDistribution(d_dataset) is not None:
          distribution = te_utils.get_cgns_distribution(d_dataset, 'Index')
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', ds_path)
        else: #Fallback to bc distribution
          distribution = te_utils.get_cgns_distribution(d_bc, 'Index')
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', bc_path)
        #Get data
        data_paths = PT.predicates_to_paths(mask_dataset, ['*', '*'])
        dist_data = {data_path : I.getNodeFromPath(d_dataset, data_path)[1] for data_path in data_paths}

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
  mask_tree = te_utils.create_mask_tree(dist_zone, ['ZoneSubRegion_t', 'DataArray_t'], include, exclude)
  for mask_zsr in I.getChildren(mask_tree):
    d_zsr = I.getNodeFromName1(dist_zone, I.getName(mask_zsr)) #True ZSR
    # Search matching region
    matching_region_path = PT.getSubregionExtent(d_zsr, dist_zone)
    matching_region = I.getNodeFromPath(dist_zone, matching_region_path)
    assert matching_region is not None

    #Get distribution and lngn
    distribution = te_utils.get_cgns_distribution(matching_region, 'Index')
    lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', matching_region_path)

    #Get Data
    fields = [I.getName(n) for n in I.getChildren(mask_zsr)]
    dist_data = {field : I.getNodeFromName1(d_zsr, field)[1] for field in fields}

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
