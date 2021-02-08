import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

import maia.sids.sids     as SIDS
from maia.utils.parallel import utils as par_utils
from maia.utils import py_utils
from maia.tree_exchange import utils as te_utils

def dist_to_part(partial_distri, dist_data, ln_to_gn_list, comm):
  pdm_distrib = par_utils.partial_to_full_distribution(partial_distri, comm)

  part_data = dict()
  for data_name in dist_data:
    npy_type = dist_data[data_name].dtype
    part_data[data_name] = [np.empty(ln_to_gn.shape[0], dtype=npy_type) for ln_to_gn in ln_to_gn_list]

  BTP = PDM.BlockToPart(pdm_distrib, comm, ln_to_gn_list, len(ln_to_gn_list))
  BTP.BlockToPart_Exchange(dist_data, part_data)

  return part_data

def dist_coords_to_part_coords(dist_zone, part_zones, comm):

  #Get distribution
  distribution_vtx = te_utils.get_cgns_distribution(dist_zone, ':CGNS#Distribution/Vertex')

  #Get data
  dist_data = dict()
  dist_gc = I.getNodeFromType1(dist_zone, "GridCoordinates_t")
  for grid_co in I.getNodesFromType1(dist_gc, 'DataArray_t'):
    dist_data[I.getName(grid_co)] = I.getValue(grid_co)

  vtx_lntogn_list = te_utils.collect_cgns_g_numbering(part_zones, ':CGNS#GlobalNumbering/Vertex')
  part_data = dist_to_part(distribution_vtx, dist_data, vtx_lntogn_list, comm)
  
  for ipart, part_zone in enumerate(part_zones):
    part_gc = I.newGridCoordinates(parent=part_zone)
    for data_name, data in part_data.items():
      #F is mandatory to keep shared reference. Normally no copy is done
      shaped_data = data[ipart].reshape(SIDS.VertexSize(part_zone), order='F')
      I.newDataArray(data_name, shaped_data, parent=part_gc)

def dist_flowsol_to_part_flowsol(dist_zone, part_zones, comm):
  #Get distribution
  for d_flow_sol in  I.getNodesFromType1(dist_zone, "FlowSolution_t"):
    location = SIDS.GridLocation(d_flow_sol)
    if location == 'Vertex':
      distribution = te_utils.get_cgns_distribution(dist_zone, ':CGNS#Distribution/Vertex')
      lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, ':CGNS#GlobalNumbering/Vertex')
    elif location == 'CellCenter':
      distribution = te_utils.get_cgns_distribution(dist_zone, ':CGNS#Distribution/Cell')
      lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, ':CGNS#GlobalNumbering/Cell')
    else:
      raise NotImplementedError("Only cell or vertex flow solutions are supported")

    #Get data
    dist_data = dict()
    for field in I.getNodesFromType1(d_flow_sol, 'DataArray_t'):
      dist_data[I.getName(field)] = I.getValue(field)

    #Exchange
    part_data = dist_to_part(distribution, dist_data, lntogn_list, comm)

    for ipart, part_zone in enumerate(part_zones):
      p_flow_sol = I.newFlowSolution(I.getName(d_flow_sol), location, parent=part_zone)
      shape = SIDS.VertexSize(part_zone) if location == 'Vertex' else SIDS.CellSize(part_zone)
      for data_name, data in part_data.items():
        #F is mandatory to keep shared reference. Normally no copy is done
        shaped_data = data[ipart].reshape(shape, order='F')
        I.newDataArray(data_name, shaped_data, parent=p_flow_sol)


def dist_dataset_to_part_dataset(dist_zone, part_zones, comm):
  for d_zbc in I.getNodesFromType1(dist_zone, "ZoneBC_t"):
    for d_bc in I.getNodesFromType1(d_zbc, "BC_t"):
      bc_path   = I.getName(d_zbc) + '/' + I.getName(d_bc)
      #Get BC distribution and lngn
      distribution_bc = te_utils.get_cgns_distribution(d_bc, ':CGNS#Distribution/Index')
      lngn_list_bc    = te_utils.collect_cgns_g_numbering(part_zones, bc_path + '/:CGNS#GlobalNumbering/Index')
      for d_dataset in I.getNodesFromType1(d_bc, 'BCDataSet_t'):
        #If dataset has its own PointList, we must override bc distribution and lngn
        if I.getNodeFromPath(d_dataset, ':CGNS#Distribution/Index') is not None:
          distribution = te_utils.get_cgns_distribution(d_dataset, ':CGNS#Distribution/Index')
          ds_path      = bc_path + '/' + I.getName(d_dataset)
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, ds_path + '/:CGNS#GlobalNumbering/Index')
        else: #Fallback to bc distribution
          distribution = distribution_bc
          lngn_list    = lngn_list_bc
        #Get data
        dist_data = dict()
        for bc_data, field in py_utils.getNodesWithParentsFromTypePath(d_dataset, 'BCData_t/DataArray_t'):
          dist_data[I.getName(bc_data) + '/' + I.getName(field)] = I.getValue(field)

        #Exchange
        part_data = dist_to_part(distribution, dist_data, lngn_list, comm)

        #Put part data in tree
        for ipart, part_zone in enumerate(part_zones):
          part_bc = I.getNodeFromPath(part_zone, bc_path)
          # Skip void bcs
          if lngn_list[ipart].size > 0:
            # Create dataset if no existing
            part_ds = I.createUniqueChild(part_bc, I.getName(d_dataset), I.getType(d_dataset), I.getValue(d_dataset))
            part_ds_pl = I.getNodeFromName1(part_ds, 'PointList')
            # Get shape
            shape = I.getValue(part_ds_pl).shape if part_ds_pl else I.getNodeFromName1(part_bc, 'PointList')[1].shape
            # Add data
            for data_name, data in part_data.items():
              container_name, field_name = data_name.split('/')
              p_container = I.createUniqueChild(part_ds, container_name, 'BCData_t')
              I.newDataArray(field_name, data[ipart].reshape(shape, order='F'), parent=p_container)


