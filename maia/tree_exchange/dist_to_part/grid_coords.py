import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

import maia.sids.sids     as SIDS
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils.parallel import utils as par_utils
from maia.utils import py_utils

import maia.tree_exchange.dist_to_part.point_list as PLT

def collect_lntogn_from_path(part_zones, path):
  return [I.getNodeFromPath(part_zone, path)[1] if I.getNodeFromPath(part_zone, path)
      is not None else np.empty(0, pdm_gnum_dtype) for part_zone in part_zones]

def collect_lntogn_from_splited_path(part_zones, prefix, suffix):
  lngn_list = list()
  for p_zone in part_zones:
    extension  = '.'.join(I.getName(p_zone).split('.')[-2:])
    ln_gn_path = '{0}.{1}/{2}'.format(prefix, extension, suffix)
    ln_gn_node = I.getNodeFromPath(p_zone, ln_gn_path)
    if ln_gn_node:
      lngn_list.append(I.getValue(ln_gn_path))
  return lngn_list

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
  distrib_ud = I.getNodeFromName1(dist_zone, ':CGNS#Distribution')
  distribution_vtx = I.getNodeFromName1(distrib_ud, 'Vertex')[1].astype(pdm_gnum_dtype)

  #Get data
  dist_data = dict()
  dist_gc = I.getNodeFromType1(dist_zone, "GridCoordinates_t")
  for grid_co in I.getNodesFromType1(dist_gc, 'DataArray_t'):
    dist_data[I.getName(grid_co)] = I.getValue(grid_co)

  vtx_lntogn_list = collect_lntogn_from_path(part_zones, ':CGNS#GlobalNumbering/Vertex')
  part_data = dist_to_part(distribution_vtx, dist_data, vtx_lntogn_list, comm)
  
  for ipart, part_zone in enumerate(part_zones):
    part_gc = I.newGridCoordinates(parent=part_zone)
    for data_name, data in part_data.items():
      #F is mandatory to keep shared reference. Normally no copy is done
      shaped_data = data[ipart].reshape(SIDS.VertexSize(part_zone), order='F')
      I.newDataArray(data_name, shaped_data, parent=part_gc)

def dist_flowsol_to_part_flowsol(dist_zone, part_zones, comm):
  #Get distribution
  distrib_ud = I.getNodeFromName1(dist_zone, ':CGNS#Distribution')
  for d_flow_sol in  I.getNodesFromType1(dist_zone, "FlowSolution_t"):
    location = SIDS.GridLocation(d_flow_sol)
    if location == 'Vertex':
      distribution = I.getNodeFromName1(distrib_ud, 'Vertex')[1].astype(pdm_gnum_dtype)
      lntogn_list  = collect_lntogn_from_path(part_zones, ':CGNS#GlobalNumbering/Vertex')
    elif location == 'CellCenter':
      distribution = I.getNodeFromName1(distrib_ud, 'Cell')[1].astype(pdm_gnum_dtype)
      lntogn_list  = collect_lntogn_from_path(part_zones, ':CGNS#GlobalNumbering/Cell')
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
      distribution_bc = I.getNodeFromPath(d_bc, ':CGNS#Distribution/Index')[1].astype(pdm_gnum_dtype)
      lngn_list_bc    = collect_lntogn_from_path(part_zones, bc_path + '/:CGNS#GlobalNumbering/Index')
      for d_dataset in I.getNodesFromType1(d_bc, 'BCDataSet_t'):
        #If dataset has its own PointList, we must override bc distribution and lngn
        if I.getNodeFromPath(d_dataset, ':CGNS#Distribution/Index') is not None:
          distribution = I.getNodeFromPath(d_dataset, ':CGNS#Distribution/Index')[1].astype(pdm_gnum_dtype)
          ds_path      = bc_path + '/' + I.getName(d_dataset)
          lngn_list    = collect_lntogn_from_path(part_zones, ds_path + '/:CGNS#GlobalNumbering/Index')
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


