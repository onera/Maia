import Converter.Internal as I
import numpy              as NPY
import Pypdm.Pypdm        as PDM

def dist_flow_sol_to_part_flow_sol(dist_tree, part_tree, dzone_to_part, comm):
  """
  Transfert all the flowSolution nodes found in dist_tree to the
  corresponding partitions in part_tree.

  Args:
      dist_tree (pyTree)  : A distributed pyTree
      part_tree (pyTree)  : A partitioned pyTree
      dzone_to_part (dict) : Mapping from dist_zones to part_zones : for each
                           dist_zone name (key), list of size partN on this
                           zone for this process (value). The content of the
                           list is unused here, only size matters.
      comm (MPI.comm)    : MPI communicator (from mpi4py)
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  for dist_zone_name in dzone_to_part:

    dist_zone           = I.getNodeFromName2(dist_tree, dist_zone_name)
    flow_solution_nodes = I.getNodesFromType1(dist_zone, "FlowSolution_t")
    distrib_ud          = I.getNodeFromName1(dist_zone, ':CGNS#Distribution')
    n_parts             = len(dzone_to_part[dist_zone_name])

    for flow_solution_node in flow_solution_nodes:

      pln_to_gn = list()
      dist_data = dict()
      part_data = dict()

      gridloc_n = I.getNodeFromType1(flow_solution_node, 'GridLocation_t')
      gridloc   = I.getValue(gridloc_n)

      if(gridloc == 'CellCenter'):
        ppart_name    = 'npCellLNToGN'
        distrib_name  = 'Distribution_cell'
        np_elem_name  = 'nCell'
      elif(gridloc == 'Vertex'):
        ppart_name    = 'npVertexLNToGN'
        distrib_name  = 'Distribution_vtx'
        np_elem_name  = 'nVertex'
      elif(gridloc == 'FaceCenter'):
        ppart_name    = 'npFaceLNToGN'
        distrib_name  = 'Distribution_face'
        np_elem_name  = 'nFace'
      else:
        LOG.error(' '*6 + 'Bad grid location for solution {0} on zone {1}'.format(
          flow_solution_node[0], dist_zone_name))
        comm.abort()

      for data_array in I.getNodesFromType1(flow_solution_node, 'DataArray_t'):
        dist_data[data_array[0]] = DataArray[1]
        part_data[data_array[0]] = list()

      distribution_sol = I.getNodeFromName1(distrib_ud, distrib_name)[1]
      dn_elmt = distribution_sol[1] - distribution_sol[0]
      pdm_distrib = NPY.empty((n_rank + 1), order='C', dtype=distribution_sol.dtype)
      pdm_distrib[0]  = 0
      pdm_distrib[1:] = comm.allgather(dn_elmt)
      for j in xrange(n_rank):
        pdm_distrib[j+1] = pdm_distrib[j+1] + pdm_distrib[j]

      for i_part in xrange(n_parts):

        part_zone_name = "{0}.P{1}.N{2}".format(dist_zone_name, i_rank, i_part)
        part_zone = I.getNodeFromName2(part_tree, part_zone_name)

        ppart_n = I.getNodeFromName1(part_zone, ':CGNS#Ppart')
        ln_to_gn_npy = I.getNodeFromName1(ppart_n, ppart_name)[1]
        pln_to_gn.append(ln_to_gn_npy)

        part_flow_sol = I.newFlowSolution(name=flow_solution_node[0],
                                          gridLocation=gridloc, parent=part_zone)
        np_elem = I.getNodeFromName1(ppart_n, np_elem_name)[1][0]
        for field in dist_data:
          npy_type    = dist_data[field].dtype
          empty_array = NPY.empty(np_elem, order='C', dtype=npy_type)
          I.createChild(part_flow_sol, field, 'DataArray_t', empty_array)
          part_data[field].append(empty_array)

      BTP = PDM.BlockToPart(pdm_distrib, comm, pln_to_gn, n_parts)
      BTP.BlockToPart_Exchange(dist_data, part_data)

