from   mpi4py import MPI
import numpy          as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import py_utils, np_utils
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.transfer.dist_to_part.index_exchange import collect_distributed_pl

from Pypdm.Pypdm import DistributedMeshNodal

def _split_point_list_by_dim(pl_list, range_by_dim, comm):
  """
  Split a list of PointList nodes into 4 sublists depending on the dimension
  of each PointList.
  Dimension is recovered using the values of the PointList and the range_by_dim
  array (GridLocation may be a better choice ?)
  """
  def _get_dim(pl):
    min_l_pl = np.amin(pl[0,:], initial=np.iinfo(pl.dtype).max)
    max_l_pl = np.amax(pl[0,:], initial=-1)

    min_pl    = comm.allreduce(min_l_pl, op=MPI.MIN)
    max_pl    = comm.allreduce(max_l_pl, op=MPI.MAX)
    for i_dim in range(len(range_by_dim)):
      if(min_pl >= range_by_dim[i_dim][0] and max_pl <= range_by_dim[i_dim][1]):
        return i_dim

  return py_utils.bucket_split(pl_list, lambda pl: _get_dim(pl), size=4)


def cgns_dist_zone_to_pdm_dmesh_nodal(dist_zone, comm, needs_vertex=True, needs_bc=True):
  """
  Create a pdm_dmesh_nodal structure from a distributed zone
  """
  distrib_vtx = I.getVal(MT.getDistribution(dist_zone, 'Vertex'))
  n_vtx   = distrib_vtx[2]
  dn_vtx  = distrib_vtx[1] - distrib_vtx[0]

  n_elt_per_dim  = [0,0,0]
  sorted_elts_by_dim = PT.Zone.get_ordered_elements_per_dim(dist_zone)
  for elt_dim in sorted_elts_by_dim:
    for elt in elt_dim:
      assert PT.Element.CGNSName(elt) not in ["NGON_n", "NFACE_n"]
      n_elt_per_dim[PT.Element.Dimension(elt)-1] += PT.Element.Size(elt)

  #Create DMeshNodal
  dmesh_nodal = DistributedMeshNodal(comm, n_vtx, *n_elt_per_dim[::-1])

  #Vertices
  if needs_vertex:
    if dn_vtx > 0:
      gridc_n    = I.getNodeFromName1(dist_zone, 'GridCoordinates')
      cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
      cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
      cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
      dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])
    else:
      dvtx_coord = np.empty(0, dtype='float64', order='F')
    dmesh_nodal.set_coordinnates(dvtx_coord)

    # keep dvtx_coord object alive for ParaDiGM
    multi_part_node = I.createUniqueChild(dist_zone, ':CGNS#MultiPart', 'UserDefinedData_t')
    I.newDataArray('dvtx_coord', dvtx_coord, parent=multi_part_node)

  #Elements
  to_elmt_size = lambda e : I.getVal(MT.getDistribution(e, 'Element'))[1] - I.getVal(MT.getDistribution(e, 'Element'))[0]

  for i_dim, elts in enumerate(sorted_elts_by_dim):
    elt_pdm_types = np.array([MT.pdm_elts.element_pdm_type(PT.Element.Type(e)) for e in elts], dtype=np.int32)
    elt_lengths   = np.array([to_elmt_size(e) for e in elts], dtype=np.int32)
    elmts_connectivities = [I.getNodeFromName1(e, "ElementConnectivity")[1] for e in elts]
    dmesh_nodal.set_sections(MT.pdm_elts.elements_dim_to_pdm_kind[i_dim], elmts_connectivities, elt_pdm_types, elt_lengths)

  # Boundaries
  if needs_bc:
    range_by_dim = PT.Zone.get_elt_range_per_dim(dist_zone)
    # print("range_by_dim --> ", range_by_dim)

    bc_point_lists = collect_distributed_pl(dist_zone, [['ZoneBC_t', 'BC_t']])
    # Find out in which dim the boundary refers
    bc_point_lists_by_dim = _split_point_list_by_dim(bc_point_lists, range_by_dim, comm)

    for i_dim, bc_pl in enumerate(bc_point_lists_by_dim):
      if(len(bc_pl) > 0 ):
        delmt_bound_idx, delmt_bound = np_utils.concatenate_point_list(bc_pl, pdm_gnum_dtype)
        # Shift because CGNS global numbering is for all elements / ParaDiGM is by dimension
        delmt_bound -= (range_by_dim[i_dim][0] - 1)
        n_elmt_group = delmt_bound_idx.shape[0] - 1
        #Need an holder to prevent memory deletion
        pdm_node = I.createUniqueChild(dist_zone, ':CGNS#DMeshNodal#Bnd{0}'.format(i_dim), 'UserDefinedData_t')
        I.newDataArray('delmt_bound_idx', delmt_bound_idx, parent=pdm_node)
        I.newDataArray('delmt_bound'    , delmt_bound    , parent=pdm_node)

        dmesh_nodal.set_group_elmt(MT.pdm_elts.elements_dim_to_pdm_kind[i_dim], n_elmt_group, delmt_bound_idx, delmt_bound)

  return dmesh_nodal

