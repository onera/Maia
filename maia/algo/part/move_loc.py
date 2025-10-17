import numpy as np
from mpi4py import MPI

import maia
from maia.typing import *
import maia.pytree as PT
import maia.pytree.maia as MT
from maia.utils import np_utils
from maia.factory.dist_from_part import get_parts_per_blocks

from . import multidom_gnum
from . import connectivity_utils
from . import geometry

from .utils import gather_containers_name

import Pypdm.Pypdm as PDM

class CenterToNode:

  CONTAINER_PRED = MT.pred.FULL_CTN_CELL

  def __init__(self, tree: CGNSPartTree, comm: MPIComm,
               idw_power: int = 1, cross_domain: bool = True):

    self.parts    = []
    self.weights  = []
    self.vtx_cell = []
    self.comm     = comm

    parts_per_dom = get_parts_per_blocks(tree, comm)
    vtx_gnum_shifted = multidom_gnum.get_mdom_gnum_vtx(parts_per_dom, comm, cross_domain)

    gnum_list   = []
    for i_dom, zone_path in enumerate(parts_per_dom):
      dist_base = PT.find_child_from_name(tree, PT.utils.path_head(zone_path))
      dim = PT.get_np_value(dist_base)[0]
      for i_part, zone in enumerate(parts_per_dom[zone_path]):

          n_vtx = PT.Zone.n_vtx(zone)
          cell_vtx = connectivity_utils.cell_vtx_connectivity(zone, dim)
          vtx_cell = connectivity_utils.PDM_connectivity_transpose(int(n_vtx), cell_vtx)

          # Compute the distance between vertices and cellcenters
          cx,cy,cz  = PT.Zone.coordinates(zone)
          assert (cx is not None) and (cy is not None) and (cz is not None)
          if PT.Zone.Type(zone)=='Structured' :
            cx = cx.flatten()
            cy = cy.flatten()
            cz = cz.flatten()
          # Use direct api since cell_vtx is already computed
          cell_center = geometry.centers._mean_coords_from_connectivity(cell_vtx, cx,cy,cz)

          # This one is just the local index of each vertices, repeated for
          # each cell the vertex touches. Eg [0, 1, 1, 2,2,2,2] if vtx 0,
          # 1 and 2 belongs to 1, 2 and 4 cells. It it used to
          # compute vtx -> cell center distance for each connected cell
          vtx_idx_rep = np_utils.repeated_arange(vtx_cell.counts)

          diff_x = cx[vtx_idx_rep] - cell_center[0::3][vtx_cell.values-1]
          diff_y = cy[vtx_idx_rep] - cell_center[1::3][vtx_cell.values-1]
          diff_z = cz[vtx_idx_rep] - cell_center[2::3][vtx_cell.values-1]
          norm_rep = (diff_x**2 + diff_y**2 + diff_z**2)**(0.5*idw_power)

          gnum_rep = vtx_gnum_shifted[i_dom][i_part][vtx_idx_rep]

          gnum_list.append(gnum_rep)

          # Store objects needed for exchange
          self.parts.append(zone)
          self.weights.append(1./norm_rep)
          self.vtx_cell.append(vtx_cell)

    self.gmean = PDM.GlobalMean(gnum_list, comm)

  def all_containers(self) -> List[str]:
    return gather_containers_name(self.parts, CenterToNode.CONTAINER_PRED, 'all', self.comm)

  def move_fields(self, container_name: str) -> None:

    #Check that solutions are known on each source partition
    fields_per_part = list()
    for part in self.parts:
      container = PT.find_node_from_path(part, container_name)
      assert PT.Container.GridLocation(container) == 'CellCenter'
      fields_name = sorted([PT.get_name(array) for array in PT.iter_children_from_label(container, 'DataArray_t')])
    fields_per_part.append(fields_name)
    assert fields_per_part.count(fields_per_part[0]) == len(fields_per_part)

    #Collect src sol
    cell_fields = {}
    asflat = lambda val, zone : val.flatten(order='F') if PT.Zone.Type(zone) == 'Structured' else val
    for field_name in fields_per_part[0]:
      field_path = container_name + '/' + field_name
      cell_fields[field_name] = [asflat(PT.find_node_from_path(part, field_path)[1], part)[vtx_cell.values-1].astype(float, copy=False) \
          for part, vtx_cell in zip(self.parts, self.vtx_cell)]

    # Do all reductions
    node_fields = {}
    for field_name, field_values in cell_fields.items():
      node_fields[field_name] = self.gmean.compute_field(field_values, self.weights)

    # Add node fields in tree
    for i_part, part in enumerate(self.parts):
      vtx_shape = PT.Zone.VertexSize(part)
      is_struct = PT.Zone.Type(part) == 'Structured'
      PT.rm_children_from_name(part, f'{container_name}#Vtx')
      fs = PT.new_FlowSolution(f'{container_name}#Vtx', loc='Vertex', parent=part)
      PT.set_label(fs, PT.get_label(PT.find_node_from_path(part, container_name)))
      vtx_cell_idx = self.vtx_cell[i_part].displs
      for field_name, field_values in node_fields.items():
        data_out = field_values[i_part][vtx_cell_idx[:-1]]
        if is_struct:
          data_out = data_out.reshape(vtx_shape, order='F')
        PT.new_DataArray(field_name, data_out, parent=fs)

class NodeToCenter:

  CONTAINER_PRED = MT.pred.FULL_CTN_VTX

  def __init__(self, tree: CGNSPartTree, comm: MPIComm, idw_power: int = 1) -> None:

    self.parts        = []
    self.weights      = []
    self.weightssum   = []
    self.cell_vtx     = []
    self.comm         = comm

    for base in PT.get_all_CGNSBase_t(tree):
      dim = PT.get_np_value(base)[0]
      for p_zone in PT.get_all_Zone_t(base):
        cx,cy,cz = PT.Zone.coordinates(p_zone)
        assert (cx is not None) and (cy is not None) and (cz is not None)
        if PT.Zone.Type(p_zone)=='Structured' :
           cx = cx.flatten()
           cy = cy.flatten()
           cz = cz.flatten()
        cell_vtx = connectivity_utils.cell_vtx_connectivity(p_zone, dim)
        cell_vtx_n = cell_vtx.counts

        # Use direct api since cell_vtx is already computed
        cell_center = geometry.centers._mean_coords_from_connectivity(cell_vtx, cx,cy,cz)

        diff_x = cx[cell_vtx.values-1] - np.repeat(cell_center[0::3], cell_vtx_n)
        diff_y = cy[cell_vtx.values-1] - np.repeat(cell_center[1::3], cell_vtx_n)
        diff_z = cz[cell_vtx.values-1] - np.repeat(cell_center[2::3], cell_vtx_n)
        norm = (diff_x**2 + diff_y**2 + diff_z**2)**(0.5*idw_power)
        weights = 1./norm

        self.parts.append(p_zone)
        self.weights.append(weights)
        self.weightssum.append(np.add.reduceat(weights, cell_vtx.displs[:-1]))
        self.cell_vtx.append(cell_vtx)


  def all_containers(self) -> List[str]:
    return gather_containers_name(self.parts, NodeToCenter.CONTAINER_PRED, 'all', self.comm)

  def move_fields(self, container_name: str) -> None:

    for i_part, part in enumerate(self.parts):
      cell_vtx_idx = self.cell_vtx  [i_part].displs
      cell_vtx     = self.cell_vtx  [i_part].values
      weights      = self.weights   [i_part]
      weightssum   = self.weightssum[i_part]

      container = PT.find_node_from_path(part, container_name)
      container_lbl = PT.get_label(container)
      assert PT.Container.GridLocation(container) == 'Vertex'

      PT.rm_children_from_name(part, f'{container_name}#Cell')
      fs_out = PT.new_FlowSolution(f'{container_name}#Cell', loc='CellCenter', parent=part)
      PT.set_label(fs_out, container_lbl)

      for array in PT.iter_children_from_label(container, 'DataArray_t'):
        data_in = PT.get_np_value(array)
        shape = data_in.shape
        if len(shape) != 1 :
           data_in=data_in.flatten(order='F')
        data_out = np.add.reduceat(data_in[cell_vtx-1] * weights, cell_vtx_idx[:-1])
        data_out /= weightssum
        if len(shape) != 1 :
           data_out=data_out.reshape(np.array(shape)-1, order='F')
        PT.new_DataArray(PT.get_name(array), data_out, parent=fs_out)




def centers_to_nodes(part_tree: CGNSPartTree,
                     comm: MPIComm,
                     containers_name: Union[List[str], Literal['ALL']] = [],
                     **options) -> None:
  """ Create Vertex located fields from CellCenter located fields.

  This transformation is performed for all the fields found under the requested container(s),
  which must be CellCenter located full containers.
  Input tree is modified inplace: Vertex containers are created using
  ``#Vtx`` suffix.

  Interpolation is based on Inverse Distance Weighting
  `(IDW) <https://en.wikipedia.org/wiki/Inverse_distance_weighting>`_ method:
  each cell contributes to each of its vertices with a weight computed from the distance
  between the cell isobarycenter and the vertice. The method can be tuned with
  the following kwargs:

  - ``idw_power`` (float, default = 1) -- Power to which the cell-vertex distance is elevated.

  - ``cross_domain`` (bool, default = True) -- If True, vertices located at domain
    interfaces also receive data from the opposite domain cells. This parameter does not
    apply to internal partitioning interfaces, which are always crossed.

  Args:
    part_tree  (CGNSPartTree): Partionned tree
    comm       (MPIComm): MPI communicator
    containers_name (list of str or ``'ALL'``) : Name of each container node to transfer.
    **options: Options related to interpolation, see above.

  See also:
    A :class:`CenterToNode` object can be instanciated with the same parameters, excluding ``containers_name``,
    and then be used to move containers more than once with its
    ``move_fields(container_name)`` method.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #centers_to_nodes@start
        :end-before: #centers_to_nodes@end
        :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  C2N = CenterToNode(part_tree, comm, **options)

  if containers_name == 'ALL':
    containers_name = C2N.all_containers()
  for container_name in containers_name:
    C2N.move_fields(container_name)

def nodes_to_centers(part_tree: CGNSPartTree,
                     comm: MPIComm,
                     containers_name: Union[List[str], Literal['ALL']] = [],
                     **options) -> None:
  """ Create CellCenter located fields from Vertex located fields.

  This transformation is performed for all the fields found under the requested container(s),
  which must be vertex located full containers.
  Input tree is modified inplace: CellCenter containers are created using
  ``#Cell`` suffix.

  Interpolation is based on Inverse Distance Weighting
  `(IDW) <https://en.wikipedia.org/wiki/Inverse_distance_weighting>`_ method:
  each vertex contributes to the cell value with a weight computed from the distance
  between the cell isobarycenter and the vertice. The method can be tuned with
  the following kwargs:

  - ``idw_power`` (float, default = 1) -- Power to which the cell-vertex distance is elevated.

  Args:
    part_tree  (CGNSPartTree): Partionned tree
    comm       (MPIComm): MPI communicator
    containers_name (list of str or ``'ALL'``) : Name of each container node to transfer.
    **options: Options related to interpolation, see above.

  See also:
    A :class:`NodeToCenter` object can be instanciated with the same parameters, excluding ``containers_name``,
    and then be used to move containers more than once with its
    ``move_fields(container_name)`` method.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #nodes_to_centers@start
        :end-before: #nodes_to_centers@end
        :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  N2C = NodeToCenter(part_tree, comm, **options)

  if containers_name == 'ALL':
    containers_name = N2C.all_containers()
  for container_name in containers_name:
    N2C.move_fields(container_name)
