import numpy as np

import maia
import maia.pytree as PT

from maia.factory.dist_from_part import get_parts_per_blocks

from . import multidom_gnum

import Pypdm.Pypdm as PDM

class CenterToNode:

  def __init__(self, tree, comm, idw_power=1, cross_domain=True):

    self.parts        = []
    self.weights      = []
    self.vtx_cell     = []
    self.vtx_cell_idx = []

    parts_per_dom = get_parts_per_blocks(tree, comm)
    vtx_gnum_shifted = multidom_gnum.get_mdom_gnum_vtx(parts_per_dom, comm, cross_domain)

    gnum_list   = []
    for i_dom, parts in enumerate(parts_per_dom.values()):
      for i_part, zone in enumerate(parts):
          ngon_node = PT.Zone.NGonNode(zone)
          nface_node = PT.Zone.NFaceNode(zone)
          n_vtx = PT.Zone.n_vtx(zone)

          nface_eso = PT.get_child_from_name(nface_node, 'ElementStartOffset')[1]
          ngon_eso = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
          nface_ec = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]
          ngon_ec = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
          cell_vtx_idx, cell_vtx = PDM.combine_connectivity(nface_eso, nface_ec, ngon_eso, ngon_ec)

          vtx_cell_idx, vtx_cell = PDM.connectivity_transpose(int(PT.Zone.n_vtx(zone)), cell_vtx_idx, cell_vtx)
          
          cell_center = maia.algo.part.compute_cell_center(zone)

          # Compute the distance between vertices and cellcenters
          cx,cy,cz  = PT.Zone.coordinates(zone)

          # This one is just the local index of each vertices, repeated for
          # each cell the vertex touches. Eg [0, 1, 1, 2,2,2,2] if vtx 0,
          # 1 and 2 belongs to 1, 2 and 4 cells. It it used to
          # compute vtx -> cell center distance for each connected cell
          vtx_idx_rep = np.repeat(np.arange(n_vtx), np.diff(vtx_cell_idx))

          diff_x = cx[vtx_idx_rep] - cell_center[0::3][vtx_cell-1]
          diff_y = cy[vtx_idx_rep] - cell_center[1::3][vtx_cell-1]
          diff_z = cz[vtx_idx_rep] - cell_center[2::3][vtx_cell-1]
          norm_rep = (diff_x**2 + diff_y**2 + diff_z**2)**(0.5*idw_power)
          
          gnum_rep = vtx_gnum_shifted[i_dom][i_part][vtx_idx_rep]

          gnum_list.append(gnum_rep)

          # Store objects needed for exchange
          self.parts.append(zone)
          self.weights.append(1./norm_rep)
          self.vtx_cell.append(vtx_cell)
          self.vtx_cell_idx.append(vtx_cell_idx) # This one will be usefull to go back to unique vtx value

    self.gmean = PDM.GlobalMean(gnum_list, comm)


  def move_fields(self, container_name):

    #Check that solutions are known on each source partition
    fields_per_part = list()
    for part in self.parts:
      container = PT.get_node_from_path(part, container_name)
      assert PT.Subset.GridLocation(container) == 'CellCenter'
      fields_name = sorted([PT.get_name(array) for array in PT.iter_children_from_label(container, 'DataArray_t')])
    fields_per_part.append(fields_name)
    assert fields_per_part.count(fields_per_part[0]) == len(fields_per_part)

    #Collect src sol
    cell_fields = {}
    for field_name in fields_per_part[0]:
      field_path = container_name + '/' + field_name
      cell_fields[field_name] = [PT.get_node_from_path(part, field_path)[1][vtx_cell-1] for \
          part, vtx_cell in zip(self.parts, self.vtx_cell)]

    # Do all reductions
    node_fields = {}
    for field_name, field_values in cell_fields.items():
      node_fields[field_name] = self.gmean.compute_field(field_values, self.weights)

    # Add node fields in tree
    for i_part, part in enumerate(self.parts):
      PT.rm_children_from_name(part, f'{container_name}#Vtx')
      fs = PT.new_FlowSolution(f'{container_name}#Vtx', loc='Vertex', parent=part)
      vtx_cell_idx = self.vtx_cell_idx[i_part]
      for field_name, field_values in node_fields.items():
        PT.new_DataArray(field_name, field_values[i_part][vtx_cell_idx[:-1]], parent=fs)


def centers_to_nodes(tree, comm, containers_name=[], **options):
  """ Create Vertex located FlowSolution_t from CellCenter located FlowSolution_t.

  Interpolation is based on Inverse Distance Weighting 
  `(IDW) <https://en.wikipedia.org/wiki/Inverse_distance_weighting>`_ method:
  each cell contributes to each of its vertices with a weight computed from the distance
  between the cell isobarycenter and the vertice.  The method can be tuned with
  the following kwargs:

  - ``idw_power`` (float, default = 1) -- Power to which the cell-vertex distance is elevated.

  - ``cross_domain`` (bool, default = True) -- If True, vertices located at domain
    interfaces also receive data from the opposite domain cells. This parameter does not
    apply to internal partitioning interfaces, which are always crossed.

  Args:
    tree      (CGNSTree): Partionned tree. Only U-NGon connectivities are managed.
    comm       (MPIComm): MPI communicator
    containers_name (list of str) : List of the names of the FlowSolution_t nodes to transfer.
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
  C2N = CenterToNode(tree, comm, **options)

  for container_name in containers_name:
    C2N.move_fields(container_name)
