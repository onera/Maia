import numpy as np

import Pypdm.Pypdm as PDM
import Converter.Internal as I

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo import indexing
from maia.utils import par_utils


def pe_to_nface(zone, comm, remove_PE=False):
  """Create a NFace node from a NGon node with ParentElements.

  NGon range is supposed to start at 1.
  Input tree is modified inplace.

  Args:
    zone       (CGNSTree): Distributed zone
    comm       (MPIComm) : MPI communicator
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.
  """
  ngon_node = PT.Zone.NGonNode(zone)
  nface_distri = MT.getDistribution(zone, 'Cell')[1]
  ngon_distri  = MT.getDistribution(ngon_node, 'Element')[1]
  face_distri = par_utils.partial_to_full_distribution(ngon_distri, comm)
  cell_distri = par_utils.partial_to_full_distribution(nface_distri, comm)
  assert PT.Element.Range(ngon_node)[0] == 1
  local_pe = indexing.get_ngon_pe_local(ngon_node).reshape(-1, order='C')

  cell_face_idx, cell_face = PDM.dfacecell_to_dcellface(comm, face_distri, cell_distri, local_pe)
  cell_face_range  = np.array([1, PT.Zone.n_cell(zone)], zone[1].dtype) + PT.Zone.n_face(zone)
  nface_ec_distr_f = par_utils.gather_and_shift(cell_face_idx[-1], comm)
  nface_ec_distri  = par_utils.full_to_partial_distribution(nface_ec_distr_f, comm)
  eso = cell_face_idx+nface_ec_distri[0]

  nface = I.newElements('NFaceElements', 'NFACE',  parent=zone)
  I.newPointRange("ElementRange", cell_face_range, parent=nface)
  I.newDataArray("ElementStartOffset", eso,        parent=nface)
  I.newDataArray("ElementConnectivity", cell_face, parent=nface)
  MT.newDistribution({"Element" : nface_distri, "ElementConnectivity" : nface_ec_distri}, nface)
  
  if remove_PE:
    I._rmNodesByName(ngon_node, "ParentElements")


