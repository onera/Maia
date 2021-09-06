import Converter.Internal      as I
import maia.sids.Internal_ext  as IE

from maia.sids import sids
from maia import npy_pdm_gnum_dtype as pdm_dtype
from . import distribution_function as DIF

def distribute_pl_node(node, comm):
  """
  Distribute a standard node having a PointList (and its childs) over several processes,
  using uniform distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  dist_node = I.copyTree(node)
  pl = I.getNodeFromName(dist_node, 'PointList')
  assert pl is not None
  n_elem = pl[1].shape[1]
  distri = DIF.uniform_distribution(n_elem, comm).astype(pdm_dtype)

  #Arrays & PLs
  bcds_without_pl = lambda n : I.getType(n) == 'BCDataSet_t' and I.getNodeFromName1(n, 'PointList') is None
  bcds_without_pl_query = [bcds_without_pl, 'BCData_t', 'DataArray_t']
  for array_path in ['IndexArray_t', 'DataArray_t', 'BCData_t/DataArray_t', bcds_without_pl_query]:
    for array_n in IE.getNodesByMatching(dist_node, array_path):
      array_n[1] = array_n[1][0][distri[0]:distri[1]].reshape(1,-1, order='F')

  #Additionnal treatement for subnodes with PL (eg bcdataset)
  has_pl = lambda n : I.getNodeFromName1(n, 'PointList') is not None
  for child in [node for node in I.getChildren(dist_node) if has_pl(node)]:
    dist_child = distribute_pl_node(child, comm)
    child[2] = dist_child[2]

  IE.newDistribution({'Index' : distri}, dist_node)

  return dist_node

def distribute_data_node(node, comm):
  """
  Distribute a standard node having arrays supported by allCells or allVertices over several processes,
  using uniform distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  dist_node = I.copyTree(node)
  assert I.getNodeFromName(dist_node, 'PointList') is None

  for array in I.getNodesFromType1(dist_node, 'DataArray_t'):
    distri = DIF.uniform_distribution(array[1].shape[0], comm)
    array[1] = array[1][distri[0] : distri[1]]

  return dist_node

def distribute_element_node(node, comm):
  """
  Distribute a standard element node over several processes, using uniform distribution.
  Mainly useful for unit tests. Node must be know by each process.
  """
  assert I.getType(node) == 'Elements_t'
  assert sids.ElementCGNSName(node) != "MIXED", "Mixed elements are not supported"
  dist_node = I.copyTree(node)

  n_elem = sids.ElementSize(node)
  distri = DIF.uniform_distribution(n_elem, comm).astype(pdm_dtype)
  IE.newDistribution({'Element' : distri}, dist_node)

  ec = I.getNodeFromName1(dist_node, 'ElementConnectivity')
  if sids.ElementCGNSName(node) in ['NGON_n', 'NFACE_n']:
    eso = I.getNodeFromName1(dist_node, 'ElementStartOffset')
    distri_ec = eso[1][[distri[0], distri[1], -1]]
    ec[1] = ec[1][distri_ec[0] : distri_ec[1]]
    eso[1] = eso[1][distri[0]:distri[1]+1]

    IE.newDistribution({'ElementConnectivity' : distri_ec.astype(pdm_dtype)}, dist_node)
  else:
    n_vtx = sids.ElementNVtx(node)
    ec[1] = ec[1][n_vtx*distri[0] : n_vtx*distri[1]]
    IE.newDistribution({'ElementConnectivity' : n_vtx*distri}, dist_node)
  
  pe = I.getNodeFromName1(dist_node, 'ParentElements')
  if pe is not None:
    pe[1] = pe[1][distri[0] : distri[1]]
  
  return dist_node
