from typing import List, Tuple
import numpy as np
from mpi4py import MPI
import logging as LOG

import Converter.Internal as I

import maia.sids.cgns_keywords  as CGK
import maia.sids.sids           as SIDS
import maia.sids.Internal_ext   as IE

from maia.sids.cgns_keywords import Label as CGL

from cmaia.geometry.geometry import *

# ------------------------------------------------------------------------
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

fmt = f'%(levelname)s[{mpi_rank}/{mpi_size}]:%(message)s '
LOG.basicConfig(filename = f"maia_workflow_log.{mpi_rank}.log",
                level    = 10,
                format   = fmt,
                filemode = 'w')

# ------------------------------------------------------------------------
@IE.check_is_label("Zone_t")
def get_center_cell(zone_node: List) -> Tuple:
  """
  Args:
      zone_node (List): CGNS Zone_t node

  Returns:
      Tuple: Returns center cell coordinates and local to global numerotation for cells

  Raises:
      NotImplementedError: Description
      SIDS.NotImplementedForElementError: Description
  """
  n_cell = SIDS.Zone.n_cell(zone_node)
  LOG.info(f"n_cell = {n_cell}")

  # Get coordinates
  cx, cy, cz = SIDS.coordinates(zone_node)
  LOG.info(f"cx = {cx}")

  pdm_nodes = IE.getChildFromName1(zone_node, ":CGNS#Ppart")
  vtx_coords    = I.getVal(IE.getChildFromName1(pdm_nodes, "np_vtx_coord"))
  vtx_ln_to_gn  = I.getVal(IE.getGlobalNumbering(zone_node, 'Vertex'))
  cell_ln_to_gn = I.getVal(IE.getGlobalNumbering(zone_node, 'Cell'))
  LOG.info(f"vtx_coords = {vtx_coords}")

  if SIDS.Zone.Type(zone_node) == "Unstructured":
    element_node = getChildFromLabel1(zone_node, CGL.Elements_t.name)
    # NGon elements
    if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
      face_vtx, face_vtx_idx, ngon_pe = SIDS.face_connectivity(zone_node)
      center_cell = compute_center_cell_u(n_cell,
                                          cx, cy, cz,
                                          face_vtx,
                                          face_vtx_idx,
                                          ngon_pe)
    else:
      raise SIDS.NotImplementedForElementError(zone_node, element_node)
  else:
    raise NotImplementedError(f"cell center computation is only available for Unstructured Zone.")

  return center_cell, cell_ln_to_gn
