from functools import partial
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.io.distribution_tree import compute_connectivity_distribution
from .hdf_dataspace import create_pe_dataspace

def gen_elemts(zone_tree):
  elmts_ini = PT.get_children_from_label(zone_tree, 'Elements_t')
  for elmt in elmts_ini:
    yield elmt

def load_element_connectivity_from_eso(elmt, zone_path, hdf_filter):
  """
  """
  #If needed (reading), update distribution using ESO, which is now loaded
  distrib = MT.getDistribution(elmt)
  if PT.get_child_from_name(distrib, 'ElementConnectivity') is None:
    compute_connectivity_distribution(elmt)

  distrib_ec = MT.getDistribution(elmt, "ElementConnectivity")[1]
  dn_elmt_c  = distrib_ec[1] - distrib_ec[0]
  n_elmt_c   = distrib_ec[2]

  DSMMRYEC = [[0            ], [1], [dn_elmt_c], [1]]
  DSFILEEC = [[distrib_ec[0]], [1], [dn_elmt_c], [1]]
  DSGLOBEC = [[n_elmt_c]]
  DSFORMEC = [[0]]

  ec_path = zone_path+"/"+elmt[0]+"/ElementConnectivity"
  hdf_filter[ec_path] = DSMMRYEC + DSFILEEC + DSGLOBEC + DSFORMEC

def create_zone_eso_elements_filter(elmt, zone_path, hdf_filter, mode):
  """
  """
  distrib_elmt = PT.get_value(MT.getDistribution(elmt, 'Element'))
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

  # > For NGon only
  pe = PT.get_child_from_name(elmt, 'ParentElements')
  if(pe):
    data_space = create_pe_dataspace(distrib_elmt)
    hdf_filter[f"{zone_path}/{PT.get_name(elmt)}/ParentElements"] = data_space
    if PT.get_child_from_name(elmt, 'ParentElementsPosition'):
      hdf_filter[f"{zone_path}/{PT.get_name(elmt)}/ParentElementsPosition"] = data_space

  eso = PT.get_child_from_name(elmt, 'ElementStartOffset')
  eso_path = None
  if(eso):
    # Distribution for NGon/NFace -> ElementStartOffset is the same than DistrbutionFace, except
    # that the last proc have one more element
    n_elmt      = distrib_elmt[2]
    if(mode == 'read'):
      dn_elmt_idx = dn_elmt + 1 # + int(distrib_elmt[1] == n_elmt)
    elif(mode == 'write'):
      dn_elmt_idx = dn_elmt + int((distrib_elmt[1] == n_elmt) and (distrib_elmt[0] != distrib_elmt[1]))
    DSMMRYESO = [[0              ], [1], [dn_elmt_idx], [1]]
    DSFILEESO = [[distrib_elmt[0]], [1], [dn_elmt_idx], [1]]
    DSGLOBESO = [[n_elmt+1]]
    DSFORMESO = [[0]]

    eso_path = zone_path+"/"+elmt[0]+"/ElementStartOffset"
    hdf_filter[eso_path] = DSMMRYESO + DSFILEESO + DSGLOBESO + DSFORMESO

  ec = PT.get_child_from_name(elmt, 'ElementConnectivity')
  if(ec):
    if(eso_path is None):
      raise RuntimeError("In order to load ElementConnectivity, the ElementStartOffset is mandatory")
    ec_path = zone_path+"/"+elmt[0]+"/ElementConnectivity"
    hdf_filter[ec_path] = partial(load_element_connectivity_from_eso, elmt, zone_path)

def create_zone_std_elements_filter(elmt, zone_path, hdf_filter):
  """
  """
  distrib_elmt = PT.get_value(MT.getDistribution(elmt, 'Element'))
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

  elmt_npe = PT.Element.NVtx(elmt)

  DSMMRYElmt = [[0                       ], [1], [dn_elmt*elmt_npe], [1]]
  DSFILEElmt = [[distrib_elmt[0]*elmt_npe], [1], [dn_elmt*elmt_npe], [1]]
  DSGLOBElmt = [[distrib_elmt[2]*elmt_npe]]
  DSFORMElmt = [[0]]

  path = zone_path+"/"+elmt[0]+"/ElementConnectivity"
  hdf_filter[path] = DSMMRYElmt + DSFILEElmt + DSGLOBElmt + DSFORMElmt

  pe = PT.get_child_from_name(elmt, 'ParentElements')
  if(pe):
    data_space = create_pe_dataspace(distrib_elmt)
    hdf_filter[f"{zone_path}/{PT.get_name(elmt)}/ParentElements"] = data_space
    if PT.get_child_from_name(elmt, 'ParentElementsPosition'):
      hdf_filter[f"{zone_path}/{PT.get_name(elmt)}/ParentElementsPosition"] = data_space


def create_zone_elements_filter(zone_tree, zone_path, hdf_filter, mode):
  """
  Prepare the hdf_filter for all the Element_t nodes found in the zone.
  """
  zone_elmts = gen_elemts(zone_tree)
  for elmt in zone_elmts:
    if PT.Element.CGNSName(elmt) in ['NGON_n', 'NFACE_n', 'MIXED']:
      create_zone_eso_elements_filter(elmt, zone_path, hdf_filter, mode)
    else:
      create_zone_std_elements_filter(elmt, zone_path, hdf_filter)

