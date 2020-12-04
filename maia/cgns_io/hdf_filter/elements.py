from functools import partial
import numpy as NPY

import Converter.Internal as I
from maia.utils import zone_elements_utils as EZU

def gen_elemts(zone_tree):
  elmts_ini = I.getNodesFromType1(zone_tree, 'Elements_t')
  for elmt in elmts_ini:
    yield elmt

def load_element_connectivity_from_eso(elmt, zone_path, hdf_filter):
  """
  """
  distrib_ud   = I.getNodeFromName1(elmt      , ':CGNS#Distribution')
  distrib_elmt = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

  eso_n = I.getNodeFromName1(elmt, 'ElementStartOffset') # Maintenant il est chargé
  if(eso_n[1] is None):
    raise RuntimeError
  eso = eso_n[1]

  beg_face_vtx = eso[0]
  end_face_vtx = eso[eso.shape[0]-1]
  dn_face_vtx  = end_face_vtx - beg_face_vtx

  # print("beg_face_vtx::", beg_face_vtx)
  # print("end_face_vtx::", end_face_vtx)
  distrib_n  = None
  ec_size_n  = I.getNodeFromName1(elmt, 'ElementConnectivity#Size')
  if(ec_size_n is not None):
    n_face_vtx = NPY.prod(ec_size_n[1])
  else:
    distrib_ud = I.getNodeFromName1(elmt      , ":CGNS#Distribution")
    distrib_n  = I.getNodeFromName1(distrib_ud, "DistributionElementConnectivity")
    assert(distrib_n is not None)
    n_face_vtx = distrib_n[1][2]

  # print("n_face_vtx::", n_face_vtx)

  n_face      = distrib_elmt[2]
  dn_face_idx = dn_elmt + int(distrib_elmt[1] == n_face)
  DSMMRYEC = [[0           ], [1], [dn_face_vtx], [1]]
  DSFILEEC = [[beg_face_vtx], [1], [dn_face_vtx], [1]]
  DSGLOBEC = [[n_face_vtx ]]
  DSFORMEC = [[0]]

  ec_path = zone_path+"/"+elmt[0]+"/ElementConnectivity"
  hdf_filter[ec_path] = DSMMRYEC + DSFILEEC + DSGLOBEC + DSFORMEC

  if(distrib_n is None):
    distrib = NPY.empty(3, dtype=eso.dtype)
    distrib[0] = beg_face_vtx
    distrib[1] = end_face_vtx
    distrib[2] = n_face_vtx
    I.newDataArray("DistributionElementConnectivity", value=distrib, parent=distrib_ud)



def create_zone_eso_elements_filter(elmt, zone_path, hdf_filter, mode):
  """
  """
  distrib_ud   = I.getNodeFromName1(elmt      , ':CGNS#Distribution')
  distrib_elmt = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

  # > For NGon only
  pe = I.getNodeFromName1(elmt, 'ParentElements')
  if(pe):
    DSMMRYPE = [[0              , 0], [1, 1], [dn_elmt, 2], [1, 1]]
    DSFILEPE = [[distrib_elmt[0], 0], [1, 1], [dn_elmt, 2], [1, 1]]
    DSGLOBPE = [[distrib_elmt[2], 2]]
    DSFORMPE = [[1]]

    pe_path = zone_path+"/"+elmt[0]+"/ParentElements"
    hdf_filter[pe_path] = DSMMRYPE + DSFILEPE + DSGLOBPE + DSFORMPE

  eso = I.getNodeFromName1(elmt, 'ElementStartOffset')
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

  ec = I.getNodeFromName1(elmt, 'ElementConnectivity')
  if(ec):
    if(eso_path is None):
      raise RuntimeError("In order to load ElementConnectivity, the ElementStartOffset is mandatory")
    ec_path = zone_path+"/"+elmt[0]+"/ElementConnectivity"
    hdf_filter[ec_path] = partial(load_element_connectivity_from_eso, elmt, zone_path)


def create_zone_mixed_elements_filter(elmt, zone_path, hdf_filter):
  """
  """
  distrib_ud   = I.getNodeFromName1(elmt      , ':CGNS#Distribution')
  distrib_elmt = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

  raise NotImplementedError("Mixed elements are not allowed ")


def create_zone_std_elements_filter(elmt, zone_path, hdf_filter):
  """
  """
  distrib_ud   = I.getNodeFromName1(elmt      , ':CGNS#Distribution')
  distrib_elmt = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

  elmt_npe = EZU.get_npe_with_element_type_cgns(elmt[1][0])

  DSMMRYElmt = [[0                       ], [1], [dn_elmt*elmt_npe], [1]]
  DSFILEElmt = [[distrib_elmt[0]*elmt_npe], [1], [dn_elmt*elmt_npe], [1]]
  DSGLOBElmt = [[distrib_elmt[2]*elmt_npe]]
  DSFORMElmt = [[0]]

  path = zone_path+"/"+elmt[0]+"/ElementConnectivity"
  hdf_filter[path] = DSMMRYElmt + DSFILEElmt + DSGLOBElmt + DSFORMElmt

def create_zone_elements_filter(zone_tree, zone_path, hdf_filter, mode):
  """
  """
  zone_elmts = gen_elemts(zone_tree)
  for elmt in zone_elmts:
    if(elmt[1][0] == 22) or (elmt[1][0] == 23):
      create_zone_eso_elements_filter(elmt, zone_path, hdf_filter, mode)
    elif(elmt[1][0] == 20):
      create_zone_mixed_elements_filter(elmt, zone_path, hdf_filter)
    else:
      create_zone_std_elements_filter(elmt, zone_path, hdf_filter)

