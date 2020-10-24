import Converter.Internal as I
from maia.utils import zone_elements_utils as EZU

def gen_elemts(zone_tree):
  elmts_ini = I.getNodesFromType1(zone_tree, 'Elements_t')
  for elmt in elmts_ini:
    yield elmt


def create_zone_ngon_elements_filter(elmt, zone_path, hdf_filter):
  """
  """
  distrib_ud   = I.getNodeFromName1(elmt      , ':CGNS#Distribution')
  distrib_elmt = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

  pe = I.getNodeFromName1(elmt, 'ParentElements')
  if(pe):
    DSMMRYPE = [[0              , 0], [1, 1], [dn_elmt, 2], [1, 1]]
    DSFILEPE = [[distrib_elmt[0], 0], [1, 1], [dn_elmt, 2], [1, 1]]
    DSGLOBPE = [[distrib_elmt[2], 0]]
    DSFORMPE = [[1]]

    path = zone_path+"/"+elmt[0]+"/ParentElements"
    hdf_filter[path] = DSMMRYPE + DSFILEPE + DSGLOBPE + DSFORMPE



def create_zone_nfac_elements_filter(elmt, zone_path, hdf_filter):
  """
  """
  distrib_ud   = I.getNodeFromName1(elmt      , ':CGNS#Distribution')
  distrib_elmt = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]



def create_zone_mixed_elements_filter(elmt, zone_path, hdf_filter):
  """
  """
  distrib_ud   = I.getNodeFromName1(elmt      , ':CGNS#Distribution')
  distrib_elmt = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
  dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

  raise NotImplemented("Mixed elements are not allowed ")


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

def create_zone_elements_filter(zone_tree, zone_path, hdf_filter):
  """
  """
  zone_elmts = gen_elemts(zone_tree)
  print(zone_elmts)
  for elmt in zone_elmts:
    if(elmt[1][0] == 22):
      create_zone_ngon_elements_filter(elmt, zone_path, hdf_filter)
    elif(elmt[1][0] == 23):
      create_zone_nfac_elements_filter(elmt, zone_path, hdf_filter)
    elif(elmt[1][0] == 20):
      create_zone_mixed_elements_filter(elmt, zone_path, hdf_filter)
    else:
      create_zone_std_elements_filter(elmt, zone_path, hdf_filter)

  # create_zone_std_elements_filter(zone_tree, zone_path, hdf_filter)
  # create_zone_ngon_elements_filter(zone_tree, zone_path, hdf_filter)
