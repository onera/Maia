import Converter.Internal as I
from maia.utils import zone_elements_utils as EZU


def create_zone_std_elements_filter(zone_tree,
                                    zone_path,
                                    hdf_filter):
  """
  """
  print("create_zone_std_elements_filter")

  elmts_ini = I.getNodesFromType1(zone_tree, 'Elements_t')
  for elmt in elmts_ini:

    if(elmt[1][0] == 22 or elmt[1][0] == 23 or elmt[1][0] == 20): # Skip NGon/NFac/Mixed
      continue

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

