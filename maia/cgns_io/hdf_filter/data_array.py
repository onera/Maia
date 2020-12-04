from .range_to_slab          import compute_slabs
import Converter.Internal as I

def create_data_array_filterS(data_shape, distrib):
  slab_list  = compute_slabs(data_shape, distrib[0:2])
  dn_da    = distrib[1] - distrib[0]
  DSFILEDA = []
  for slab in slab_list:
    iS,iE, jS,jE, kS,kE = [item for bounds in slab for item in bounds]
    # For DSFile, each slab must be of form
    # [[offsetI, offsetJ, offsetK], [1,1,1], [nbI, nbJ, nbK], [1,1,1]]
    # such that we have a list looking like (here i=0...4 pour j=k=0 then i=0...2 for j=k=1)
    # DataSpaceFILE = [[[0,0,0], [1,1,1], [5,1,1], [1,1,1],
                      # [0,1,1], [1,1,1], [3,1,1], [1,1,1]]]
    DSFILEDA.extend([[iS,jS,kS], [1,1,1], [iE-iS, jE-jS, kE-kS], [1,1,1]])
  DSMMRYDA = [[0]    , [1]    , [dn_da], [1]]
  DSFILEDA = list([list(DSFILEDA)])
  DSGLOBDA = [list(data_shape)]
  DSFORMDA = [[0]]
  return DSMMRYDA + DSFILEDA + DSGLOBDA + DSFORMDA

def create_data_array_filterU(distrib):
  dn_da    = distrib[1] - distrib[0]
  DSMMRYDA = [[0         ], [1], [dn_da], [1]]
  DSFILEDA = [[distrib[0]], [1], [dn_da], [1]]
  DSGLOBDA = [[distrib[2]]]
  DSFORMDA = [[0]]
  return DSMMRYDA + DSFILEDA + DSGLOBDA + DSFORMDA

def create_data_array_filter(cgns_node, cgns_path, distrib, hdf_filter, data_shape=None):
  """
  """
  if data_shape is None or len(data_shape) == 1: #Unstructured
    hdf_data_space = create_data_array_filterU(distrib)
  else: #Structured
    hdf_data_space = create_data_array_filterS(data_shape, distrib)

  for data_array in I.getNodesFromType1(cgns_node, 'DataArray_t'):
    path = cgns_path+"/"+data_array[0]
    hdf_filter[path] = hdf_data_space
