from maia.utils.numbering.range_to_slab import compute_slabs

def create_combined_dataspace(data_shape, distrib):
  """
  Create a dataspace from a flat distribution, but for arrays having a 3d (resp. 2d) stucture
  ie (Nx, Ny, Nz) (resp. (Nx, Ny)) numpy arrays.
  First, the 1d distribution is converted into slabs to load with the function compute_slabs.
  Those slabs are then combined to create the dataspace :
   for DSFile, we are expecting a list including all the slabs looking like
   [[startI_1, startJ_1, startK_1], [1,1,1], [nbI_1, nbJ_1, nbK_1], [1,1,1],
    [startI_2, startJ_2, startK_2], [1,1,1], [nbI_2, nbJ_2, nbK_2], [1,1,1], ...
    [startI_N, startJ_N, startK_N], [1,1,1], [nbI_N, nbJ_N, nbK_N], [1,1,1]]
   DSGlob me be the list of the tree dimensions sizes
   DSMmry and DSFrom have the same structure than flat / 1d dataspaces

  Mostly usefull for structured blocks.
  """
  slab_list  = compute_slabs(data_shape, distrib[0:2])
  dn_da    = distrib[1] - distrib[0]
  DSFILEDA = []
  for slab in slab_list:
    iS,iE, jS,jE, kS,kE = [item for bounds in slab for item in bounds]
    DSFILEDA.extend([[iS,jS,kS], [1,1,1], [iE-iS, jE-jS, kE-kS], [1,1,1]])
  DSMMRYDA = [[0]    , [1]    , [dn_da], [1]]
  DSFILEDA = list([list(DSFILEDA)])
  DSGLOBDA = [list(data_shape)]
  DSFORMDA = [[0]]
  return DSMMRYDA + DSFILEDA + DSGLOBDA + DSFORMDA

def create_flat_dataspace(distrib):
  """
  Create the most basic dataspace (1d / flat) for a given
  distribution.
  """
  dn_da    = distrib[1] - distrib[0]
  DSMMRYDA = [[0         ], [1], [dn_da], [1]]
  DSFILEDA = [[distrib[0]], [1], [dn_da], [1]]
  DSGLOBDA = [[distrib[2]]]
  DSFORMDA = [[0]]
  return DSMMRYDA + DSFILEDA + DSGLOBDA + DSFORMDA

def create_pe_dataspace(distrib):
  """
  Create a dataspace from a flat distribution, of elements,
  but adapted to "ParentElements" arrays ie (N,2) numpy arrays.
  """
  dn_pe    = distrib[1] - distrib[0]
  DSMMRYPE = [[0              , 0], [1, 1], [dn_pe, 2], [1, 1]]
  DSFILEPE = [[distrib[0], 0], [1, 1], [dn_pe, 2], [1, 1]]
  DSGLOBPE = [[distrib[2], 2]]
  DSFORMPE = [[1]]
  return DSMMRYPE + DSFILEPE + DSGLOBPE + DSFORMPE

def create_pointlist_dataspace(distrib):
  """
  Create a dataspace from a flat distribution, but adapted to "fake 2d" arrays
  ie (1,N) numpy arrays.
  Mostly usefull for PointList arrays and DataArray of the related BCDataSets.
  """
  dn_pl    = distrib[1] - distrib[0]
  DSMMRYPL = [[0,0          ], [1, 1], [1, dn_pl], [1, 1]]
  DSFILEPL = [[0, distrib[0]], [1, 1], [1, dn_pl], [1, 1]]
  DSGLOBPL = [[1, distrib[2]]]
  DSFORMPL = [[0]]
  return DSMMRYPL + DSFILEPL + DSGLOBPL + DSFORMPL

def create_data_array_filter(distrib, data_shape=None):
  """
  Create an hdf dataspace for the given distribution. The kind of
  dataspace depends of the data_shape optional argument, representing
  the size of the array for which the dataspace is created in each dimension:
  - If data_shape is None or a single value, dataspace is 1d/flat
  - If data_shape is a 2d list [1, N], a dataspace adpated to pointlist is created
  - In other cases (which should correspond to true 2d array or 3d array), the
    dataspace is create from combine method (flat in memory, block in file).
  """
  if data_shape is None or len(data_shape) == 1: #Unstructured
    hdf_data_space = create_flat_dataspace(distrib)
  elif len(data_shape) == 2 and data_shape[0] == 1:
    hdf_data_space = create_pointlist_dataspace(distrib)
  else: #Structured
    hdf_data_space = create_combined_dataspace(data_shape, distrib)

  return hdf_data_space
