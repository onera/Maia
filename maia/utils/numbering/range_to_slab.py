def cell_to_indexes(i_cell, plan_size, line_size):
  """ Compute the (i,j,k) indices of a cell or a node
  from its global index.
  Numbering convention is increasing i,j,k. Here global index
  and i,j,k start at 0.
  """
  k = i_cell // plan_size
  j = (i_cell - k*plan_size) // line_size
  i = i_cell - k*plan_size - j*line_size
  return i,j,k

def compute_slabs(array_shape, gnum_interval):
  """ Compute HDF HyperSlabs to be used in order to contiguously load a part
  of a structured tridimensionnal array.

  array_shape   : Number of elements in x,y,z directions
  gnum_interval : semi open interval of elements to load in global numbering
  returns    : list of hyperslabs

  Each slab is a list [[istart, iend], [jstart, jend], [kstart, kend]] of
  semi open intervals, starting at zero. The flattening order for the 3d
  array is increasing i, j, k.
  """
  dim = len(array_shape)
  hslab_list = []
  nx, ny = array_shape[0:2]
  if dim == 3 : nz = array_shape[2]
  line_size = nx
  plan_size = nx*ny

  ncell_to_load = gnum_interval[1] - gnum_interval[0]
  imin, jmin, kmin = cell_to_indexes(gnum_interval[0],   plan_size, line_size)
  imax, jmax, kmax = cell_to_indexes(gnum_interval[1]-1, plan_size, line_size)


  istart = imin
  jstart = jmin
  kstart = kmin

  #If the line is full, merged it with next plan
  this_line_size = min(nx, istart+ncell_to_load) - istart
  if this_line_size != nx:
    jstart += 1
    if this_line_size > 0:
      start_line  = [[istart, min(nx, istart+ncell_to_load)], [jmin, jmin+1], [kmin, kmin+1]]
      ncell_to_load -= this_line_size
      hslab_list.append(start_line)

  #If the plan is full, merged it with the block
  this_plan_size = min(ny, jstart+(ncell_to_load // nx)) - jstart
  if this_plan_size != ny:
    kstart += 1
    if this_plan_size > 0:
      start_plane = [[0, nx], [jstart, min(ny, jstart+(ncell_to_load // nx))], [kmin, kmin+1]]
      ncell_to_load -= nx*this_plan_size
      hslab_list.append(start_plane)

  if dim == 3:
    this_block_size = min(nz, kstart+(ncell_to_load // plan_size)) - kstart
    if this_block_size > 0:
      central_block = [[0, nx], [0, ny], [kstart, min(nz, kstart+(ncell_to_load // plan_size))]]
      ncell_to_load -= plan_size*this_block_size
      hslab_list.append(central_block)

  if ncell_to_load >= nx:
    end_plane = [[0, nx], [0, (ncell_to_load // nx)], [kmax, kmax+1]]
    ncell_to_load -= nx*(end_plane[1][1] - end_plane[1][0])
    hslab_list.append(end_plane)
    
  if ncell_to_load > 0:
    end_line = [[0, ncell_to_load], [jmax, jmax+1], [kmax, kmax+1]]
    ncell_to_load -= (end_line[0][1] - end_line[0][0])
    hslab_list.append(end_line)
    
  assert(ncell_to_load == 0)

  return hslab_list
