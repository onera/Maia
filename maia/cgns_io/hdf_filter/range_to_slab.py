def _cellToIndexes(iCell, planSize, lineSize):
    k = iCell // planSize
    j = (iCell - k*planSize) // lineSize
    i = iCell - k*planSize - j*lineSize
    return i,j,k

def compute_slabs(arrayShape, gNumInterval):
  """ Compute HDF HyperSlabs to be used in order to contiguously load a part
  of a structured tridimensionnal array.

  arrayShape   : Number of elements in x,y,z directions
  gNumInterval : semi open interval of elements to load in global numbering
  returns    : list of hyperslabs

  Each slab is a list [[istart, iend], [jstart, jend], [kstart, kend]] of
  semi open intervals, starting at zero. The flattening order for the 3d
  array is increasing i, j, k.
  """
  hSlabList = []
  Nx, Ny, Nz = arrayShape
  lineSize = Nx
  planSize = Nx*Ny

  nCellToLoad = gNumInterval[1] - gNumInterval[0]
  # print("{0} : cellInterval is [{1}:{2}[\n".format(iRank, gNumInterval[0], gNumInterval[1]))
  imin, jmin, kmin = _cellToIndexes(gNumInterval[0],   planSize, lineSize)
  imax, jmax, kmax = _cellToIndexes(gNumInterval[1]-1, planSize, lineSize)

  # print('toLoad : {0}  -- {1} {2} {3}  -- {4} {5} {6} \n'.format(
  #  nCellToLoad, imin, jmin, kmin, imax, jmax, kmax))

  istart = imin
  jstart = jmin
  kstart = kmin

  #If the line is full, merged it with next plan
  thisLineSize = min(Nx, istart+nCellToLoad) - istart
  if thisLineSize != Nx:
    jstart += 1
    if thisLineSize > 0:
      startLine  = [[istart, min(Nx, istart+nCellToLoad)], [jmin, jmin+1], [kmin, kmin+1]]
      nCellToLoad -= thisLineSize
      hSlabList.append(startLine)
      # print('startLine {0}, loaded {1} elmts\n'.format(
        # startLine, startLine[0][1] - startLine[0][0]))

  #If the plan is full, merged it with the block
  thisPlanSize = min(Ny, jstart+(nCellToLoad // Nx)) - jstart
  if thisPlanSize != Ny:
    kstart += 1
    if thisPlanSize > 0:
      startPlane = [[0, Nx], [jstart, min(Ny, jstart+(nCellToLoad // Nx))], [kmin, kmin+1]]
      nCellToLoad -= Nx*thisPlanSize
      hSlabList.append(startPlane)
      # print('startPlane {0}, loaded {1} lines ({2} elmts)\n'.format(
        # startPlane, startPlane[1][1] - startPlane[1][0], Nx*(startPlane[1][1] - startPlane[1][0])))

  thisBlockSize = min(Nz, kstart+(nCellToLoad // planSize)) - kstart
  if thisBlockSize > 0:
    centralBloc = [[0, Nx], [0, Ny], [kstart, min(Nz, kstart+(nCellToLoad // planSize))]]
    nCellToLoad -= planSize*thisBlockSize
    hSlabList.append(centralBloc)
    # print('centralBloc {0}, loaded {1} planes ({2} elmts)\n'.format(
      # centralBloc, centralBloc[2][1] - centralBloc[2][0], planSize*(centralBloc[2][1] - centralBloc[2][0])))

  if nCellToLoad >= Nx:
    endPlane = [[0, Nx], [0, (nCellToLoad // Nx)], [kmax, kmax+1]]
    nCellToLoad -= Nx*(endPlane[1][1] - endPlane[1][0])
    hSlabList.append(endPlane)
    # print('endPlane {0}, loaded {1} lines ({2} elmts)\n'.format(
      # endPlane, endPlane[1][1] - endPlane[1][0], Nx*(endPlane[1][1] - endPlane[1][0])))
  if nCellToLoad > 0:
    endLine = [[0, nCellToLoad], [jmax, jmax+1], [kmax, kmax+1]]
    nCellToLoad -= (endLine[0][1] - endLine[0][0])
    hSlabList.append(endLine)
    # print('endLine {0}, loaded {1} elmts\n'.format(
      # endLine, endLine[0][1] - endLine[0][0]))
  assert(nCellToLoad == 0)

  return hSlabList
