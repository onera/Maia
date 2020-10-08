
def get_zone_nb_vtx( zone ):
  """
     Return number of elements for a zone z (if unstructured), z obtained by getzones.
     Zone_t : VertexDim[indexDimeension], CellDim[indexDimension]
     Usage: nelts=getzoneElementNumber( z )
  """
  n_vtx = 1
  indexDimension = len(zone[1])
  for idx in range(indexDimension):
    n_vtx *= zone[1][idx][0]
  return n_vtx

def get_zone_nb_cell( zone ):
  """
     Return number of elements for a zone z (if unstructured), z obtained by getzones.
     Zone_t : VertexDim[indexDimeension], CellDim[indexDimension]
     Usage: nelts=getzoneElementNumber( z )
  """
  n_cell = 1
  indexDimension = len(zone[1])
  for idx in range(indexDimension):
    n_cell *= zone[1][idx][1]
  return n_cell
