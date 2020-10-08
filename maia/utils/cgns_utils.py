
def get_zone_nb_vtx( zone ):
  """
     Return number of elements for a zone z (if unstructured), z obtained by getzones.
     Zone_t : VertexDim[indexDimeension], CellDim[idx_dim]
     Usage: nelts=getzoneElementNumber( z )
  """
  n_vtx = 1
  idx_dim = len(zone[1])
  for idx in range(idx_dim):
    n_vtx *= zone[1][idx][0]
  return n_vtx

def get_zone_nb_cell( zone ):
  """
     Return number of elements for a zone z (if unstructured), z obtained by getzones.
     Zone_t : VertexDim[indexDimeension], CellDim[idx_dim]
     Usage: nelts=getzoneElementNumber( z )
  """
  n_cell = 1
  idx_dim = len(zone[1])
  for idx in range(idx_dim):
    n_cell *= zone[1][idx][1]
  return n_cell

def get_zone_nb_vtx_bnd( zone ):
  """
     Return number of elements for a zone z (if unstructured), z obtained by getzones.
     Zone_t : VertexDim[indexDimeension], CellDim[idx_dim]
     Usage: nelts=getzoneElementNumber( z )
  """
  n_vtx_bnd = 1
  idx_dim = len(zone[1])
  for idx in range(idx_dim):
    n_vtx_bnd *= zone[1][idx][2]
  return n_vtx_bnd
