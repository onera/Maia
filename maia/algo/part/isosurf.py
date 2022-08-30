'''
NOTES:
- bien gérer le cas ou on fait iso-surface puis interpolation 
  pour ne pas payer la localisation 
- API avec Julien
- Actuellement : base rendue mais changer en arbre

[Questions pour Bruno] 
- Le param de level pour l'isosurf ? Uniquement 0 ??
- le type SHPERE fait avec le field ?
'''
# import sys, time


# Import from CASSIOPEE
import Converter.PyTree   as C
import Generator.PyTree   as G
import Converter.Internal as I


# Import from MAIA
from maia.transfer    import utils           as TEU
from maia.factory     import dist_from_part  as disc
from maia.pytree.maia import conventions     as conv
from maia.pytree.sids import node_inspect    as sids
from maia.utils       import np_utils,layouts

# Import from PARADIGM
import  Pypdm.Pypdm as PDM



def iso_surface_one_domain(part_zones, PDM_type, fldpath, comm):
  """
  Compute isosurface in a domain
  """
  # print("[i] ISOSURF.PY : Entree dans la fonction",flush=True)

  n_part = len(part_zones)
  dim    = 3 # Mauvaise idée le codage en dur

  # Definition of the PDM object IsoSurface
  # print("[i] ISOSURF.PY : deb def objet PDM.IsoSurface()",flush=True)
  pdm_isos = PDM.IsoSurface(comm, dim, PDM_type, n_part)
  # print("[i] ISOSURF.PY : fin def objet PDM.IsoSurface()",flush=True)

  # Loop over domains of the partition
  for i_part, part_zone in enumerate(part_zones):
    # Get NGon + NFac
    gridc_n    = I.getNodeFromName1(part_zone, 'GridCoordinates')
    cx         = I.getNodeFromName1(gridc_n  , 'CoordinateX'    )[1]
    cy         = I.getNodeFromName1(gridc_n  , 'CoordinateY'    )[1]
    cz         = I.getNodeFromName1(gridc_n  , 'CoordinateZ'    )[1]
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

    # Julien : fonction pour faire ca ?
    ngons  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.Element.CGNSName(e) == 'NGON_n']
    nfaces = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.Element.CGNSName(e) == 'NFACE_n']
    assert len(nfaces) == len(ngons) == 1

    cell_face_idx = I.getNodeFromName1(nfaces[0], "ElementStartOffset" )[1]
    cell_face     = I.getNodeFromName1(nfaces[0], "ElementConnectivity")[1]
    face_vtx_idx  = I.getNodeFromName1( ngons[0], "ElementStartOffset" )[1]
    face_vtx      = I.getNodeFromName1( ngons[0], "ElementConnectivity")[1]

    vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)

    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]
    n_edge = 0
    n_vtx  = vtx_ln_to_gn .shape[0]

    # Partition definition for PDM object
    pdm_isos.part_set(i_part,
                      n_cell,
                      n_face,
                      n_edge,
                      n_vtx,
                      cell_face_idx,
                      cell_face    ,
                      None,
                      None,
                      None,
                      face_vtx_idx ,
                      face_vtx     ,
                      cell_ln_to_gn,
                      face_ln_to_gn,
                      None,
                      vtx_ln_to_gn ,
                      vtx_coords)

    # Get field from path to compute the isosurf / Placement in PDM object
    field = I.getNodeFromPath(part_zone, fldpath)
    pdm_isos.part_field_set(i_part, field[1])
  # print("[i] ISOSURF.PY : fin param pdm.isosurf",flush=True)

  # Isosurfaces compute in PDM  
  pdm_isos.compute()
  # print("[i] ISOSURF.PY : fin compute",flush=True)

  # Mesh build from result
  results = pdm_isos.part_iso_surface_surface_get()

  n_iso_vtx = results['np_vtx_ln_to_gn'].shape[0]
  n_iso_elt = results['np_elt_ln_to_gn'].shape[0]

  iso_part_base = I.newCGNSBase('Base', cellDim=dim-1, physDim=3)

  iso_part_zone = I.newZone('zone', [[n_iso_vtx, n_iso_elt, 0]],
                            'Unstructured', parent=iso_part_base)

  ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, n_iso_elt], parent=iso_part_zone)
  I.newDataArray('ElementConnectivity', results['np_elt_vtx'    ], parent=ngon_n)
  I.newDataArray('ElementStartOffset' , results['np_elt_vtx_idx'], parent=ngon_n)

  # > Grid coordinates
  cx, cy, cz      = layouts.interlaced_to_tuple_coords(results['np_vtx_coord'])
  iso_grid_coord  = I.newGridCoordinates(parent=iso_part_zone)
  I.newDataArray('CoordinateX', cx, parent=iso_grid_coord)
  I.newDataArray('CoordinateY', cy, parent=iso_grid_coord)
  I.newDataArray('CoordinateZ', cz, parent=iso_grid_coord)
  # print("[i] ISOSURF.PY : fin traitement",flush=True)

  return iso_part_base





# def iso_surface(part_tree,var,level,comm):
def iso_surface(part_tree,
                type,fldpath,
                comm):
  ''' 
  Compute isosurface from field for a partitioned tree
  Return partition of the isosurface
  Arguments :
    - part_tree  : partitioned tree
    - type       : type of isosurface ('PLANE','SPHERE','FIELD' are available)
    - fldpath    : path to the field used for isosurface
    - comm       : MPI communicator
  '''

  # MPI infos
  mpi_rank = comm.Get_rank()
  mpi_size = comm.Get_size()
  # print("[i] ISOSURF.PY : fin def MPI",flush=True)

  # Type of isosurf
  if   type=="PLANE" : PDM_type = PDM._PDM_ISO_SURFACE_KIND_PLANE 
  elif type=="SPHERE": PDM_type = PDM._PDM_ISO_SURFACE_KIND_SPHERE
  elif type=="FIELD" : PDM_type = PDM._PDM_ISO_SURFACE_KIND_FIELD 
  else:
    print("[!][WARNING] isosurface.py : Error in type of IsoSurface ; Check your script")
    return None
  # print("[i] ISOSURF.PY : fin typeof isosurf",flush=True)

  # Distribution ??
  dist_doms = I.newCGNSTree()
  disc.discover_nodes_from_matching(dist_doms, [part_tree], 'CGNSBase_t/Zone_t', comm,
                                    merge_rule=lambda zpath : conv.get_part_prefix(zpath))

  # Domains in the partition
  part_tree_per_dom = list()
  for base in I.getNodesFromType(dist_doms,'CGNSBase_t'):
    for zone in I.getNodesFromType(dist_doms,'Zone_t'):
      part_tree_per_dom.append(TEU.get_partitioned_zones(part_tree, I.getName(base) + '/' + I.getName(zone)))
  # print("[i] ISOSURF.PY : fin def part",flush=True)

  # Piece of isosurfaces for each domains of the partition
  iso_doms = I.newCGNSTree()
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    iso_part = iso_surface_one_domain(part_zones, PDM_type, fldpath, comm)
    I._addChild(iso_doms, iso_part)

  return iso_doms