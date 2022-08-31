'''
NOTES:

[Questions - Julien]
  - API
  - disc.discover_nodes_from_matching() ca sert à quoix ?

[Questions - Bruno] 
  - le type SHPERE fait avec le field ?

[A FAIRE] 
  -> prévoir intégration du niveau de l'iso
  -> prévoir intégration de l'interpolation

'''
# import sys, time


# Import from CASSIOPEE
import Converter.PyTree   as C
import Generator.PyTree   as G
import Converter.Internal as I


# Import from MAIA
from maia.transfer    import utils                as TEU
from maia.factory     import dist_from_part       as disc
from maia.pytree.maia import conventions          as conv
from maia.pytree.sids import node_inspect         as sids
from maia.utils       import np_utils,layouts

# Import from PARADIGM
import  Pypdm.Pypdm as PDM



def iso_surface_one_domain(part_zones, PDM_type, fldpath, comm, iso_value=0.):
  """
  Compute isosurface in a domain
  """

  n_part = len(part_zones)
  dim    = 3 # Mauvaise idée le codage en dur

  # Definition of the PDM object IsoSurface
  pdm_isos = PDM.IsoSurface(comm, dim, PDM_type, n_part)

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

  # Isosurfaces compute in PDM  
  pdm_isos.compute()


  # Mesh build from result
  results = pdm_isos.part_iso_surface_surface_get()
  n_iso_vtx = results['np_vtx_ln_to_gn'].shape[0]
  n_iso_elt = results['np_elt_ln_to_gn'].shape[0]

  print(results.keys())

  # > Tree construction
  iso_part_tree = I.newCGNSTree()
  iso_part_base = I.newCGNSBase('Base', cellDim=dim-1, physDim=3, parent=iso_part_tree)

  # > Zone construction
  iso_part_zone = I.newZone(f'zone.{comm.Get_rank()}',
                            [[n_iso_vtx, n_iso_elt, 0]],
                            'Unstructured',
                            parent=iso_part_base)
  # > Grid coordinates
  cx, cy, cz      = layouts.interlaced_to_tuple_coords(results['np_vtx_coord'])
  iso_grid_coord  = I.newGridCoordinates(parent=iso_part_zone)
  I.newDataArray('CoordinateX', cx, parent=iso_grid_coord)
  I.newDataArray('CoordinateY', cy, parent=iso_grid_coord)
  I.newDataArray('CoordinateZ', cz, parent=iso_grid_coord)

  # > Elements
  ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, n_iso_elt], parent=iso_part_zone)
  I.newDataArray('ElementConnectivity', results['np_elt_vtx'    ]     , parent=ngon_n)
  I.newDataArray('ElementStartOffset' , results['np_elt_vtx_idx']     , parent=ngon_n)

  gn_elmt = I.newUserDefinedData(':CGNS#GlobalNumbering', parent=ngon_n )
  I.newDataArray('Element', results['np_elt_ln_to_gn']  , parent=gn_elmt)

  # > LN to GN
  gn_zone = I.newUserDefinedData(':CGNS#GlobalNumbering', parent=iso_part_zone)
  I.newDataArray('Vertex', results['np_vtx_ln_to_gn'], parent=gn_zone)
  I.newDataArray('Cell'  , results['np_elt_ln_to_gn'], parent=gn_zone)

  elmt_parent_gn = results["np_elt_parent_g_num"]
  print("results[np_elt_parent_g_num]=",results["np_elt_parent_g_num"].shape,flush=True)

  # > Interpolation -> should be moved


  return iso_part_base




def iso_surface(part_tree,
                type,fldpath,
                comm,
                iso_value=0.):
  ''' 
  Compute isosurface from field for a partitioned tree
  Return partition of the isosurface
  Arguments :
    - part_tree  : partitioned tree
    - type       : type of isosurface ('PLANE','SPHERE','FIELD' are available)
    - fldpath    : path to the field used for isosurface
    - comm       : MPI communicator
    - iso_value  : isosurface value
  '''

  # Type of isosurf
  if   type=="PLANE" : PDM_type = PDM._PDM_ISO_SURFACE_KIND_PLANE 
  elif type=="SPHERE": PDM_type = PDM._PDM_ISO_SURFACE_KIND_SPHERE
  elif type=="FIELD" : PDM_type = PDM._PDM_ISO_SURFACE_KIND_FIELD 
  else:
    print("[!][WARNING] isosurface.py : Error in type of IsoSurface ; Check your script")
    return None

  # from the part_tree, retrieve the paths of the distributed blocks
  # and return a dictionnary associating each path to the list of the corresponding
  # partitioned zones
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()
  assert(len(part_tree_per_dom)==1) # On gère le monodomaine pour l'instanx


  # Piece of isosurfaces for each domains of the partition
  iso_doms = I.newCGNSTree()
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    iso_part = iso_surface_one_domain(part_zones, PDM_type, fldpath, comm)
    I._addChild(iso_doms, iso_part)
  return iso_doms