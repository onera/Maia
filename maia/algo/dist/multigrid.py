import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia                 import npy_pdm_gnum_dtype                         as pdm_gnum_dtype
from maia.utils           import par_utils, pr_utils, py_utils, s_numbering, np_utils
from maia.utils.numbering import range_to_slab                              as HFR2S

from maia.typing        import *
from maia.pytree.typing import Predicate


def _slab_half_size(slab):
  # Compute the number of entities if the slab had only one point over two
  # In each direction: (end - start + 1 - start%2) // 2
  # (<=> ceil(size / 2) if start%2==0 and  floor(size / 2) otherwise)
  _slab = np.asarray(slab)
  half_sizes = (_slab[:,1] - _slab[:,0] + 1 - _slab[:,0] % 2) // 2

  return half_sizes.prod()

def _nb_mg_entities_from_slabs(ent_slabs):
  return sum(_slab_half_size(slab) for slab in ent_slabs)

def _deepcopy_children_if(src:CGNSTree, tgt:CGNSTree, predicate:Predicate):
  """ Deepcopy children from src to tgt if they satisfy predicate """
  for child in PT.get_children_from_predicate(src, predicate):
    PT.add_child(tgt, PT.deep_copy(child))

def compute_agglomerated_parent(tree:CGNSDistTree, comm:MPIComm):
  for z in PT.iter_all_Zone_t(tree):
    idx_dim = PT.Zone.IndexDimension(z)
    
    vtx_shape  = PT.Zone.VertexSize(z)
    cell_shape = PT.Zone.CellSize(z)
    cell_slabs = HFR2S.compute_slabs(cell_shape, MT.distribution_value(z, 'Cell')[:2])

    cell_shape_coarse = tuple(s//2     for s in cell_shape)
    vtx_shape_coarse  = tuple(s//2 + 1 for s in vtx_shape)
    
    _cell_icoarseidx = list()
    _cell_jcoarseidx = list()
    _cell_kcoarseidx = list()
    for cell_slab in cell_slabs:
      # Works also for 2D meshes since krange will be 'empty' in this case
      (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = cell_slab
      icoarserange = (np.arange(imin_slab,imax_slab)//2+1)
      jcoarserange = (np.arange(jmin_slab,jmax_slab)//2+1).reshape(-1,1)
      if idx_dim == 2:
        kcoarserange = (np.ones((1,),dtype=pdm_gnum_dtype)).reshape(-1,1,1)
      else:
        kcoarserange = (np.arange(kmin_slab,kmax_slab)//2+1).reshape(-1,1,1)
      _cell_icoarseidx.append(np.tile(icoarserange, len(jcoarserange)*len(kcoarserange)))
      _cell_jcoarseidx.append(np.tile(np.tile(jcoarserange, len(icoarserange)).flatten(), len(kcoarserange)).flatten())
      _cell_kcoarseidx.append(np.tile(kcoarserange, len(icoarserange)*len(jcoarserange)).flatten())

    fields = {'I': np_utils.concatenate_np_arrays(_cell_icoarseidx)[1],
              'J': np_utils.concatenate_np_arrays(_cell_jcoarseidx)[1],
              'K': np_utils.concatenate_np_arrays(_cell_kcoarseidx)[1]}

    PT.new_DiscreteData('MultiGridCellInfo', loc='CellCenter', fields={f'{d}CoarseIdx': fields[d] for d in 'IJK'[0:idx_dim]}, parent=z)
    
    for bc in PT.get_nodes_from_predicates(z, "ZoneBC_t/BC_t"):
      # Compute FaceIdx for BC subsets: we work as if BC were FaceCenter thanks to transform_bnd_pr_size
      bc_loc       = PT.Subset.GridLocation(bc)
      bc_pr        = PT.get_np_value(PT.find_node_from_name(bc, 'PointRange'))
      bc_size      = pr_utils.transform_bnd_pr_size(bc_pr, bc_loc, 'FaceCenter')
      bc_range     = py_utils.uniform_distribution_at(bc_size.prod(), comm.rank, comm.size)
      bc_slabs     = HFR2S.compute_slabs(bc_size, bc_range)
      bnd_axis     = PT.Subset.normal_axis(bc)
      bc_face_loc  = f"{'IJK'[bnd_axis]}{'Edge' if idx_dim == 2 else 'Face'}Center"
      _bc_face_icoarseidx = list()
      _bc_face_jcoarseidx = list()
      _bc_face_kcoarseidx = list()

      for bc_slab in bc_slabs:
        (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = bc_slab
        # TODO maybe shift missing ?
        icoarserange = (np.arange(imin_slab+bc_pr[0][0]-1,imax_slab+bc_pr[0][0]-1)//2+1)
        jcoarserange = (np.arange(jmin_slab+bc_pr[1][0]-1,jmax_slab+bc_pr[1][0]-1)//2+1).reshape(-1,1)
        if idx_dim == 2:
          kcoarserange = (np.ones((1,),dtype=pdm_gnum_dtype)).reshape(-1,1,1)
        else:
          kcoarserange = (np.arange(kmin_slab+bc_pr[2][0]-1,kmax_slab+bc_pr[2][0]-1)//2+1).reshape(-1,1,1)
        _bc_face_icoarseidx.append(np.tile(icoarserange, len(jcoarserange)*len(kcoarserange)))
        _bc_face_jcoarseidx.append(np.tile(np.tile(jcoarserange, len(icoarserange)).flatten(), len(kcoarserange)).flatten())
        _bc_face_kcoarseidx.append(np.tile(kcoarserange, len(icoarserange)*len(jcoarserange)).flatten())
      
      face_loc_pr = np.zeros_like(bc_pr)
      face_loc_pr[:,0] = bc_pr[:, 0]
      face_loc_pr[:,1] = bc_pr[:, 0] + bc_size - 1

      fields = {'I': np_utils.concatenate_np_arrays(_bc_face_icoarseidx)[1],
                'J': np_utils.concatenate_np_arrays(_bc_face_jcoarseidx)[1],
                'K': np_utils.concatenate_np_arrays(_bc_face_kcoarseidx)[1]}

      bcds = PT.new_BCDataSet('MultiGridBCFaceInfo', loc=bc_face_loc, point_range=face_loc_pr, parent=bc)
      PT.new_BCData('DirichletData', fields={f'{d}CoarseIdx': fields[d] for d in 'IJK'[0:idx_dim]},parent=bcds)
      PT.new_Descriptor('BCStructuredLocation', bc_face_loc, parent=bcds)
      distri_face_bc = par_utils.dn_to_distribution(bc_range[1]-bc_range[0],  comm)
      MT.new_Distribution({"Index": distri_face_bc}, parent=bcds)

def create_agglomerated_tree(tree:CGNSDistTree, comm:MPIComm) -> CGNSDistTree:
  COPY_ON_BASE = ~PT.pred.label_is('Zone_t')
  COPY_ON_ZONE = ~PT.pred.label_in(['GridCoordinates_t', 'ArbitraryGridMotion_t',
                                    'FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t',
                                    'ZoneBC_t', 'ZoneGridConnectivity_t', 'ZoneType_t', 'ZoneBC_t']) \
                &~PT.pred.name_is(':CGNS#Distribution')
  COPY_ON_BC   = PT.pred.label_in(['Descriptor_t', 'FamilyName_t', 'AdditionalFamilyName_t', 'Ordinal_t'])
  COPY_ON_GC   = COPY_ON_BC | PT.pred.name_in(['GridConnectivityProperty', 'GridConnectivityType', 'Transform'])
  mg_tree = PT.new_CGNSTree()
  for base in PT.iter_all_CGNSBase_t(tree):

    mg_base = PT.new_child(mg_tree,
                           PT.get_name(base),
                           PT.get_label(base),
                           PT.get_np_value(base).copy())
    _deepcopy_children_if(base, mg_base, COPY_ON_BASE)

    for zone in PT.iter_all_Zone_t(base):

      vtx_slabs  = HFR2S.compute_slabs(PT.Zone.VertexSize(zone),
                                       MT.distribution_value(zone, 'Vertex')[:2])
      cell_slabs = HFR2S.compute_slabs(PT.Zone.CellSize(zone),
                                       MT.distribution_value(zone, 'Cell')[:2])
      nb_mg_vtx_loc  = _nb_mg_entities_from_slabs(vtx_slabs)
      nb_mg_cell_loc = _nb_mg_entities_from_slabs(cell_slabs)

      # Create MG Zone
      mg_zone_size = np.copy(PT.get_np_value(zone))
      mg_zone_size[:,0] = mg_zone_size[:,0] // 2 + 1 # Vtx
      mg_zone_size[:,1] = mg_zone_size[:,1] // 2     # Cell
      mg_zone = PT.new_Zone(PT.get_name(zone), type='Structured', size=mg_zone_size, parent=mg_base)

      ini_coords = {key : val for key,val in PT.Zone.coordinates(zone)._asdict().items()
                    if val is not None}
      mg_coords:Dict[str, List[NDArray]] = {key: list() for key in ini_coords}
      start_vtx = 0
      
      # Filter coordinates: reuse slab because we can not simply remove one vtx
      # over two (domain boundaries)
      # The idea is put data in dimensional form to extract one point over two in each direction,
      # and then we recover flat data (for each slab)
      for vtx_slab in vtx_slabs:
        (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = vtx_slab
        dimi = imax_slab - imin_slab
        dimj = jmax_slab - jmin_slab
        dimk = kmax_slab - kmin_slab
        
        end_vtx = start_vtx + (dimi*dimj*dimk)
        
        shift_i = imin_slab%2
        shift_j = jmin_slab%2
        shift_k = kmin_slab%2

        for key, ini_coord in ini_coords.items():
          ini_coord_shaped = ini_coord[start_vtx:end_vtx].reshape((dimi,dimj,dimk), order='F')
          mg_coords[key].append(ini_coord_shaped[shift_i::2, shift_j::2, shift_k::2].reshape(-1, order='F'))
        
        start_vtx = end_vtx
        
      # Concatenate lists when creating coordinates
      PT.new_GridCoordinates(PT.get_name(PT.find_child_from_label(zone, 'GridCoordinates_t')),
                             fields={key:np_utils.concatenate_np_arrays(val)[1] for key,val in mg_coords.items()},
                             parent=mg_zone)

      for zbc, bc in PT.get_nodes_from_predicates(zone, "ZoneBC_t/BC_t", ancestors=True):
        mg_zbc = PT.update_child(mg_zone, PT.get_name(zbc), 'ZoneBC_t')
        assert PT.Subset.GridLocation(bc) == "Vertex", "Only Vertex located subsets are managed"

        mg_bc = PT.new_BC(PT.get_name(bc), PT.get_str_value(bc), parent=mg_zbc)

        if (pr := PT.get_child_from_name(bc, 'PointRange')) is not None:
          PT.new_IndexRange('PointRange', value = PT.get_np_value(pr)//2 + 1, parent=mg_bc)
        else:
          raise RuntimeError("PointList are not supported for BC_t nodes")

        _deepcopy_children_if(bc, mg_bc, COPY_ON_BC)
        distri_idx = par_utils.dn_to_distribution(PT.Subset.n_elem(mg_bc), comm)
        MT.new_Distribution({"Index": distri_idx}, parent=mg_bc)
      
      for zgc, gc in PT.iter_children_from_predicates(mg_zone, ['ZoneGridConnectivity_t', PT.pred.IS_GC], ancestors=True):
        mg_zgc = PT.update_child(mg_zone, PT.get_name(zgc), 'ZoneGridConnectivity_t')
        assert PT.Subset.GridLocation(gc) == "Vertex", "Only Vertex located subsets are managed"

        mg_gc = PT.new_node(PT.get_name(gc), PT.get_label(gc), PT.get_value(gc), parent=mg_zgc)

        if (pr := PT.get_child_from_name(gc, 'PointRange')) is not None:
          PT.new_IndexRange('PointRange', value = PT.get_np_value(pr)//2 + 1, parent=mg_gc)
        else:
          raise RuntimeError("PointList are not supported for GridConnectivity(1to1)_t nodes")
        if (prd := PT.get_child_from_name(gc, 'PointRangeDonor')) is not None:
          PT.new_IndexRange('PointRangeDonor', value = PT.get_np_value(prd)//2 + 1, parent=mg_gc)
        elif PT.get_child_from_name(gc, 'PointListDonor') is not None:
          raise RuntimeError("PointListDonor are not supported for GridConnectivity(1to1)_t nodes")

        _deepcopy_children_if(gc, mg_gc, COPY_ON_GC)
        distri_idx  = par_utils.dn_to_distribution(PT.Subset.n_elem(mg_gc),  comm)
        MT.new_Distribution({"Index": distri_idx}, parent=mg_gc)

      _deepcopy_children_if(zone, mg_zone, COPY_ON_ZONE)

      zone_distri = {"Vertex" : par_utils.dn_to_distribution(nb_mg_vtx_loc,  comm),
                      "Cell"  : par_utils.dn_to_distribution(nb_mg_cell_loc, comm)}
      #Remark: 'face' distribution is not used in structured mesh so imposed uniform
      if PT.Zone.IndexDimension(mg_zone) == 3:
        zone_distri["Face"] = par_utils.uniform_distribution(PT.Zone.n_face(mg_zone), comm)
      MT.new_Distribution(zone_distri, parent=mg_zone)

  return mg_tree

def agglomerate_s(tree:CGNSDistTree, comm:MPIComm) -> CGNSDistTree:

  mg_tree = create_agglomerated_tree(tree, comm)
  compute_agglomerated_parent(tree, comm)
  return mg_tree

def multigrid_s(dt, nb_lvl, comm):

  for z in PT.iter_all_Zone_t(dt):
    
    for dir,nci in zip('IJK', PT.Zone.CellSize(z)):
      if nci % (2**nb_lvl) != 0:
        raise ValueError(f"Invalid number of cells in {dir} direction: {nci} is not a multiple of 2^{nb_lvl}")
    
    predicates = [PT.pred.label_in(['ZoneBC_t', 'ZoneGridConnectivity_t']),
                  PT.pred.label_is('BC_t') | PT.pred.IS_GC] 
    for subset in PT.iter_children_from_predicates(z, predicates):
      if PT.get_child_from_name(subset, 'PointList') is not None:
        raise ValueError("Only PointRange subsets are supported")
      if PT.Subset.GridLocation(subset) not in ['Vertex', 'CellCenter']:
        raise NotImplementedError("'I/J/KFaceCenter' subsets are not yet managed for multigrid !")
      rest_div = 1 if PT.Subset.GridLocation(subset) == 'Vertex' else 0
      pr_size = PT.Subset.SizePerIndex(subset)
      # Skip null direction when checking
      _pr_size = tuple(k for d,k in enumerate(pr_size) if d != PT.Subset.normal_axis(subset))
      if any(s % (2**nb_lvl) != rest_div for s in _pr_size):
        raise ValueError(f"Subset {PT.get_name(subset)} has a PointRange size incompatible with requested agglomeration")

  trees = [dt]
  for lvl in range(nb_lvl):
    trees.append(agglomerate_s(trees[-1], comm))
  
  return trees
