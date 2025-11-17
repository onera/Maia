import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils           import par_utils, pr_utils, py_utils, s_numbering, np_utils
from maia.utils.numbering import range_to_slab                              as HFR2S


def _slab_half_size(slab):
  # Compute the number of entities if the slab had only one point over two
  # In each direction: (end - start + 1 - start%2) // 2
  # (<=> ceil(size / 2) if start%2==0 and  floor(size / 2) otherwise)
  _slab = np.asarray(slab)
  half_sizes = (_slab[:,1] - _slab[:,0] + 1 - _slab[:,0] % 2) // 2

  return half_sizes.prod()

def nb_mg_entities_from_slabs(ent_slabs):
  return sum(_slab_half_size(slab) for slab in ent_slabs)


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
    dt = trees[lvl]
    mg_dt = PT.deep_copy(dt)
    
    for z in PT.iter_all_Zone_t(dt):
      idx_dim = PT.Zone.IndexDimension(z)
      vtx_shape = PT.Zone.VertexSize(z)
      
      cell_distrib = MT.distribution_value(z, 'Cell')
      cell_shape   = PT.Zone.CellSize(z)
      cell_slabs   = HFR2S.compute_slabs(cell_shape, cell_distrib[:2])
      
      _unst_cell_idx = list()
      _unst_cell_coarseidx = list()
      for cell_slab in cell_slabs:
        # Works also for 2D meshes since krange will be 'empty' in this case
        (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = cell_slab
        irange = np.arange(imin_slab+1,imax_slab+1)
        jrange = np.arange(jmin_slab+1,jmax_slab+1).reshape(-1,1)
        krange = np.arange(kmin_slab+1,kmax_slab+1).reshape(-1,1,1)
        _unst_cell_idx.append(s_numbering.ijk_to_index(irange, jrange, krange, cell_shape).reshape(-1))
        icoarserange = (np.arange(imin_slab,imax_slab)//2+1)
        jcoarserange = (np.arange(jmin_slab,jmax_slab)//2+1).reshape(-1,1)
        kcoarserange = (np.arange(kmin_slab,kmax_slab)//2+1).reshape(-1,1,1)
        _unst_cell_coarseidx.append(s_numbering.ijk_to_index(icoarserange, jcoarserange, kcoarserange, np.array(cell_shape)//2).reshape(-1))

      unst_cell_idx = np_utils.concatenate_np_arrays(_unst_cell_idx)[1]
      unst_cell_coarseidx = np_utils.concatenate_np_arrays(_unst_cell_coarseidx)[1]

      PT.new_DiscreteData('MultiGridCellInfo', loc='CellCenter', fields={'CurUnstIdx': unst_cell_idx, 'CoarseUnstIdx': unst_cell_coarseidx}, parent=z)
      # PT.new_FlowSolution('MultiGridCellInfo', loc='CellCenter', fields={'CurUnstIdx': unst_cell_idx, 'CoarseUnstIdx': unst_cell_coarseidx}, parent=z)
      
      for bc in PT.get_nodes_from_predicates(z, "ZoneBC_t/BC_t"):
        # Compute FaceIdx for BC subsets: we work as if BC were FaceCenter thanks to transform_bnd_pr_size
        bc_loc       = PT.Subset.GridLocation(bc)
        bc_pr        = PT.get_np_value(PT.find_node_from_name(bc, 'PointRange'))
        bc_size      = pr_utils.transform_bnd_pr_size(bc_pr, bc_loc, 'FaceCenter')
        bc_range     = py_utils.uniform_distribution_at(bc_size.prod(), comm.rank, comm.size)
        bc_slabs     = HFR2S.compute_slabs(bc_size, bc_range)
        bnd_axis     = PT.Subset.normal_axis(bc)
        bc_face_loc  = f'{["I","J","K"][bnd_axis]}{'Edge' if idx_dim == 2 else 'Face'}Center'
        _unst_bc_face_idx = list()
        _unst_bc_face_coarseidx = list()

        for bc_slab in bc_slabs:
          (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = bc_slab
          # TODO maybe shift missing ?
          irange = np.arange(imin_slab+bc_pr[0][0],imax_slab+bc_pr[0][0])
          jrange = np.arange(jmin_slab+bc_pr[1][0],jmax_slab+bc_pr[1][0]).reshape(-1,1)
          if idx_dim == 2:
            _unst_bc_face_idx.append(s_numbering.ij_to_index_from_loc(irange, jrange, bc_face_loc, vtx_shape).reshape(-1))
          else:
            krange = np.arange(kmin_slab+bc_pr[2][0],kmax_slab+bc_pr[2][0]).reshape(-1,1,1)
            _unst_bc_face_idx.append(s_numbering.ijk_to_index_from_loc(irange, jrange, krange, bc_face_loc, vtx_shape).reshape(-1))
          icoarserange = (np.arange(imin_slab+bc_pr[0][0]-1,imax_slab+bc_pr[0][0]-1)//2+1)
          jcoarserange = (np.arange(jmin_slab+bc_pr[1][0]-1,jmax_slab+bc_pr[1][0]-1)//2+1).reshape(-1,1)
          if idx_dim == 2:
            _unst_bc_face_coarseidx.append(s_numbering.ij_to_index_from_loc(icoarserange, jcoarserange, bc_face_loc, np.array(vtx_shape)//2+1).reshape(-1))
          else:
            kcoarserange = (np.arange(kmin_slab+bc_pr[2][0]-1,kmax_slab+bc_pr[2][0]-1)//2+1).reshape(-1,1,1)
            _unst_bc_face_coarseidx.append(s_numbering.ijk_to_index_from_loc(icoarserange, jcoarserange, kcoarserange, bc_face_loc, np.array(vtx_shape)//2+1).reshape(-1))
        
        face_loc_pr = np.zeros_like(bc_pr)
        face_loc_pr[:,0] = bc_pr[:, 0]
        face_loc_pr[:,1] = bc_pr[:, 0] + bc_size - 1

        unst_bc_face_idx       = np_utils.concatenate_np_arrays(_unst_bc_face_idx)[1]
        unst_bc_face_coarseidx = np_utils.concatenate_np_arrays(_unst_bc_face_coarseidx)[1]

        bcds = PT.new_BCDataSet('MultiGridBCFaceInfo', loc=bc_face_loc, point_range=face_loc_pr, parent=bc)
        PT.new_BCData('DirichletData', fields={'CurUnstIdx': unst_bc_face_idx, 'CoarseUnstIdx': unst_bc_face_coarseidx},parent=bcds)
        distri_face_bc = par_utils.dn_to_distribution(bc_range[1]-bc_range[0],  comm)
        MT.new_Distribution({"Index": distri_face_bc}, parent=bcds)

  
    for mg_z in PT.iter_all_Zone_t(mg_dt):
      
      PT.rm_nodes_from_label(mg_z, 'BCDataSet_t')
      PT.rm_nodes_from_label(mg_z, 'DiscreteData_t')
      PT.rm_nodes_from_label(mg_z, 'FlowSolution_t')
      PT.rm_nodes_from_label(mg_z, 'ZoneSubRegion_t')
      
      vtx_distrib   = MT.distribution_value(mg_z, 'Vertex')
      vtx_shape     = PT.Zone.VertexSize(mg_z)
      vtx_slabs     = HFR2S.compute_slabs(vtx_shape, vtx_distrib[:2])
      nb_mg_vtx_loc = nb_mg_entities_from_slabs(vtx_slabs)
      
      cell_distrib   = MT.distribution_value(mg_z, 'Cell')
      cell_shape     = PT.Zone.CellSize(mg_z)
      cell_slabs     = HFR2S.compute_slabs(cell_shape, cell_distrib[:2])
      nb_mg_cell_loc = nb_mg_entities_from_slabs(cell_slabs)
      
      ini_coords = {key : val for key,val in PT.Zone.coordinates(mg_z)._asdict().items()
                    if val is not None}
      mg_coords  = {key: list() for key in ini_coords}
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

        
      # Concatenate lists
      mg_coords = {key: np_utils.concatenate_np_arrays(val)[1] for key,val in mg_coords.items()}
      
      grid_co = PT.find_child_from_label(mg_z, 'GridCoordinates_t')
      for key, val in mg_coords.items():
        PT.set_value(PT.find_child_from_name(grid_co, key), val)

      # Update zone sizes
      cell_size = PT.get_np_value(mg_z)[:,1]
      vtx_size  = PT.get_np_value(mg_z)[:,0]
      cell_size[:] = cell_size // 2
      vtx_size[:]  =  vtx_size // 2 + 1
      

      zone_distri = {"Vertex" : par_utils.dn_to_distribution(nb_mg_vtx_loc,  comm),
                     "Cell"   : par_utils.dn_to_distribution(nb_mg_cell_loc, comm)}
      #Remark: 'face' distribution is not used in structured mesh so imposed uniform
      if PT.Zone.IndexDimension(mg_z) == 3:
        zone_distri["Face"] = par_utils.uniform_distribution(PT.Zone.n_face(mg_z), comm)
      MT.new_Distribution(zone_distri, parent=mg_z)
      
      for mg_bc in PT.get_nodes_from_predicates(mg_z, "ZoneBC_t/BC_t"):
        assert PT.Subset.GridLocation(mg_bc) == "Vertex", "Only Vertex located subsets are managed"
        pr_n = PT.find_node_from_name(mg_bc, 'PointRange')
        pr   = PT.get_np_value(pr_n)
        PT.set_value(pr_n, pr//2+1)
        distri_idx  = par_utils.dn_to_distribution(PT.Subset.n_elem(mg_bc), comm)
        MT.new_Distribution({"Index": distri_idx}, parent=mg_bc)
      
      for mg_gc in PT.iter_children_from_predicates(mg_z, ['ZoneGridConnectivity_t', PT.pred.IS_GC]):
        assert PT.Subset.GridLocation(mg_gc) == "Vertex", "Only Vertex located subsets are managed"
        pr_n = PT.find_node_from_name(mg_gc, 'PointRange')
        pr   = PT.get_np_value(pr_n)
        assert (pr[:,1] - pr[:,0] % 2 == 0).all()
        PT.set_value(pr_n, pr//2+1)
        prd_n = PT.get_node_from_name(mg_gc, 'PointRangeDonor')
        if prd_n is not None:
          prd = PT.get_np_value(prd_n)
          PT.set_value(prd_n, prd//2+1) # TODO : what if PR/PRd decreasing ?
        distri_idx  = par_utils.dn_to_distribution(PT.Subset.n_elem(mg_gc),  comm)
        MT.new_Distribution({"Index": distri_idx}, parent=mg_gc)

    # PT.print_tree(mg_dt)
    trees.append(mg_dt)
  
  return trees
