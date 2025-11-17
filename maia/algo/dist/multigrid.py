import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia                 import npy_pdm_gnum_dtype                         as pdm_gnum_dtype
from maia.utils           import par_utils, pr_utils, py_utils, s_numbering
from maia.utils.numbering import range_to_slab                              as HFR2S


def nb_mg_entities_from_slabs(ent_slabs):
  nb_mg_ent_loc = 0
  for ent_slab in ent_slabs:
    (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = ent_slab
    dimi = imax_slab - imin_slab
    dimj = jmax_slab - jmin_slab
    dimk = kmax_slab - kmin_slab
    slab_size = dimi*dimj*dimk
    
    shift_i = imin_slab%2
    shift_j = jmin_slab%2
    shift_k = kmin_slab%2
    
    fake_mg_ent_array = np.ones(slab_size).reshape((dimk,dimj,dimi))
    
    new_mg_j = []
    for k in range(dimk):
      new_mg_i = []
      for j in range(dimj):
        new_mg_i.append(fake_mg_ent_array[k][j][shift_i::2])
      new_mg_j.append(new_mg_i[shift_j::2])
    nb_mg_ent_loc += len(np.array(new_mg_j[shift_k::2]).reshape(-1))
  return nb_mg_ent_loc


def multigrid_s(dt, nb_lvl, comm):

  for z in PT.iter_all_Zone_t(dt):
    
    ni,nj,nk = PT.Zone.VertexSize(z)
    assert(ni%(2**nb_lvl) == 1)
    assert(nj%(2**nb_lvl) == 1)
    assert(nk%(2**nb_lvl) == 1)
    
    for subset in PT.iter_all_subsets(z):
      pr = PT.get_value(PT.Subset.getPatch(subset))
      axis_l = [0, 1, 2]
      bnd_axis = PT.Subset.normal_axis(subset)
      axis_l.pop(bnd_axis)
      if PT.Subset.GridLocation(subset) == "Vertex":
        rest_div = 0
      elif PT.Subset.GridLocation(subset) == ("CellCenter"):
        rest_div = 1
      else:
        raise NotImplementedError("'I/J/KFaceCenter' subsets are not yet managed for multigrid !")
      for axis in axis_l:
        assert((pr[axis][1]-pr[axis][0])%2 == rest_div)
  
  trees = [dt]
  for lvl in range(nb_lvl):
    dt = trees[lvl]
    mg_dt = PT.deep_copy(dt)
    
    for z in PT.iter_all_Zone_t(dt):
      vtx_distrib_n = MT.get_Distribution(z, 'Vertex')
      vtx_distrib   = PT.get_value(vtx_distrib_n)
      vtx_shape     = PT.Zone.VertexSize(z)
      
      cell_distrib_n = MT.get_Distribution(z, 'Cell')
      cell_distrib   = PT.get_value(cell_distrib_n)
      cell_shape     = PT.Zone.CellSize(z)
      cell_slabs     = HFR2S.compute_slabs(cell_shape, cell_distrib[:2])
      
      unst_cell_idx = np.array([],dtype=pdm_gnum_dtype)
      unst_cell_coarseidx = np.array([],dtype=pdm_gnum_dtype)
      for cell_slab in cell_slabs:
        (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = cell_slab
        irange = np.arange(imin_slab+1,imax_slab+1)
        jrange = np.arange(jmin_slab+1,jmax_slab+1).reshape(-1,1)
        krange = np.arange(kmin_slab+1,kmax_slab+1).reshape(-1,1,1)
        unst_cell_idx = np.concatenate((unst_cell_idx,s_numbering.ijk_to_index_from_loc(irange, jrange, krange, "CellCenter", vtx_shape).reshape(-1)))
        icoarserange = (np.arange(imin_slab,imax_slab)//2+1)
        jcoarserange = (np.arange(jmin_slab,jmax_slab)//2+1).reshape(-1,1)
        kcoarserange = (np.arange(kmin_slab,kmax_slab)//2+1).reshape(-1,1,1)
        unst_cell_coarseidx = np.concatenate((unst_cell_coarseidx,s_numbering.ijk_to_index_from_loc(icoarserange, jcoarserange, kcoarserange, "CellCenter", np.array(vtx_shape)//2+1).reshape(-1)))

      PT.new_DiscreteData('MultiGridCellInfo', loc='CellCenter', fields={'CurUnstIdx': unst_cell_idx, 'CoarseUnstIdx': unst_cell_coarseidx}, parent=z)
      # PT.new_FlowSolution('MultiGridCellInfo', loc='CellCenter', fields={'CurUnstIdx': unst_cell_idx, 'CoarseUnstIdx': unst_cell_coarseidx}, parent=z)
      
      for bc in PT.get_nodes_from_predicates(z, "ZoneBC_t/BC_t"):
        bc_distrib_n = MT.get_Distribution(bc, 'Index')
        bc_distrib   = PT.get_value(bc_distrib_n)
        bc_loc       = PT.Subset.GridLocation(bc)
        bc_pr        = PT.get_value(PT.get_node_from_name(bc, 'PointRange'))
        bc_size      = pr_utils.transform_bnd_pr_size(bc_pr, bc_loc, 'FaceCenter')
        bc_range     = py_utils.uniform_distribution_at(bc_size.prod(), comm.rank, comm.size)
        bc_slabs     = HFR2S.compute_slabs(bc_size, bc_range)
        bnd_axis     = PT.Subset.normal_axis(bc)
        bc_face_loc  = f'{["I","J","K"][bnd_axis]}FaceCenter'
        unst_bc_face_idx = np.array([],dtype=pdm_gnum_dtype)
        unst_bc_face_coarseidx = np.array([],dtype=pdm_gnum_dtype)
        nb_face_bc_loc = 0
        for bc_slab in bc_slabs:
          (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = bc_slab
          dimi = imax_slab - imin_slab
          dimj = jmax_slab - jmin_slab
          dimk = kmax_slab - kmin_slab
          nb_face_bc_loc += dimi*dimj*dimk
          irange = np.arange(imin_slab+bc_pr[0][0],imax_slab+bc_pr[0][0])
          jrange = np.arange(jmin_slab+bc_pr[1][0],jmax_slab+bc_pr[1][0]).reshape(-1,1)
          krange = np.arange(kmin_slab+bc_pr[2][0],kmax_slab+bc_pr[2][0]).reshape(-1,1,1)
          unst_bc_face_idx = np.concatenate((unst_bc_face_idx,s_numbering.ijk_to_index_from_loc(irange, jrange, krange, bc_face_loc, vtx_shape).reshape(-1)))
          icoarserange = (np.arange(imin_slab+bc_pr[0][0]-1,imax_slab+bc_pr[0][0]-1)//2+1)
          jcoarserange = (np.arange(jmin_slab+bc_pr[1][0]-1,jmax_slab+bc_pr[1][0]-1)//2+1).reshape(-1,1)
          kcoarserange = (np.arange(kmin_slab+bc_pr[2][0]-1,kmax_slab+bc_pr[2][0]-1)//2+1).reshape(-1,1,1)
          unst_bc_face_coarseidx = np.concatenate((unst_bc_face_coarseidx,s_numbering.ijk_to_index_from_loc(icoarserange, jcoarserange, kcoarserange, bc_face_loc, np.array(vtx_shape)//2+1).reshape(-1)))
        face_loc_pr = np.zeros((3,2), dtype=pdm_gnum_dtype)
        face_loc_pr[0][0] = bc_pr[0][0]
        face_loc_pr[1][0] = bc_pr[1][0]
        face_loc_pr[2][0] = bc_pr[2][0]
        face_loc_pr[0][1] = bc_pr[0][0]+bc_size[0]-1
        face_loc_pr[1][1] = bc_pr[1][0]+bc_size[1]-1
        face_loc_pr[2][1] = bc_pr[2][0]+bc_size[2]-1
        bcds = PT.new_BCDataSet('MultiGridBCFaceInfo', loc=bc_face_loc, point_range=face_loc_pr, parent=bc)
        PT.new_BCData('DirichletData', fields={'CurUnstIdx': unst_bc_face_idx, 'CoarseUnstIdx': unst_bc_face_coarseidx},parent=bcds)
        distri_face_bc = par_utils.dn_to_distribution(nb_face_bc_loc,  comm)
        MT.new_Distribution({"Index": distri_face_bc}, parent=bcds)
      
  
    for mg_z in PT.iter_all_Zone_t(mg_dt):
      
      PT.rm_nodes_from_label(mg_z, 'BCDataSet_t')
      PT.rm_nodes_from_label(mg_z, 'DiscreteData_t')
      PT.rm_nodes_from_label(mg_z, 'FlowSolution_t')
      PT.rm_nodes_from_label(mg_z, 'ZoneSubRegion_t')
      
      ni,nj,nk = PT.Zone.VertexSize(mg_z)
      mg_ni = ni//2 + 1
      mg_nj = nj//2 + 1
      mg_nk = nk//2 + 1
      
      vtx_distrib_n = MT.get_Distribution(mg_z, 'Vertex')
      vtx_distrib   = PT.get_value(vtx_distrib_n)
      vtx_shape     = PT.Zone.VertexSize(mg_z)
      vtx_slabs     = HFR2S.compute_slabs(vtx_shape, vtx_distrib[:2])
      nb_mg_vtx_loc = nb_mg_entities_from_slabs(vtx_slabs)
      
      cell_distrib_n = MT.get_Distribution(mg_z, 'Cell')
      cell_distrib   = PT.get_value(cell_distrib_n)
      cell_shape     = PT.Zone.CellSize(mg_z)
      cell_slabs     = HFR2S.compute_slabs(cell_shape, cell_distrib[:2])
      nb_mg_cell_loc = nb_mg_entities_from_slabs(cell_slabs)
      
      mg_cx, mg_cy, mg_cz = PT.Zone.coordinates(mg_z)
      new_mg_cx = []
      new_mg_cy = []
      new_mg_cz = []
      start_num_vtx = 0
      for vtx_slab in vtx_slabs:
        (imin_slab,imax_slab), (jmin_slab,jmax_slab), (kmin_slab,kmax_slab) = vtx_slab
        dimi = imax_slab - imin_slab
        dimj = jmax_slab - jmin_slab
        dimk = kmax_slab - kmin_slab
        slab_size = dimi*dimj*dimk
        
        shift_i = imin_slab%2
        shift_j = jmin_slab%2
        shift_k = kmin_slab%2
        
        mg_cx_slab = mg_cx[start_num_vtx:start_num_vtx+slab_size].reshape((dimk,dimj,dimi))
        mg_cy_slab = mg_cy[start_num_vtx:start_num_vtx+slab_size].reshape((dimk,dimj,dimi))
        mg_cz_slab = mg_cz[start_num_vtx:start_num_vtx+slab_size].reshape((dimk,dimj,dimi))
        
        start_num_vtx += slab_size
        
        new_mg_cx_j = []
        new_mg_cy_j = []
        new_mg_cz_j = []
        for k in range(dimk):
          new_mg_cx_i = []
          new_mg_cy_i = []
          new_mg_cz_i = []
          for j in range(dimj):
            new_mg_cx_i.append(mg_cx_slab[k][j][shift_i::2])
            new_mg_cy_i.append(mg_cy_slab[k][j][shift_i::2])
            new_mg_cz_i.append(mg_cz_slab[k][j][shift_i::2])
          new_mg_cx_j.append(new_mg_cx_i[shift_j::2])
          new_mg_cy_j.append(new_mg_cy_i[shift_j::2])
          new_mg_cz_j.append(new_mg_cz_i[shift_j::2])
        new_mg_cx.extend(np.array(new_mg_cx_j[shift_k::2]).reshape(-1))
        new_mg_cy.extend(np.array(new_mg_cy_j[shift_k::2]).reshape(-1))
        new_mg_cz.extend(np.array(new_mg_cz_j[shift_k::2]).reshape(-1))
      
      # Remark: need to impose R8 because empty list is I4 by default
      PT.set_value(PT.get_node_from_path(mg_z, 'GridCoordinates/CoordinateX'), np.array(new_mg_cx, dtype=np.float64))
      PT.set_value(PT.get_node_from_path(mg_z, 'GridCoordinates/CoordinateY'), np.array(new_mg_cy, dtype=np.float64))
      PT.set_value(PT.get_node_from_path(mg_z, 'GridCoordinates/CoordinateZ'), np.array(new_mg_cz, dtype=np.float64))
      
      PT.set_value(mg_z, [[mg_ni, mg_ni-1, 0],
                          [mg_nj, mg_nj-1, 0],
                          [mg_nk, mg_nk-1, 0]])
      
      distri_vtx  = par_utils.dn_to_distribution(nb_mg_vtx_loc,  comm)
      distri_cell = par_utils.dn_to_distribution(nb_mg_cell_loc, comm)
      #Remark: 'face' distribution is not used in structured mesh so imposed uniform
      mg_n_face = PT.Zone.n_face(mg_z)
      distri_face = par_utils.uniform_distribution(mg_n_face, comm)
      PT.rm_node_from_path(mg_z, ':CGNS#Distribution')
      MT.new_Distribution({"Vertex": distri_vtx, "Cell": distri_cell, "Face": distri_face}, parent=mg_z)
      
      for mg_bc in PT.get_nodes_from_predicates(mg_z, "ZoneBC_t/BC_t"):
        if PT.Subset.GridLocation(mg_bc) == "Vertex":
          pr_n = PT.get_node_from_name(mg_bc, 'PointRange')
          pr   = PT.get_value(pr_n)
          PT.set_value(pr_n, pr//2+1)
          PT.rm_nodes_from_name(mg_bc, ':CGNS#Distribution')
          distri_idx  = par_utils.dn_to_distribution(np.prod(PT.Subset.SizePerIndex(mg_bc)),  comm)
          MT.new_Distribution({"Index": distri_idx}, parent=mg_bc)
        else:
          raise NotImplementedError("BC without 'Vertex' gridlocation is not managed !")
      
      for mg_gc in PT.get_nodes_from_predicates(mg_z, "ZoneGridConnectivity_t/GridConnectivity_t") \
                 + PT.get_nodes_from_predicates(mg_z, "ZoneGridConnectivity_t/GridConnectivity1to1_t"):
        if PT.Subset.GridLocation(mg_gc) == "Vertex":
          pr_n = PT.get_node_from_name(mg_gc, 'PointRange')
          pr   = PT.get_value(pr_n)
          assert((pr[0][1]-pr[0][0])%2 == 0)
          assert((pr[1][1]-pr[1][0])%2 == 0)
          assert((pr[2][1]-pr[2][0])%2 == 0)
          PT.set_value(pr_n, pr//2+1)
          prd_n = PT.get_node_from_name(mg_gc, 'PointRangeDonor')
          if prd_n is not None:
            prd = PT.get_value(prd_n)
            PT.set_value(prd_n, prd//2+1)
          PT.rm_nodes_from_name(mg_gc, ':CGNS#Distribution')
          distri_idx  = par_utils.dn_to_distribution(np.prod(PT.Subset.SizePerIndex(mg_gc)),  comm)
          MT.new_Distribution({"Index": distri_idx}, parent=mg_gc)
        else:
          raise NotImplementedError("GC without 'Vertex' gridlocation is not managed !")
    # PT.print_tree(mg_dt)
    trees.append(mg_dt)
  
  return trees
