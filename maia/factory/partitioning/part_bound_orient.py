from mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia

from maia.utils    import par_utils
from maia.transfer import protocols as EP

is_ngon_3d = lambda z : PT.Zone.CellDimension(z) == 3 and PT.Zone.has_ngon_elements(z)

def orientation_preserved(part_zones, comm):
  """
  Return False if partitions interface faces has be reoriented during split
  operation in order to force outward normal (typically if
  preserve_orientation is not set to True).

  Only relevant for 3D NGON zones.
  """
  assert all([is_ngon_3d(z) for z in part_zones]), "Only 3D NGon zones are supported"

  gnum_list = list()
  data_list = list()
  for part_zone in part_zones:
    ngon_node = PT.Zone.NGonNode(part_zone)
    pe_n = PT.get_child_from_name(ngon_node, 'ParentElements')
    face_gnum = MT.getGlobalNumbering(ngon_node, 'Element')[1]

    if pe_n is not None:
      # PE case : flag external faces having only a left parent, ie pe[iFace,1] == 0
      ext_faces = np.nonzero(pe_n[1][:,1] == 0)[0]
      
      gnum_list.append(face_gnum[ext_faces])
      data_list.append(np.ones(ext_faces.size, np.int32))
    else:
      # NFace   : flag faces having a left parent, ie positive sign in cell_face connectivity
      nface_node = PT.Zone.NFaceNode(part_zone)
      ng_offset = PT.Element.Range(ngon_node)[0]
      cell_face = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]
      
      gnum_list.append(face_gnum[np.abs(cell_face) - ng_offset])
      data_list.append(np.sign(cell_face))

  # In both cases, we flagged faces having a left parent. When summing flags, if a face has a value > 1, it means
  # that it has two times a left parent, and thus that faces has been reverted after split to have output normal
  out = EP.part_to_block(data_list, None, gnum_list, comm, reduce_func=EP.reduce_sum)
  return not comm.allreduce((out > 1).any(), op=MPI.LOR)


def preserve_orientation(part_zones, comm):
  """
  Swap the orientation of some partition interface faces in order to have a unique
  orientation for each boudary face.
  This function is usefull if the mesh has not been splitted with preserve_orientation=True

  Only relevant for 3D NGON zones.
  """
  assert all([is_ngon_3d(z) for z in part_zones]), "Only 3D NGon zones are supported"

  zone_proc_offset = par_utils.dn_to_distribution(len(part_zones), comm)[0]

  bnd_list  = list()
  gnum_list = list()
  data_list = list()
  for izone, part_zone in enumerate(part_zones):
    cur_zone_glob = zone_proc_offset + izone
    ngon_node  = PT.Zone.NGonNode(part_zone)
    face_gnum = MT.getGlobalNumbering(ngon_node, 'Element')[1]
    if PT.get_child_from_name(ngon_node, 'ParentElements') is None:
      maia.algo.nface_to_pe(part_zone)
    pe = PT.get_child_from_name(ngon_node, 'ParentElements')[1]
    
    # Faces having a right parent == 0. If same face has 2 times right parent == 0,
    # it must a swapped on one of the two partitions
    ext_faces_left_pe = np.nonzero(pe[:,1] == 0)[0]
    
    bnd_list.append(ext_faces_left_pe)
    gnum_list.append(face_gnum[ext_faces_left_pe])
    data_list.append(cur_zone_glob*np.ones(ext_faces_left_pe.size, np.int32))
    
  # Gather data to identify faces having two times right parent == 0
  PTB = EP.PartToBlock(None, gnum_list, comm, keep_multiple=True)
  mask = PTB.getBlockGnumCountCopy() >= 2 # <-- these ones
  distri = PTB.getDistributionCopy()

  # In addition, exchange partition id and reduce with min to choose a master
  p_stride = [np.ones(p_f.size, dtype=np.int32) for p_f in data_list]
  dist_stride, dist_data = PTB.exchange_field(data_list, p_stride)
  dist_data = EP.reduce_min(dist_data, dist_stride)
  
  # Send back master part. id to partitions (we do it only for duplicated faces)
  dist_stride = np.zeros(distri[comm.rank+1]-distri[comm.rank], np.int32)
  dist_stride[PTB.getBlockGnumCopy()[mask] - distri[comm.rank] - 1] = 1
  dist_data = dist_data[mask]

  out_stride, out_data = EP.block_to_part_strided(dist_stride, dist_data, distri, [g-1 for g in gnum_list], comm, legacy=False)

  # Now treat partitions to swap faces 
  for izone, part_zone in enumerate(part_zones):
    cur_zone_glob = zone_proc_offset + izone

    # Only faces having out_stride == 1 should be considered, and in addition
    # we need to retrieve their local num in all face (because we extracted bnd faces)
    ext_faces_left_pe = bnd_list[izone]
    todeal = ext_faces_left_pe[out_stride[izone]==1]

    ngon_node  = PT.Zone.NGonNode(part_zone)
    pe = PT.get_child_from_name(ngon_node, 'ParentElements')[1]
    eso = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
    ec = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]

    has_nface = PT.Zone.has_nface_elements(part_zone)
    if has_nface:
      nface_node  = PT.Zone.NFaceNode(part_zone)
      nface_ec  = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]
      nface_eso = PT.get_child_from_name(nface_node, 'ElementStartOffset')[1]

    for iface, master in zip(todeal, out_data[izone]):
      if master != cur_zone_glob:
        pe[iface, 1] = pe[iface, 0] # Swap PE
        pe[iface, 0] = 0
        
        # Swap face_vtx connectivity
        ec_view = ec[eso[iface]:eso[iface+1]]
        ec_view[:] = ec_view[::-1]

        # Change sign in cell_face
        if has_nface:
          parent_cell = pe[iface, 1] - PT.Element.Range(nface_node)[0]
          nface_ec_view = nface_ec[nface_eso[parent_cell]:nface_eso[parent_cell+1]]
          nface_ec_view[nface_ec_view == (iface+1)] *= -1
    
        