from mpi4py import MPI
import logging as LOG

# ------------------------------------------------------------------------
# > Initilise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------------------------------
fmt = '%(levelname)s:%(message)s '.format(rank, size)
LOG.basicConfig(filename = '{0}.{1}.log'.format('maia_workflow_log', rank),
                level    = 10,
                format   = fmt,
                filemode = 'w')
# ---------------------------------------------------------

import Converter.PyTree   as C
import Converter.Internal as I
import Geom.PyTree        as D
import numpy              as NPY
import sys

from maia.cgns_io            import load_collective_size_tree       as LST
from maia.cgns_io            import cgns_io_tree                    as IOT
from maia.cgns_io            import save_part_tree                  as SPT
from maia.cgns_io.hdf_filter import tree                            as HTF
from maia.connectivity       import generate_ngon_from_std_elements as FTH
from maia.partitioning       import part                            as PPA
from maia.partitioning.load_balancing import setup_partition_weights as DBA
import maia.distribution.distribution_tree                          as MDI

from   Converter import cgnskeywords as CGK

from Pypdm import Pypdm
from maia import npy_pdm_gnum_dtype as pdm_dtype

# ------------------------------------------------------------------------
# > Pick a file
inputfile    = '/home/bmaugars/dev/dev-Tools/etc/test/pypart/data/CaseU_C1_Cube.hdf'
inputfile    = '/home/bmaugars/dev/dev-Tools/etc/test/pypart/data/CaseU_C1_Cube_BND_NGON2.hdf'

# ------------------------------------------------------------------------
def get_coords_and_gnum(t):
  """
  """
  for Zone in I.getZones(t):
    CX = I.getNodeFromName2(Zone, "CoordinateX")[1]
    CY = I.getNodeFromName2(Zone, "CoordinateY")[1]
    CZ = I.getNodeFromName2(Zone, "CoordinateZ")[1]
    coords = NPY.concatenate([CX, CY, CZ])
    coords = coords.reshape( (3, CX.shape[0]))
    coords = coords.transpose()
    coords = coords.reshape( 3*CX.shape[0], order='C')
    print(coords)
    g_nums = NPY.linspace(1, CX.shape[0], CX.shape[0], dtype=pdm_dtype)
    print(g_nums)
  return coords, g_nums

# ------------------------------------------------------------------------
def get_zone_info(t):
  """
  """
  for Zone in I.getZones(t):
    CX = I.getNodeFromName2(Zone, "CoordinateX")[1]
    CY = I.getNodeFromName2(Zone, "CoordinateY")[1]
    CZ = I.getNodeFromName2(Zone, "CoordinateZ")[1]
    coords = NPY.concatenate([CX, CY, CZ])
    coords = coords.reshape( (3, CX.shape[0]))
    coords = coords.transpose()
    coords = coords.reshape( 3*CX.shape[0], order='C')
    print(coords)

    cell_face_idx = I.getNodeFromName2(Zone, "np_cell_face_idx")[1]
    cell_face     = I.getNodeFromName2(Zone, "np_cell_face")[1]
    cell_ln_to_gn = I.getNodeFromName2(Zone, "np_cell_ln_to_gn")[1]
    face_vtx_idx  = I.getNodeFromName2(Zone, "np_face_vtx_idx")[1]
    face_vtx      = I.getNodeFromName2(Zone, "np_face_vtx")[1]
    face_ln_to_gn = I.getNodeFromName2(Zone, "np_face_ln_to_gn")[1]
    vtx_ln_to_gn  = I.getNodeFromName2(Zone, "np_vtx_ln_to_gn")[1]
  return cell_face_idx, cell_face, cell_ln_to_gn, face_vtx_idx, face_vtx, face_ln_to_gn, coords, vtx_ln_to_gn

# ------------------------------------------------------------------------
# > Load only the list of zone and sizes ...
dist_tree = LST.load_collective_size_tree(inputfile, comm)

MDI.add_distribution_info(dist_tree, comm, distribution_policy='uniform')

hdf_filter = dict()
HTF.create_tree_hdf_filter(dist_tree, hdf_filter)

# skip_type_ancestors = ["Zone_t/FlowSolution_t/"]
# skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun"], ["ZoneSubRegion_t", "VelocityY"]]
# skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun", "*"], ["Zone_t", "ZoneSubRegion_t", "VelocityY"]]
skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun", "Momentum*"],
                       ["Zone_t", "ZoneSubRegion_t", "Velocity*"]]
# hdf_filter_wo_fs = HTF.filtering_filter(dist_tree, hdf_filter, skip_type_ancestors, skip=True)
# # IOT.load_tree_from_filter(inputfile, dist_tree, comm, hdf_filter)

# for key, val in hdf_filter_wo_fs.items():
#   print(key, val)
# IOT.load_tree_from_filter(inputfile, dist_tree, comm, hdf_filter_wo_fs)
IOT.load_tree_from_filter(inputfile, dist_tree, comm, hdf_filter)

dzone_to_weighted_parts = DBA.npart_per_zone(dist_tree, comm, 1)

# print(dzone_to_weighted_parts)

dloading_procs = dict()
for zone in I.getZones(dist_tree):
  dloading_procs[zone[0]] = list(range(comm.Get_size()))
# print(dloading_procs)

part_tree = PPA.partitioning(dist_tree, comm, zone_to_parts=dzone_to_weighted_parts, dump_pdm_output=True)

for zone in I.getZones(part_tree):
  fs_n = I.newFlowSolution(name="FlowSolution#EndOfRun", gridLocation='Vertex', parent=zone)
  vtx_gi_n = I.getNodeFromName(zone, "np_vtx_ghost_information")
  I.newDataArray("GhostInfo", vtx_gi_n[1], parent=fs_n)


SPT.save_part_tree(part_tree, 'part_tree', comm)

# > Create a line
line_cloud = D.line((0,0,0), (2,2,2), 5)
# line_cloud = D.line((-1,0,0), (2,0,0), 5)
C.convertPyTree2File(line_cloud, 'out.cgns')
coords, gnum = get_coords_and_gnum(line_cloud)


# > Call at ParaDiGM
ml = Pypdm.MeshLocation(1, 1, comm)
ml.n_part_cloud_set(0, 1) # Pour l'instant 1 cloud et 1 partition
n_points = gnum.shape[0]
ml.cloud_set(0, 0, n_points, coords, gnum)
ml.mesh_global_data_set(1)

cell_face_idx, cell_face, cell_ln_to_gn, face_vtx_idx, face_vtx, face_ln_to_gn, coords_part, vtx_ln_to_gn = get_zone_info(part_tree)
n_cell = cell_ln_to_gn.shape[0]
n_face = face_ln_to_gn.shape[0]
n_vtx  = vtx_ln_to_gn .shape[0]
print("ooooooooooooooooooooooooooooooooooooooooo")
print("cell_face_idx", cell_face_idx)
print("cell_face    ", cell_face    )
print("cell_ln_to_gn", cell_ln_to_gn)
print("face_vtx_idx ", face_vtx_idx )
print("face_vtx     ", face_vtx     )
print("face_ln_to_gn", face_ln_to_gn)
print("vtx_ln_to_gn ", vtx_ln_to_gn )
print("ooooooooooooooooooooooooooooooooooooooooo")
# cell_face, cell_ln_to_gn, face_vtx_idx, face_vtx, face_ln_to_gn, coords_part, vtx_ln_to_gn)
ml.part_set(0, n_cell, cell_face_idx, cell_face, cell_ln_to_gn,
               n_face, face_vtx_idx, face_vtx, face_ln_to_gn,
               n_vtx, coords_part, vtx_ln_to_gn)
ml.compute()
results = ml.location_get(0, 0)
for key, val in results.items():
  print(key, val)

# print(results)
results_pts = ml.points_in_elt_get(0, 0)
# print(results_pts)
for key, val in results_pts.items():
  print(key, val)

# C.convertPyTree2File(part_tree, "part_tree_{0}.hdf".format(rank))
