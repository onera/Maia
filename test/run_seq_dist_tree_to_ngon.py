#!/usr/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD

from maia.cgns_io import cgns_io_tree as IOT
from maia.connectivity import generate_ngon_from_std_elements as FTH
from maia.transform import ngon_new_to_old
import Converter.PyTree as C

prefix = "/scratchm/bberthou/travail/git_all_projects/external/fs_cgns_adapter/examples/M6_wing_MPI"
input_file = prefix+"/data/out/M6_dist.cgns"
#input_file = "/scratchm/bberthou/cases/CODA_tests/cube/data/in/cube_4.cgns"
#input_file = '/home/bmaugars/dev/dev-Tools/maia/unit_tests_case/CUBES_POUR_BRUNO/cube8.cgns'
dist_tree = IOT.file_to_dist_tree(input_file, comm, distribution_policy='uniform')

FTH.generate_ngon_from_std_elements(dist_tree, comm)

ngon_new_to_old(dist_tree)
C.convertPyTree2File(dist_tree,prefix+"/data/out/M6_dist_ngon.cgns")
