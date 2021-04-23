#!/usr/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD

from maia.cgns_io import cgns_io_tree as IOT
from maia.connectivity import generate_ngon_from_std_elements as FTH
from maia.transform import ngon_new_to_old,split_boundary_subzones_according_to_bcs
from maia.sids.shorten_names import shorten_field_names,shorten_names
import Converter.PyTree as C
import Converter.Internal as I


prefix = "/scratchm/bberthou/cases/CODA_tests/M6_Ilias"
input_file = prefix+"/solution_beg_1.cgns"

t = C.convertFile2PyTree(input_file)
shorten_field_names(t)
C.convertPyTree2File(t,prefix+"/solution_beg_1c.cgns")

dist_tree = IOT.file_to_dist_tree(prefix+"/solution_beg_1c.cgns", comm, distribution_policy='uniform')
I.printTree(dist_tree)

for pr in I.getNodesFromName(dist_tree,"PointList"):
    pr[1] += -279+1 # to offset at 1
shorten_field_names(dist_tree,quiet=True)
C.convertPyTree2File(dist_tree,prefix+"/solution_beg_1b.cgns")

split_boundary_subzones_according_to_bcs(dist_tree)
shorten_names(dist_tree,quiet=True)

FTH.generate_ngon_from_std_elements(dist_tree, comm)

ngon_new_to_old(dist_tree)
C.convertPyTree2File(dist_tree,prefix+"/solution_ngon.cgns")
