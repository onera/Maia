import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import Converter.Internal as I

from maia.cgns_io            import cgns_io_tree                    as IOT
from maia.connectivity       import generate_ngon_from_std_elements as FTH

from maia.utils              import test_utils as TU

@mark_mpi_test([1,4])
def test_elt_to_ngon(sub_comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  # > As simple as it looks
  FTH.generate_ngon_from_std_elements(dist_tree, sub_comm)

  # > Old elements are cleaned up
  assert len(I.getNodesFromType(dist_tree, 'Elements_t')) == 2
  assert I.getNodeFromName(dist_tree, 'NGonElements') is not None
  assert I.getNodeFromName(dist_tree, 'NFaceElements') is not None

  # TODO Add some checks, this functionnality seems to be broken now (593b1a5),
  # maybe check after PDM update

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    IOT.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing.hdf'), sub_comm)
