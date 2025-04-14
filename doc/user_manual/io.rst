File management
===============

Maia supports HDF5/CGNS file reading and writing,
see `related documention <https://cgns.github.io/CGNS_docs_current/hdf5/index.html>`_.

The IO functions are provided by the ``maia.io`` module. The low level hdf
calls are performed by the python module `h5py <https://www.h5py.org/>`_.
Note that both the hdf5 library and the h5py module must have
been installed on your computer with parallel support. 

.. _user_man_dist_io:

Distributed IO
--------------

Distributed IO is the privileged way to deal with CGNS files within your maia workflows.
Files are loaded as distributed trees, and, inversely, distributed trees can be written
into a single CGNS file.

High level IO operations can be performed with the two following functions, which read
or write all data they found :

.. autofunction:: maia.io.file_to_dist_tree
.. autofunction:: maia.io.dist_tree_to_file


The example below shows how to uses these high level functions:

.. literalinclude:: snippets/test_io.py
  :start-after: #file_to_dist_tree_full@start
  :end-before: #file_to_dist_tree_full@end
  :dedent: 2

Finer control of what is written or loaded can be achieved with the following steps:

- For a **write** operation, the easiest way to write only some nodes in
  the file is to remove the unwanted nodes from the distributed tree.
- For a **read** operation, the load has to be divided into the following steps:

  - Loading a size_tree: this tree has only the shape of the distributed data and
    not the data itself.
  - Removing unwanted nodes in the size tree;
  - Fill the filtered tree from the file.

The example below illustrate how to filter the written or loaded nodes:

.. literalinclude:: snippets/test_io.py
  :start-after: #file_to_dist_tree_filter@start
  :end-before: #file_to_dist_tree_filter@end
  :dedent: 2

Partitioned IO
--------------

In some cases, it may be useful to write or read a partitioned tree (keeping the
partitioned zones separated). This can be achieved using the following functions:

.. autofunction:: maia.io.part_tree_to_file
.. autofunction:: maia.io.file_to_part_tree

.. _user_man_raw_io:

Raw IO
------

For debug purpose, trees can also be read or written independently in a
sequential manner. Be aware that information added by maia such as Distribution
or GlobalNumbering nodes will not be removed.

.. autofunction:: maia.io.read_tree
.. autofunction:: maia.io.read_links
.. autofunction:: maia.io.write_tree
.. autofunction:: maia.io.write_trees

Non-existing folders
--------------------

If your launch scripts don't create directly the folder in which the output
files have to be written, maia creates it:

.. autofunction:: maia.io.utils.create_parent_folder