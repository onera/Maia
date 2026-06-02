File management
===============

Maia supports HDF5/CGNS file reading and writing,
see `related documention <https://cgns.org/standard/hdf5.html>`_.

The IO functions are provided by the ``maia.io`` module. The low level hdf
calls are performed by the python module `h5py <https://www.h5py.org/>`_.
Note that both the hdf5 library and the h5py module must have
been installed on your computer with parallel support. 

All the IO functions accept ``str`` or ``Path`` objects for ``filename`` argument.
For write operations, intermediate directories are created if they do not already exist.

.. rubric:: A note about 32 characters

CGNS is infamous for its names being limited to 32 characters. However, since the Python/CGNS mapping
does not *technically* enforce this limitation, Maia allows trees having names longer than 32 chars
in IO functions. This is achived throught the following hack:

- when **writing** files, nodes having a name longer than 32 chars are shortened, the original name beeing stored
  in a special ``Descriptor_t`` child called 'FullNameLongerThan32CharactersLi'.
- when **reading** files, nodes having the special 'FullNameLongerThan32CharactersLi' child have their name updated
  with the registered value. The descriptor node is then removed.

The same apply to links: users should provide original (possibly long) paths to write functions,
and automatically get original paths when calling :func:`~maia.io.read_links`.

This ensure a transparent processus for the user, but keep in mind that
cross tree references are not updated: for exemple, ``FamilyName_t`` values are not updated, even
if the matching family is shortened. This can confuse applications not using Maia to read files.
In addition, if such application remove the special Descriptor children nodes from file, original
names will be lost.

.. hint:: Users can see the shortened ↔ original names mapping of a hdf file using ``maia_show_full_names`` utility.

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

