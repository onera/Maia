****************
Welcome to Maia!
****************

**Maia** is a C++ and Python library for parallel algorithms over CGNS trees: distributed IO, partitioning, data management and various kind of transformations (generation of faces, destructuration, extractions, ...).


Documentation summary
---------------------

:ref:`Quick start <quick_start>` is the perfect page for a first overview of Maia or to retrieve the environments to use on Onera's clusters.

:ref:`Introduction <intro>` details the extensions made to the CGNS standard in order to define parallel CGNS trees.

:ref:`User Manual <user_manual>` is the main part of this documentation. It describes most of the high level APIs provided by Maia.

:ref:`Developer Manual <dev_manual>` (under construction) provides more details on some algorithms and can be consulted if you want to contribute to Maia.

:ref:`The pytree module <pytree_module>` describes how to manipulate python CGNS trees. There is no parallel functionalities is this module, which may become an independent project one day.

!!!!

.. image:: ./_static/logo_maia.svg
  :width: 40%
  :align: center

.. rst-class:: center

Maia is an open source software developed at `ONERA <https://www.onera.fr>`_.
Associated source repository and issue tracking are hosted on `Gitlab <https://gitlab.onera.net/numerics/mesh/maia>`_.

.. toctree::
  :hidden:
  :maxdepth: 1
  :caption: Reference

  quick_start
  installation
  introduction/introduction
  user_manual/user_manual
  developer_manual/developer_manual
  maia_pytree/pytree_module

.. toctree::
  :hidden:
  :maxdepth: 1
  :caption: Appendix

  releases/release_notes
  related_projects
  license
