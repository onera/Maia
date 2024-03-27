.. _logging:

Log management
==============


Loggers
-------

A **logger** is a global object where an application or a library can log to.
It can be declared with

.. literalinclude:: snippets/test_logging.py
  :start-after: #add_logger@start
  :end-before: #add_logger@end
  :dedent: 2

or with the equivalent C++ code

.. code-block:: C++

  #include "std_e/logging/log.hpp"

  std_e::add_logger("my_logger");

A logger declared in Python is available in C++ and vice-versa: we do not need to create it twice.

.. note:: Loggers are available to pure C/C++ programs (in particular, test programs that do not use the Python interpreter).

It can then be referred to by its name. If we want to log a string to ``my_logger``, we will do it like so:

.. literalinclude:: snippets/test_logging.py
  :start-after: #log@start
  :end-before: #log@end
  :dedent: 2
  
.. code-block:: C++

  #include "std_e/logging/log.hpp"

  std_e::log("my_logger", "my message");

Typically, an application will use several loggers to log messages by themes. By convention, the name of a logger in the application ``my_app`` should begin with ``my_app``. For instance, loggers can be named: ``my_app``, ``my_app-stats``, ``my_app-errors``...

Loggers are both

- developper-oriented, that is, when a developper wants to output a message, he should select a logger (existing or not) encompassing the theme of the message he wants to log,
- user-oriented, because a user can choose what logger he wants to listen to.



Printers
--------

By itself, a logger does not do anything with the messages it receives. For that, we need to attach **printers** to a logger that will handle its messages.

For instance, we can attach a printer that will output the message to the console:

.. literalinclude:: snippets/test_logging.py
  :start-after: #add_printer@start
  :end-before: #add_printer@end
  :dedent: 2

Printers are user-oriented: it is the user of the application who decides what he wants to do with the message that each logger is receiving, by attaching none or several printers to a logger.

Available printers
^^^^^^^^^^^^^^^^^^

stdout_printer, stderr_printer
  output messages to the console (respectively stdout and stderr)

mpi_stdout_printer, mpi_stderr_printer
  output messages to the console, but prefix them by ``MPI.COMM_WORLD.Get_rank()``

mpi_rank_0_stdout_printer, mpi_rank_0_stderr_printer
  output messages to the console, if ``MPI.COMM_WORLD.Get_rank()==0``

file_printer('my_file.extension')
  output messages to file ``my_file.extension``

mpi_file_printer('my_file.extension')
  output messages to files ``my_file.{rk}.extension``, with ``rk = MPI.COMM_WORLD.Get_rank()``

.. note::
  MPI-aware printers use ``MPI.COMM_WORLD`` rather than a local communicator. While the latter would be more correct, it would imply local instanciation of printers, at the time or after a new communicator is created. While this is certainly doable, up until now we feel that it does not worth the burden. But if the need arises, they can still be added.

Create your own printer
^^^^^^^^^^^^^^^^^^^^^^^

Any Python type can be used as a printer as long as it provides a ``log`` method that accepts a string argument.

.. literalinclude:: snippets/test_logging.py
  :start-after: #create_printer@start
  :end-before: #create_printer@end
  :dedent: 2


Configuration file
------------------

Loggers are associated to default printers. While they can be configured anytime in the Python scripts, most of the time, reading a configuration file at the start of the program is enough. The program will try to read a configuration file if the environment variable ``LOGGING_CONF_FILE`` is set. A logging configuration file looks like this:


.. code-block:: Text

  my_app : mpi_stdout_printer
  my_app-my_theme : mpi_file_printer('my_theme.log')

For developpers, a logging file ``logging.conf`` with loggers and default printers is put in the ``build/`` folder, and ``LOGGING_CONF_FILE`` is set accordingly.

Maia specifics
--------------

Maia provides 4 convenience functions that use Maia loggers

.. code-block:: python

  from maia.utils import logging as mlog
  mlog.info('info msg') # uses the 'maia' logger
  mlog.stat('stat msg') # uses the 'maia-stats' logger
  mlog.warning('warn msg') # uses the 'maia-warnings' logger
  mlog.error('error msg') # uses the 'maia-errors' logger
