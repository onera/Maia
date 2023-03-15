Logging module
==============


Vocabulary
----------


Logger
^^^^^^

A **logger** is an object where an application or a library can log to.

A logger is refered to by a name. If we want to log a string to logger ``my_logger``, we will do it like so:

.. code-block:: python

  from cmaia.utils.logging import log

  log('my_logger', 'my message')

Typically, an application will use several loggers to log messages by themes. By convention, the name of a logger in the application ``my_app`` should begin with ``my_app``. For instance, loggers can be named: ``my_app``, ``my_app-stats``, ``my_app-errors``...

Loggers are both

- developper-oriented, that is, when a developper wants to output a message, he should select a logger (existing or not) encompassing the theme of the message he wants to log,
- user-oriented, because a user can choose what logger he wants to listen to.

Printer
^^^^^^^

By itself, a logger does not do anything with the messages it receives. For that, we need to attach **printers** to a logger that will handle its messages.

For instance, we can attach a printer that will output the message to the console:

.. code-block:: python

  from cmaia.utils.logging import add_stdout_printer
  add_printer_to_logger('my_logger','stdout_printer')

Printers are user-oriented: it is the user of the application who decides what he wants to do with the message that each logger is receiving, by attaching none or several printers to a logger.


Logger configuration
--------------------

Loggers are associated to default printers. While they can be configured anytime in the Python scripts, most of the time, reading a configuration file at the start of the program is enought. The program will try to read a configuration file if the environment variable ``LOGGING_CONF_FILE`` is set. A logging configuration file looks like this:


.. code-block:: Text

  my_app : mpi_stdout_printer
  my_app-my_theme : mpi_file_printer('my_theme.log')

For developpers, a logging file ``logging.conf`` with loggers and default printers is put in the ``build/`` folder, and ``LOGGING_CONF_FILE`` is set accordingly.

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

Create your own printer
"""""""""""""""""""""""

Any Python type can be used as a printer as long as it provides a ``log`` method that accepts a string argument.

.. code-block:: python

  from cmaia.utils.logging import add_stdout_printer

  class my_printer:
    def log(self, msg):
      print(msg)

  add_printer_to_logger('my_logger',my_printer())

Note
""""

MPI-aware printers use ``MPI.COMM_WORLD`` rather than a local communicator. While the latter would be more correct, it would imply local instanciation of printers, at the time or after a new communicator is created. While this is certainly doable, up until now we feel that it does not worth the burden. But if the need arises, they can still be added.

Miscellaneous
-------------

The logger mechanism is implemented in C++, which mean it is available in compiled code:

.. code-block:: C++

  #include "std_e/logging/log.hpp"

  log("my_logger", "my message");

Loggers are available to pure C/C++ programs (in particular, test programs that do not use the Python interpreter).

The C++ API is essentially the same.


Maia specifics
--------------

Maia provides 4 convenience functions that use Maia loggers

.. code-block:: python

  from maia.utils import logging as mlog
  mlog.info('info msg') # uses the 'maia' logger
  mlog.info('stat msg') # uses the 'maia-stats' logger
  mlog.info('warn msg') # uses the 'maia-warnings' logger
  mlog.info('error msg') # uses the 'maia-errors' logger
