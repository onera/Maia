Configuration
=============


Logging
-------

Maia provides informations to the user through the loggers summarized
in the following table:

+--------------+-----------------------+---------------------------+
| Logger       | Purpose               | Default printer           |
+==============+=======================+===========================+
| maia         | Light general info    | mpi_rank_0_stdout_printer |
+--------------+-----------------------+---------------------------+
| maia-warnings| Warnings              | mpi_rank_0_stdout_printer |
+--------------+-----------------------+---------------------------+
| maia-errors  | Errors                | mpi_rank_0_stdout_printer |
+--------------+-----------------------+---------------------------+
| maia-stats   | More detailed timings | No output                 |
|              | and memory usage      |                           |
+--------------+-----------------------+---------------------------+

The easiest way to change this default configuration is to 
set the environment variable ``LOGGING_CONF_FILE`` to provide a file
(e.g. logging.conf) looking like this:

.. code-block:: Text

  maia          : mpi_stdout_printer        # All ranks output to sdtout
  maia-warnings :                           # No output
  maia-errors   : mpi_rank_0_stderr_printer # Rank 0 output to stderr
  maia-stats    : file_printer("stats.log") # All ranks output in the file

See :ref:`logging` for a full description of available 
printers or for instructions on how to create your own printers.
This page also explain how to dynamically change printers 
directly in your Python application.

Exception handling
------------------

Maia automatically override the `sys.excepthook
<https://docs.python.org/3/library/sys.html#sys.excepthook>`_
function to call ``MPI_Abort`` when an uncatched exception occurs.
This allow us to terminate the MPI execution and to avoid some deadlocks
if an exception is raised by a single process.
Note that this call force the global ``COMM_WORLD`` communicator to abort which
can have undesired effects if sub communicators are used.

This behaviour can be disabled with a call to
``maia.excepthook.disable_mpi_excepthook()``.

