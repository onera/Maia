import sys
import pytest
import pytest_parallel
from mpi4py import MPI

from maia.utils.parallel import excepthook

# We dont want the call to MPI_Abort to destroy the MPI environnemnt,
# so we replace Abort method by a dummy function storing the abort code
class DummyComm:
    def __init__(self, real_comm):
        self._real = real_comm
        self._aborted = None
    def __getattr__(self, attr):
        if attr == "Abort":
            return self.dummy_abort
        return getattr(self._real, attr)

    def dummy_abort(self, code):
        self._aborted = code


@pytest_parallel.mark.parallel(2)
def test_mpi_excepthook(monkeypatch, comm, capsys):

    dummy_comm = DummyComm(comm)

    # For this test, replace COMM_WORLD by the dummy communicator to avoid
    # the call to Abort method
    monkeypatch.setattr(MPI, "COMM_WORLD", dummy_comm)
    
    # We can not raise exception directly, because pytest will catch it
    # We call mpi_excepthook with an exception object instead
    if comm.rank == 1:
        excepthook.mpi_excepthook(ValueError, ValueError("error msg"), None)

    captured = capsys.readouterr().err

    if comm.rank == 0:
        assert captured == ''
        # Abort is not called on rank 0, but in true conditions MPI would have finalize
        assert dummy_comm._aborted == None
    elif comm.rank == 1:
        # Check that except hook added the abort line
        assert captured.startswith("Your application aborted because of an uncaught exception on rank 1:\n\n")
        # Check that exception is printed by sys_excepthook
        assert "ValueError" in captured
        assert "error msg" in captured
        # Check that MPI_Abort has been called
        assert dummy_comm._aborted == 1


def test_enable_mpi_excepthook():

    excepthook.enable_mpi_excepthook()
    assert sys.excepthook == excepthook.mpi_excepthook, "sys.excepthook must be set to mpi_excepthook after enabling."

def test_disable_mpi_excepthook():
    excepthook.enable_mpi_excepthook()
    excepthook.disable_mpi_excepthook()
    
    assert sys.excepthook == excepthook.sys_excepthook, "sys.excepthook should be reset to the original after disabling."

