import sys
import pytest
import pytest_parallel
from mpi4py import MPI
from maia.utils.parallel.excepthook import (
    mpi_excepthook, 
    enable_mpi_excepthook, 
    disable_mpi_excepthook, 
    sys_excepthook)

def dummy_abort(code):
    raise RuntimeError(f"Abort called with code {code}")


class DummyComm:
    def __init__(self, real_comm):
        self._real = real_comm
    def __getattr__(self, attr):
        if attr == "Abort":
            return dummy_abort
        return getattr(self._real, attr)


def dummy_excepthook(etype, evalue, tb):
    dummy_excepthook.called = (etype, evalue, tb)
dummy_excepthook.called = None

@pytest.fixture(autouse=True)
def restore_excepthook(monkeypatch):

    original_hook = sys.excepthook
    yield
    monkeypatch.setattr(sys, "excepthook", original_hook)

@pytest_parallel.mark.parallel(1)
def test_mpi_excepthook(monkeypatch, comm, capsys):

    original_comm = comm
    dummy_comm = DummyComm(original_comm)

    monkeypatch.setattr(MPI, "COMM_WORLD", dummy_comm)

    monkeypatch.setattr(sys.modules['maia.utils.parallel.excepthook'], "sys_excepthook", dummy_excepthook)
    
    with pytest.raises(RuntimeError, match="Abort called with code 1"):
        mpi_excepthook(ValueError, ValueError("error"), None)
    
    assert dummy_excepthook.called is not None, "Dummy excepthook was not called."
    etype, evalue, tb = dummy_excepthook.called
    assert etype is ValueError
    assert isinstance(evalue, ValueError)
    assert str(evalue) == "error"
    
    captured = capsys.readouterr().err
    rank = dummy_comm.Get_rank()
    expected_message = f"Your application aborted because of an uncaught exception on rank {rank}:\n\n"
    assert expected_message in captured

def test_enable_mpi_excepthook(monkeypatch):

    original_hook = sys.excepthook
    enable_mpi_excepthook()
    assert sys.excepthook == mpi_excepthook, "sys.excepthook must be set to mpi_excepthook after enabling."
    monkeypatch.setattr(sys, "excepthook", original_hook)

def test_disable_mpi_excepthook(monkeypatch):
    original_hook = sys.excepthook
    enable_mpi_excepthook()
    disable_mpi_excepthook()
    
    assert sys.excepthook == sys_excepthook, "sys.excepthook should be reset to the original after disabling."

    monkeypatch.setattr(sys, "excepthook", original_hook)
