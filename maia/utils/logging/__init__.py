from maia.typing import *
from cmaia.utils.logging import log, add_logger, turn_on, turn_off


def add_printer_to_logger(logger_name: str, printer: Union[str, Any]) -> None:
  from cmaia.utils.logging import _add_printer_obj_to_logger
  from cmaia.utils.logging import _add_printer_type_to_logger
  if isinstance(printer, str):
    _add_printer_type_to_logger(logger_name, printer)
  else:
    _add_printer_obj_to_logger(logger_name, printer)


def size_to_str(size: int) -> str:
  units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
  i = 0
  if size < 1000: #Corner case with no decimal
    return "{0}".format(size)
  fsize = float(size)
  while(fsize > 1000.):
      fsize /= 1000.
      i += 1
  return "{0:.1f}{1}".format(fsize, units[i])

def bsize_to_str(size: int) -> str:
  units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
  i = 0
  if size < 1000: #Corner case with no decimal
    return "{0}B".format(size)
  fsize = float(size)
  while (fsize > 1024.):
    fsize /= 1024.
    i += 1
  return f"{fsize:.1f}{units[i]}"

def info(msg: str) -> None:
  log("maia", msg)
def stat(msg: str) -> None:
  log("maia-stats", msg)
def debug(msg: str) -> None:
  log("maia-debug", msg)
def warning(msg: str) -> None:
  log("maia-warnings", "Warning: "+msg)
def error(msg: str) -> None:
  log("maia-errors", "Error: "+msg)
