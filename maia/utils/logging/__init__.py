from cmaia.utils.logging import *

def size_to_str(size):
  units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
  i = 0
  while(size > 1000.):
      size /= 1000.
      i += 1
  return "{0:.1f} {1}".format(size, units[i])

def bsize_to_str(size):
  units = ["B", "kiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
  i = 0
  while(size > 1024.):
    size /= 1024.
    i += 1
  return f"{size:.1f} {units[i]}"

def info(msg):
  log("maia", msg)
def stat(msg):
  log("maia-stats", msg)
def warning(msg):
  log("maia-warnings", "Warning: "+msg)
def error(msg):
  log("maia-errors", "Error: "+msg)
