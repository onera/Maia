from cmaia.utils.logging import *
def info(msg):
  log("maia", msg)
def stat(msg):
  log("maia-stats", msg)
def warning(msg):
  log("maia-warnings", "Warning: "+msg)
def error(msg):
  log("maia-errors", "Error: "+msg)
