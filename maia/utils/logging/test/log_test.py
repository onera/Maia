from maia.utils.logging import add_logger, log, add_printer_to_logger, turn_on, turn_off
from maia.utils.logging import size_to_str, bsize_to_str
import sys
import gc

class printer_for_unit_tests:
  def __init__(self):
    self.buffer = ''

  def log(self, msg):
    self.buffer += msg
    
def test_log():
  add_logger('logger_for_unit_tests')

  # at this point, no printer is associated to the logger
  log('logger_for_unit_tests', 'msg 0') # does nothing 

  # associate a printer
  p0 = printer_for_unit_tests()
  add_printer_to_logger('logger_for_unit_tests', p0)

  log('logger_for_unit_tests', 'msg 1')
  assert p0.buffer == 'msg 1\n'

  # turn off the logger
  turn_off('logger_for_unit_tests')
  log('logger_for_unit_tests', 'msg 2')
  assert p0.buffer == 'msg 1\n'

  # turn on the logger
  turn_on('logger_for_unit_tests')
  log('logger_for_unit_tests', 'msg 3')
  assert p0.buffer == 'msg 1\nmsg 3\n'

  # associate a second printer
  p1 = printer_for_unit_tests()
  add_printer_to_logger('logger_for_unit_tests', p1)

  log('logger_for_unit_tests', 'msg 4')
  assert p0.buffer == 'msg 1\nmsg 3\nmsg 4\n'
  assert p1.buffer == 'msg 4\n'

  # replace the logger
  add_logger('logger_for_unit_tests', replace=True)
  # logger_for_unit_tests has been reinitialized and is no longer associated to p0

  p2 = printer_for_unit_tests()
  add_printer_to_logger('logger_for_unit_tests', p2)

  log('logger_for_unit_tests', 'msg 5')
  assert p0.buffer == 'msg 1\nmsg 3\nmsg 4\n' # no change
  assert p2.buffer == 'msg 5\n'


def test_size_display():
    assert  size_to_str(4280938) == "4.3 M"
    assert bsize_to_str(4280938) == "4.1 MiB"
