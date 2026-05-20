import pytest
from maia.conftest import is_master_process, rewrite_junit_report

def pytest_configure(config):
  if is_master_process(config):
    config.option.xmlpath = "junit_maia_doc.xml"
    
def pytest_sessionfinish(session, exitstatus):
  if is_master_process(session.config):
    rewrite_junit_report(session.config.option.xmlpath, 'Doc snippet')