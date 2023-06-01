import os
import sys
import pathlib
import logging

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
ARTIFACTS_PATH = os.path.join(PATH, 'artifacts')

LOG = logging.Logger('testing')
LOG.addHandler(logging.StreamHandler(sys.stdout))

