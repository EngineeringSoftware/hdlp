#!/usr/bin/env python

import os
from datetime import datetime

### This file contains common variables/functions shared by some
### training/testing scripts.

DATADIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vhdl")
CONFIGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")
MODELDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
LOGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
TESTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tests")
SAVEDIR = datetime.now().strftime("%Y%m%d-%H%M%S")
MAX_PA = 5
