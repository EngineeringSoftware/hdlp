# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import os
from pathlib import Path

from seutil import LoggingUtils

this_dir = Path(os.path.dirname(os.path.realpath(__file__)))
log_file = this_dir.parent.parent / "experiments.log"
LoggingUtils.setup(filename=log_file)
logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    # log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    #
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(log_format)
    # logger.handlers = [console_handler]
    #
    # if log_file and log_file != '':
    #     file_handler = logging.FileHandler(log_file)
    #     file_handler.setLevel(log_file_level)
    #     file_handler.setFormatter(log_format)
    #     logger.addHandler(file_handler)

    return logger
