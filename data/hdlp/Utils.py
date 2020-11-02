
# This script is for defining utility functions used in scripts in naturalness/python/hdlp/

from typing import *

import importlib
import importlib.util
import json
import numpy as np
import os
from pathlib import Path
import re
import sys
import time

from seutil import BashUtils, LoggingUtils


class Utils:
    """
    Some utilities that doesn't tie to a specific other file. TODO: move them into seutil at some point.
    """
    logger = LoggingUtils.get_logger(__name__)

    @classmethod
    def get_option_as_boolean(cls, options, opt, default=False) -> bool:
        if opt not in options:
            return default
        else:
            # Due to limitations of CliUtils...
            return str(options.get(opt, "false")).lower() != "false"
        # end if

    @classmethod
    def get_option_as_list(cls, options, opt, default=None) -> list:
        if opt not in options:
            return copy.deepcopy(default)
        else:
            l = options[opt]
            if not isinstance(l, list):  l = [l]
            return l
        # end if

    SUMMARIES_FUNCS: Dict[str, Callable[[Union[list, np.ndarray]], Union[int, float]]] = {
        "AVG": lambda l: np.mean(l) if len(l) > 0 else np.NaN,
        "SUM": lambda l: sum(l) if len(l) > 0 else np.NaN,
        "MAX": lambda l: max(l) if len(l) > 0 else np.NaN,
        "MIN": lambda l: min(l) if len(l) > 0 else np.NaN,
        "MEDIAN": lambda l: np.median(l) if len(l) > 0 and np.NaN not in l else np.NaN,
        "STDEV": lambda l: np.std(l) if len(l) > 0 else np.NaN,
        "CNT": lambda l: len(l),
    }

    SUMMARIES_PRESERVE_INT: Dict[str, bool] = {
        "AVG": False,
        "SUM": True,
        "MAX": True,
        "MIN": True,
        "MEDIAN": False,
        "STDEV": False,
        "CNT": True,
    }

    RE_GITHUB_URL = re.compile(r"https://github\.com/(?P<user>[^/]*)/(?P<repo>.*)\.git")

    @classmethod
    def lod_to_dol(cls, list_of_dict: List[dict]) -> Dict[Any, List]:
        """
        Converts a list of dict to a dict of list.
        """
        keys = set.union(*[set(d.keys()) for d in list_of_dict])
        return {k: [d.get(k) for d in list_of_dict] for k in keys}

    @classmethod
    def counter_most_common_to_pretty_yaml(cls, most_common: List[Tuple[Any, int]]) -> str:
        s = "[\n"
        for x, c in most_common:
            s += f"[{json.dumps(x)}, {c}],\n"
        # end for
        s += "]\n"
        return s

    @classmethod
    def modify_and_import(cls, module_name, package, modification_func):
        spec = importlib.util.find_spec(module_name, package)
        source = spec.loader.get_source(module_name)
        new_source = modification_func(source)
        module = importlib.util.module_from_spec(spec)
        codeobj = compile(new_source, module.__spec__.origin, 'exec')
        exec(codeobj, module.__dict__)
        sys.modules[module_name] = module
        return module
