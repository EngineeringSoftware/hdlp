
# This script is for defining all macros used in scripts in naturalness/python/hdlp/

from typing import *

import os
from pathlib import Path

from seutil import IOUtils, BashUtils


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    proj_dir: Path = this_dir.parent.parent  # hdlp/

    data_dir: Path = proj_dir / "data"
    completion_dir: Path = proj_dir / "completion"

    all_lang: List[str] = ["vhdl", "verilog", "systemverilog", "java", "javasmall"]
    lang_2_extensions: Dict[str, str] = {
        "vhdl": "vhd",
        "verilog": "v",
        "systemverilog": "sv",
        "java": "java",
        "javasmall": "java",
    }

    MAX_PA_IN_MODEL = 5
