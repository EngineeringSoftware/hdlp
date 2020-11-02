
# This script is main function for invoking scripts and functions in naturalness/python/hdlp/

from typing import *

from distutils.version import StrictVersion
from pathlib import Path
import pkg_resources
import random
import sys
import time

from seutil import LoggingUtils, CliUtils, IOUtils

from hdlp.Macros import Macros
from hdlp.Utils import Utils


# Check seutil version
EXPECTED_SEUTIL_VERSION = "0.4.9"
if StrictVersion(pkg_resources.get_distribution("seutil").version) < StrictVersion(EXPECTED_SEUTIL_VERSION):
    print(f"seutil version does not meet expectation! Expected version: {EXPECTED_SEUTIL_VERSION}, current installed version: {pkg_resources.get_distribution('seutil').version}", file=sys.stderr)
    print(f"Hint: either upgrade seutil, or modify the expected version (after confirming that the version will work)", file=sys.stderr)
    sys.exit(-1)
# end if


logging_file = "python.log"
LoggingUtils.setup(filename=str(logging_file))

logger = LoggingUtils.get_logger(__name__)


def split_dataset(**options):
    from hdlp.DatasetSplitter import DatasetSplitter

    assignments_path = Path(options.get("assignments-path", Macros.data_dir / "vhdl" / "ALL" / "assignments.json"))
    output_dir = Path(options.get("output-dir", Macros.completion_dir / "data" / "vhdl" / "intermediate"))
    cross_project = "cross-project" in options
    cross_file = "cross-file" in options
    always_end = "always-end" in options
    seed = options.get("random-seed", 27)
    use_new_sub_tokenizer = "use-new-sub-tokenizer" in options

    if cross_project:
        DatasetSplitter.split_dataset_cross_project(assignments_path, output_dir, seed, use_new_sub_tokenizer)
    elif cross_file:
        DatasetSplitter.split_dataset_cross_file(assignments_path, output_dir, seed, use_new_sub_tokenizer)
    elif always_end:
        DatasetSplitter.split_dataset_always_end(assignments_path, output_dir, seed, use_new_sub_tokenizer)
    else:
        DatasetSplitter.split_dataset(assignments_path, output_dir, seed, use_new_sub_tokenizer)
    # end if
    return


# ==========
# Main

def normalize_options(opts: dict) -> dict:
    if "log-path" in opts:
        logger.info(f"Switching to log file {opts['log-path']}")
        LoggingUtils.setup(filename=opts['log-path'])
    # end if
    if "random-seed" in opts:
        random.seed(opts["random-seed"])
    else:
        seed = time.time_ns()
        random.seed(seed)
        logger.info(f"Random seed is {seed}")
    # end if
    return opts

if __name__ == "__main__":
    CliUtils.main(sys.argv[1:], globals(), normalize_options)
