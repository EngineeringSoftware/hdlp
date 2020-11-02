# This script copies assignment prediction model test results into
# _results/test-results/{model_name}.json.

from typing import *

import argparse
import os
import glob

TESTDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tests")
RESULT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../slpproject/_results")
DST_DIR = os.path.join(RESULT_DIR, "test-results")

parser = argparse.ArgumentParser()
parser.add_argument('-which', '--which', required=True,
                    default=[],
                    type=str, nargs='+', help='model list used for copying result files.')
parser.add_argument('-suffix', '--suffix', required=True,
                    default='',
                    type=str, help='suffix for the results to be used in files/macros')


def check_models(args):
    for m in args.which:
        result_files = glob.glob(os.path.join(TESTDIR, m, "testlog.*.log"))
        assert len(result_files)>=1, f"Can't find result files for model {m}"
    return


def copy_results(args, model, mode):
    from shutil import copyfile
    if not os.path.exists(DST_DIR):
        os.mkdir(DST_DIR)
    src = glob.glob(os.path.join(TESTDIR, model, "testlog.assignments.*.log"))
    if mode == "val":
        src = glob.glob(os.path.join(TESTDIR, model, "testlog.val.assignments.*.log"))
    assert len(src) == 1
    suffix = args.suffix
    dst = os.path.join(DST_DIR, f"{model}{suffix}.json")
    copyfile(src[0], dst)
    return

    
def copy_model_results(args):
    for m in args.which:
        copy_results(args, m, "test")
    return


def main():
    args = parser.parse_args()
    check_models(args)
    copy_model_results(args)
    return


if __name__=="__main__":
    main()
