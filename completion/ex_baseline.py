# This script runs rule-based baseline model.

from typing import *
import os
import nltk
import argparse
import numpy as np
from seutil import IOUtils


SAVE_DIR = "./tests"
DATA_DIR = "./data/vhdl"

parser=argparse.ArgumentParser()
parser.add_argument('-modelname', '--modelname', default="Baseline", required=False,
                    type=str, help='Save directory name')
parser.add_argument('-ref_modelname', '--ref_modelname', required=True,
                    type=str, help='Model name which we use dataset in')


def get_targets(ref_modelname, data_mode):
    filename = f"{DATA_DIR}/{ref_modelname}/tgt.{data_mode}.txt"
    target_list: List[List[str]] = [x.strip() for x in IOUtils.load(filename, IOUtils.Format.txt).splitlines()]
    return target_list


def get_baseline_preds(ref_modelname, data_mode):
    filename = f"{DATA_DIR}/{ref_modelname}/src.prevassign.{data_mode}.txt"
    preds: List[str] = [x.strip() for x in IOUtils.load(filename, IOUtils.Format.txt).splitlines()]
    preds_list = list()
    for pred in preds:
        if pred=="<empty>":
            preds_list.append(pred)
        else:
            preds_list.append(pred.split("<= ")[1].split(";")[0].strip())
    return preds_list

# From: measure_bleuacc
def get_accuracy(target: List[str], pred: List[str]) -> float:
    total = max(len(target), len(pred))
    #total = len(target)
    correct = 0
    for j in range(min(len(target), len(pred))):
        if target[j] == pred[j]:  correct += 1
    # end for
    return correct / total


# From: measure_bleuacc
def get_bleu(target: List[str], pred: List[str]) -> float:
    return nltk.translate.bleu_score.sentence_bleu([target], pred, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)


# From: measure_bleuacc
def get_exact_match_accuracy(target: List[str], pred: List[str]) -> float:
    tgt_str = " ".join(target)
    pred_str = " ".join(pred)
    if tgt_str==pred_str:
        return 1.0
    return 0.0

def main_test(modelname, ref_modelname):
    bleus, accs, exact_accs = [],[],[]
    target_list = get_targets(ref_modelname, "test")
    preds_list = get_baseline_preds(ref_modelname, "test")
    for pred, target in zip(preds_list, target_list):
        pred_split = [t for t in pred.split() if t!='']
        target_split = [t for t in target.split() if t!='']
        bleu = get_bleu(target=target_split, pred=pred_split)
        acc = get_accuracy(target=target_split, pred=pred_split)
        exact_acc = get_exact_match_accuracy(target=target_split, pred=pred_split)
        bleus.append(bleu)
        accs.append(acc)
        exact_accs.append(exact_acc)

    avg_bleu = np.mean(bleus)
    avg_acc = np.mean(accs)
    avg_exact_acc = np.mean(exact_accs)
    print(f"Average BLEU: {avg_bleu:.3f}, average accuracy: {avg_acc:.3f}, average exact match accuracy: {avg_exact_acc:.3f}")
    results_file = os.path.join(SAVE_DIR, modelname,"testlog.assignments.baseline.log")
    results = {
        "bleu-AVG": avg_bleu,
        "acc-AVG": avg_acc,
        "exact-acc-AVG": avg_exact_acc,
        "bleu": bleus,
        "acc": accs,
        "exact-acc": exact_accs,
    }
    IOUtils.dump(results_file, results, IOUtils.Format.jsonNoSort)

    IOUtils.dump(os.path.join(SAVE_DIR, modelname, "pred.assignments.baseline.log"), "".join([pred.strip()+"\n" for pred in preds_list]), IOUtils.Format.txt)
    return


def main_val(modelname, ref_modelname):
    bleus, accs, exact_accs = [],[],[]
    target_list = get_targets(ref_modelname, "val")
    preds_list = get_baseline_preds(ref_modelname, "val")
    for pred, target in zip(preds_list, target_list):
        pred_split = [t for t in pred.split(" ") if t!='']
        target_split = [t for t in target.split(" ") if t!='']
        bleu = get_bleu(target=target_split, pred=pred_split)
        acc = get_accuracy(target=target_split, pred=pred_split)
        exact_acc = get_exact_match_accuracy(target=target_split, pred=pred_split)
        bleus.append(bleu)
        accs.append(acc)
        exact_accs.append(exact_acc)

    avg_bleu = np.mean(bleus)
    avg_acc = np.mean(accs)
    avg_exact_acc = np.mean(exact_accs)
    print(f"Average BLEU: {avg_bleu:.3f}, average accuracy: {avg_acc:.3f}, average exact match accuracy: {avg_exact_acc:.3f}")
    results_file = os.path.join(SAVE_DIR, modelname,"testlog.val.assignments.baseline.log")
    results = {
        "bleu-AVG": avg_bleu,
        "acc-AVG": avg_acc,
        "exact-acc-AVG": avg_exact_acc,
        "bleu": bleus,
        "acc": accs,
        "exact-acc": exact_accs,
    }
    IOUtils.dump(results_file, results, IOUtils.Format.jsonNoSort)

    IOUtils.dump(os.path.join(SAVE_DIR, modelname, "pred.val.assignments.baseline.log"), "".join([pred.strip()+"\n" for pred in preds_list]), IOUtils.Format.txt)
    return


def main():
    args = parser.parse_args()
    modelname = args.modelname
    ref_modelname = args.ref_modelname
    assert os.path.exists(f"{DATA_DIR}/{ref_modelname}")
    if not os.path.exists(os.path.join(SAVE_DIR, modelname)):
        os.mkdir(os.path.join(SAVE_DIR, modelname))
    main_test(modelname, ref_modelname)
    main_val(modelname, ref_modelname)
    return
    

if __name__=="__main__":
    main()
