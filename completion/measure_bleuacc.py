# This script computes bleu, acc and exact match acc score on test set
# for all models.

from typing import *

import os
import argparse
import logging
import nltk
import numpy as np
from pathlib import Path

from seutil import IOUtils


parser = argparse.ArgumentParser(description='Arguments for measuring BLEU and ACC score from OpenNMT model')
parser.add_argument('-d', '--data_dir', help="input data directory", required=True)
parser.add_argument('-p', '--pred_file', help="input prediction txt file (pred)", required=True)
parser.add_argument('-t', '--tgt_file', help="target txt file (pred)", required=True)
parser.add_argument('-r', '--result_file', help="output txt file", required=True)
args = parser.parse_args()


def read_data_preds(data_dir: Path, pred_file: Path, target_file: Path) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    # objflag = False

    # Input lhs
    inputs: List[List[str]] = [x.split() for x in IOUtils.load(data_dir/"src.l.test.txt", IOUtils.Format.txt).splitlines()]

    # Pred rhs
    preds: List[List[str]] = [x.split() for x in IOUtils.load(pred_file, IOUtils.Format.txt).splitlines()]

    
    # Target rhs
    targets: List[List[str]] = [x.split() for x in IOUtils.load(data_dir/target_file, IOUtils.Format.txt).splitlines()]

    return inputs, preds, targets


def get_accuracy(target: List[str], pred: List[str]) -> float:
    total = max(len(target), len(pred))
    #total = len(target)
    correct = 0
    for j in range(min(len(target), len(pred))):
        if target[j] == pred[j]:  correct += 1
    # end for
    return correct / total


def get_bleu(target: List[str], pred: List[str]) -> float:
    return nltk.translate.bleu_score.sentence_bleu([target], pred, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)


def get_exact_match_accuracy(target: List[str], pred: List[str]) -> float:
    tgt_str = " ".join(target)
    pred_str = " ".join(pred)
    if tgt_str==pred_str:
        return 1.0
    return 0.0


def write_results(targets: List[List[str]], preds: List[List[str]], results_file: Path):
    bleu_scores = list()
    acc_scores = list()
    exact_acc_scores = list()
    report = ""
    for t, p in zip(targets, preds):
        bleu_score = get_bleu(t, p)
        acc_score = get_accuracy(t, p)
        exact_acc_score = get_exact_match_accuracy(t, p)
        bleu_scores.append(bleu_score)
        acc_scores.append(acc_score)
        exact_acc_scores.append(exact_acc_score)
        tgt = " ".join(t)
        pred = " ".join(p)
        report += f"TARGET : {tgt}\n"
        report += f"PREDICTION : {pred}\n"
        report += f"EXACT MATCH : YES\n\n" if exact_acc_score==1.0 else f"EXACT MATCH : NO\n\n"
    # end for
    avg_bleu = np.mean(bleu_scores)
    avg_acc = np.mean(acc_scores)
    avg_exact_acc = np.mean(exact_acc_scores)

    print(f"Average BLEU: {avg_bleu:.3f}, average accuracy: {avg_acc:.3f}, average exact match accuracy: {avg_exact_acc:.3f}")

    results = {
        "bleu-AVG": avg_bleu,
        "acc-AVG": avg_acc,
        "exact-acc-AVG": avg_exact_acc,
        "bleu": bleu_scores,
        "acc": acc_scores,
        "exact-acc": exact_acc_scores,
    }
    IOUtils.dump(results_file, results, IOUtils.Format.jsonNoSort)

    # DEBUGGING: print exact match report
    # report_file = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../slpproject/_results/vhdl/ALL/assignment-exact-match-output.txt"))
    # IOUtils.dump(report_file, report, IOUtils.Format.txt)
    return


if __name__=="__main__":
    inputs, preds, targets = read_data_preds(Path(args.data_dir), Path(args.pred_file), Path(args.tgt_file))
    write_results(targets, preds, Path(args.result_file))
