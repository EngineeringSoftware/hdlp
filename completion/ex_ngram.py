# This script runs ngram models (LM-{1...10}gram) and rnn lm models
# (LM-RNN{+PA1-5}).

from typing import *
import os
import pickle
import argparse
import nltk
import numpy as np
import random
from recordclass import RecordClass
import time

from nltk.lm.preprocessing import padded_everygram_pipeline
# from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.lm import MLE
import torch
import torch.nn as nn
from onmt.utils.Vocabulary import VocabularyBuilder, Vocabulary

from pathlib import Path
from seutil import IOUtils


NUM_PA = 1
N_GRAM_ORDER = 10
MAX_LEN = 100

parser = argparse.ArgumentParser()
parser.add_argument('-modelname', '--modelname',
                    default="", required=False,
                    type=str, help='Save directory name')
parser.add_argument('-ref_modelname', '--ref_modelname',
                    default="S2S_LHS+PA1-5+Type", required=False,
                    type=str, help='Reference model for accessing dataset')
parser.add_argument('-order', '--order',
                    default=1, required=False,
                    type=int, help='N-gram order.')
parser.add_argument('-pa', '--pa',
                    default=1, required=True,
                    type=int, help='Number of pa.')
parser.add_argument('-mode', '--mode,', dest='mode',
                    default=False, action='store_true',
                    help='Flag for ngram model experiment over ngram orders(3 - 10).')
parser.add_argument('-rnn', '--rnn', dest='rnn',
                    type=bool, default=False, required=False,
                    help='If specified, use RNNLM.')
args = parser.parse_args()


def load_data(num_pa, ref_modelname):
    src_dict = dict()
    stat_list = list()
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"data/vhdl/{ref_modelname}")
    for mode in ["train", "val", "test"]:
        src_l_file = os.path.join(data_dir, f"src.l.{mode}.txt")
        src_l = [l.split() for l in IOUtils.load(src_l_file, IOUtils.Format.txt).strip().splitlines()]
        src_r_file = os.path.join(data_dir, f"tgt.{mode}.txt")
        src_r = [l.split() for l in IOUtils.load(src_r_file, IOUtils.Format.txt).strip().splitlines()]
        src_seq = [l+["<="]+r for l,r in zip(src_l, src_r)]
        for i in range(num_pa):
            src_pa_file = os.path.join(data_dir, f"src.prevassign{i}.{mode}.txt")
            src_pa = [l.split() for l in IOUtils.load(src_pa_file, IOUtils.Format.txt).strip().splitlines()]
            for j, pa in enumerate(src_pa):
                src_seq[j] = pa+src_seq[j]
        src_dict[f"{mode}"] = src_seq
    return src_dict


def generate_sent(model, text_seed: List[str], max_len=MAX_LEN, random_seed=42):
    content = []
    # detokenize = TreebankWordDetokenizer().detokenize

    for i in range(max_len):
        token = model.generate(text_seed=text_seed)
        if token == '<s>':
            continue
        if token == '</s>':
            break
        text_seed = text_seed[1:]+[token]
        content.append(token)
    # return detokenize(content)
    return content


def last_index_of(seqlist: List[str], value: str) -> int:
    return len(seqlist) - seqlist[::-1].index(value) - 1


def train_ngram_model(src_dict: dict, ngram_order=N_GRAM_ORDER):
    print(f"Training {ngram_order}-gram model on train dataset...")
    train_data, padded_sents = padded_everygram_pipeline(ngram_order, src_dict["train"])
    model = MLE(ngram_order)
    model.fit(train_data, padded_sents)
    return model


def get_seq_len_stat(seqs):
    seq_lens = [len(x) for x in seqs]
    stat = {"avg": np.mean(seq_lens),
            "max": max(seq_lens),
            "min": min(seq_lens)
            }
    return stat


def write_seq_len_stat(num_pa, ref_modelname):
    stat_list = list()
    src_l = list()
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"data/vhdl/{ref_modelname}")
    for mode in ["train","val","test"]:
        src_l_file = os.path.join(data_dir, f"src.l.{mode}.txt")
        src_l += [l.split() for l in IOUtils.load(src_l_file, IOUtils.Format.txt).strip().splitlines()]
    stat_list.append(get_seq_len_stat(src_l))

    for i in range(num_pa):
        src_pa = list()
        result_list = list()
        for mode in ["train", "val", "test"]:
            src_pa_file = os.path.join(data_dir, f"src.prevassign{i}.{mode}.txt")
            src_pa += [l.split() for l in IOUtils.load(src_pa_file, IOUtils.Format.txt).strip().splitlines()]
        for j, pa in enumerate(src_pa):
            if pa!=["<empty>"]:
                src_l[j] = pa+src_l[j]
                result_list.append(src_l[j])
        stat_list.append(get_seq_len_stat(result_list))
    results_file = os.path.join("../slpproject/_results/vhdl/ALL/metrics", f"lhs-pa-len-stat.json")
    IOUtils.dump(results_file, stat_list, IOUtils.Format.json)
    return 


def test_ngram_model(src_dict: dict, model, max_len=MAX_LEN, mode="test"):
    preds, tgts = [],[]
    print(f"Testing {mode} dataset...")
    for seq in src_dict[mode]:
        lindex = last_index_of(seq, "<=")
        input_seq = seq[:lindex+1]
        tgt = seq[lindex+1:]
        pred = generate_sent(model=model,
                             text_seed=input_seq,
                             max_len=max_len)
        preds.append(pred)
        tgts.append(tgt)
    return preds, tgts


def save_ngram_model(model, save_dir):
    model_dir: Path = Path(os.path.join("models", save_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "ngram_model.pkl"
    with open(save_path, 'wb') as fout:
        pickle.dump(model, fout)
    print(f"Model saved to {save_path}.")
    return


def load_ngram_model(save_dir):
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"models")
    load_path: Path = Path(os.path.join(model_dir, save_dir)) / "ngram_model.pkl"
    with open(load_path, 'rb') as fin:
        model_loaded = pickle.load(fin)
    print(f"Model loaded from {load_path}.")
    return model_loaded


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


def write_results(targets: List[List[str]], preds: List[List[str]], results_dir: Path, mode="test"):
    bleu_scores = list()
    acc_scores = list()
    exact_acc_scores = list()
    results_preds=""
    for t, p in zip(targets, preds):
        bleu_score = get_bleu(t, p)
        acc_score = get_accuracy(t, p)
        exact_acc_score = get_exact_match_accuracy(t, p)
        bleu_scores.append(bleu_score)
        acc_scores.append(acc_score)
        exact_acc_scores.append(exact_acc_score)
        results_preds += " ".join(p)
        results_preds += "\n"
        
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
    isval = ".assignments" if mode=="test" else ".val.assignments"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file: Path = results_dir / f"testlog{isval}.ngram.log"
    pred_file: Path = results_dir / f"pred{isval}.ngram.log"
    IOUtils.dump(results_file, results, IOUtils.Format.jsonNoSort)
    IOUtils.dump(pred_file, results_preds, IOUtils.Format.txt)
    return


def main_ngram(save_dir, order, pa, ref_modelname):
    src_dict = load_data(pa, ref_modelname)
    write_seq_len_stat(pa, ref_modelname)
    if not os.path.exists(f"models/{save_dir}"):
        model = train_ngram_model(src_dict,
                                  ngram_order=order)
        save_ngram_model(model, save_dir)
    model = load_ngram_model(save_dir)
    preds_test, tgts_test = test_ngram_model(src_dict,
                                             model,
                                             max_len=MAX_LEN,
                                             mode="test")
    write_results(tgts_test,
                  preds_test,
                  Path(f"tests/{save_dir}"),
                  mode="test")
    
    preds_val, tgts_val = test_ngram_model(src_dict,
                                           model,
                                           max_len=MAX_LEN,
                                           mode="val")
    write_results(tgts_val,
                  preds_val,
                  Path(f"tests/{save_dir}"),
                  mode="val")
    return


def performance_over_ngram_order(pa, ref_modelname):
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../slpproject/_results/ngram-pred")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    result_file = os.path.join(results_dir, "ngram_results_over_orders.txt")
    src_dict = load_data(pa, ref_modelname)
    avg_bleus = list()
    avg_accs = list()
    for order in range(3, 11):
        model = train_ngram_model(src_dict,
                                  ngram_order=order)
        preds_test, tgts_test = test_ngram_model(src_dict,
                                                 model,
                                                 max_len=MAX_LEN,
                                                 mode="test")
        bleu_scores = list()
        acc_scores = list()
        for t, p in zip(tgts_test, preds_test):
            bleu_score = get_bleu(t, p)
            acc_score = get_accuracy(t, p)
            bleu_scores.append(bleu_score)
            acc_scores.append(acc_score)
        # end for
        avg_bleus.append(np.mean(bleu_scores))
        avg_accs.append(np.mean(acc_scores))

    with open(result_file, "w") as f:
        for avg_acc, avg_bleu in zip(avg_accs, avg_bleus):
            f.write(f"{order} {pa} {avg_bleu:.3f} {avg_acc:.3f}\n")
    return

# ===== For RNN language model =====

class RNNLMConfig(RecordClass):
    dim_embed = 512
    dim_hidden = 512
    layers = 2
    dropout = 0.5

class RNNLM(nn.Module):
    def __init__(self, config: RNNLMConfig, vocab: Vocabulary):
        super(RNNLM, self).__init__()
        self.config = config
        self.vocab = vocab

        self.embed = nn.Embedding(num_embeddings=self.vocab.size(), embedding_dim=self.config.dim_embed, padding_idx=0)
        self.gru = nn.GRU(
            input_size=self.config.dim_embed,
            hidden_size=self.config.dim_hidden,
            num_layers=self.config.layers,
            dropout=self.config.dropout,
        )
        self.out = nn.Linear(in_features=self.config.dim_hidden, out_features=self.vocab.size())
        self.act = nn.LogSoftmax(dim=2)
        return

    def forward(self,
            tokens,  # [seq, batch]
            lengths,  # [batch]
    ) -> torch.Tensor:  # predictions, [seq, batch, vocab_size]
        embeddings = self.embed(tokens)  # [seq, batch, dim_embed]
        inputs_packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, enforce_sorted=False)  # PackedSequence (seq, dim_hidden)
        rnn_output_packed, h_n = self.gru(inputs_packed)  # rnn_output_packed: (seq, dim_hidden); h_n: [2, batch, dim_hidden]
        rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output_packed)  # [seq, batch, dim_hidden]
        outputs = self.act(self.out(rnn_output))  # [seq, batch, vocab_size]
        return outputs


class RNNLMTrainer:
    vocab_pad = "<pad>"
    vocab_unk = "<unk>"
    vocab_bos = "<s>"
    vocab_eos = "</s>"

    def __init__(self):
        if not torch.cuda.is_available():  print("WARNING: Cuda is not available")
        self.device_tag = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_tag)
        self.zero_const = torch.zeros(1, dtype=torch.long, device=self.device)
        return

    def preprocess_data(self, data: List[List[str]]) -> List[List[str]]:
        return [[self.vocab_bos] + sent + [self.vocab_eos] for sent in data]

    def process_sent(self, sent: List[str], vocab: Vocabulary) -> torch.Tensor:
        return torch.tensor([vocab.word2idx(w) for w in sent], dtype=torch.long, device=self.device)

    def batch_iter(self, data: List[List[str]], vocab: Vocabulary, batch_size: int, is_train: bool = False) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        if is_train:  random.shuffle(data)
        for beg in range(0, len(data), batch_size):
            input_tensors = [self.process_sent(sent, vocab) for sent in data[beg:beg + batch_size]]
            target_tensors = [torch.cat([t[1:], self.zero_const]) for t in input_tensors]

            lengths = torch.tensor([t.shape[0] for t in input_tensors], dtype=torch.long, device=self.device)

            input = torch.nn.utils.rnn.pad_sequence(input_tensors, padding_value=0)  # [seq, batch]
            target = torch.nn.utils.rnn.pad_sequence(target_tensors, padding_value=0)  # [seq, batch]
            yield input, target, lengths

            # Trash these tensors
            del input_tensors, target_tensors, input, target, lengths
        # end for

    def train(self, src_dict: Dict[str, List[List[str]]]):
        print(f"Training RNNLM on train+val dataset...")

        # Preprocess: add <bos> and <eos> tokens for each sent
        train_data = self.preprocess_data(src_dict["train"])
        train_data = [sent for sent in train_data if len(sent) <= MAX_LEN]
        val_data = self.preprocess_data(src_dict["val"])
        print(f"Data: train (after cleaning) {len(train_data)}, val {len(val_data)}")

        # Build vocab from train set
        vb = VocabularyBuilder(pad_token = self.vocab_pad, unk_token = self.vocab_unk)
        for sent in train_data:
            for w in sent:
                vb.add_word(w)
            # end for
        # end for
        vb.secure_word(self.vocab_bos)
        vb.secure_word(self.vocab_eos)

        vocab = vb.build(max_size=5000)
        print(f"Vocab size: {vocab.size()}")

        # Init model
        config = RNNLMConfig()
        model = RNNLM(config, vocab).to(self.device)
        loss_func = torch.nn.NLLLoss(ignore_index=0).to(self.device)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_loss = np.Inf
        early_stopping_threshold = 10
        early_stopping_counter = 0
        max_epochs = 1000
        batch_size = 512
        val_batch_size = 32
        val_every_epoch = 1
        train_history = list()
        time_beg = time.time()

        for epoch_i in range(1, max_epochs+1):
            total_train_loss = 0
            # Step train
            print(f"At epoch {epoch_i}, training ", end="", flush=True)
            for input, target, lengths in self.batch_iter(train_data, vocab, batch_size=batch_size, is_train=True):
                print(".", end="", flush=True)
                optimizer.zero_grad()
                logprobs = model(input, lengths)  # [seq, batch, vocab_size]
                train_loss = loss_func(logprobs.view(-1, vocab.size()), target.view(-1))
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item() * input.shape[1]

                # Trash unused
                del logprobs
            # end for
            avg_train_loss = total_train_loss / len(train_data)
            print(f" loss: {avg_train_loss}")

            train_report = {
                "type": "train",
                "elapsed_time": time.time() - time_beg,
                "epoch": epoch_i,
                "train_loss": avg_train_loss,
            }
            train_history.append(train_report)

            # Validation
            if epoch_i % val_every_epoch == 0:
                print(f"At epoch {epoch_i}, validating ...")
                with torch.no_grad():
                    total_val_loss = 0
                    for input, target, lengths in self.batch_iter(val_data, vocab, batch_size=val_batch_size, is_train=False):
                        logprobs = model(input, lengths)  # [seq, batch, vocab_size]
                        val_loss = loss_func(logprobs.view(-1, vocab.size()), target.view(-1))
                        total_val_loss += val_loss.item() * input.shape[1]

                        # Trash unused
                        del logprobs
                    # end for
                    avg_val_loss = total_val_loss / len(val_data)

                    val_report = {
                        "type": "val",
                        "elapsed_time": time.time() - time_beg,
                        "epoch": epoch_i,
                        "train_loss": avg_val_loss,
                    }
                    train_history.append(val_report)

                    # Check early stop
                    if avg_val_loss < best_loss:
                        print(f"Val loss improved: {best_loss} --> {avg_val_loss}")
                        best_loss = avg_val_loss
                        early_stopping_counter = 0
                    else:
                        # early_stopping_counter not working as early_stopping_threshold==0
                        if early_stopping_threshold>0:
                            early_stopping_counter += 1
                            print(f"Val loss: {avg_val_loss}")
                            print(f"Val loss not improving. Early stopping counter: {early_stopping_counter}/{early_stopping_threshold}")
                            if early_stopping_counter >= early_stopping_threshold:
                                print(f"Early stopping threshold reached!")
                                break
                            # end if
                        # end if
                    # end if
                # end with
            # end if
        # end for
        return model

    def save(self, model: RNNLM, save_dir: str):
        model_dir: Path = Path(os.path.join("models", save_dir))
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = model_dir / "model.pkl"
        torch.save(model, save_path)
        print(f"Model saved to {save_path}.")
        return

    def load(self, save_dir: str) -> RNNLM:
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"models")
        load_path: Path = Path(os.path.join(model_dir, save_dir)) / "model.pkl"
        model_loaded = torch.load(load_path)
        print(f"Model loaded from {load_path}.")
        return model_loaded

    def test(self, src_dict: dict, model: RNNLM, max_len=MAX_LEN, mode="test"):
        preds, tgts = [], []
        print(f"Testing {mode} dataset")
        for seq in src_dict[mode]:
            lindex = last_index_of(seq, "<=")
            input_seq = seq[:lindex + 1]
            tgt = seq[lindex + 1:]
            pred = self.generate_sent(model=model,
                text_seed=input_seq,
                max_len=max_len)
            preds.append(pred)
            tgts.append(tgt)
        return preds, tgts

    def generate_sent(self, model: RNNLM, text_seed: List[str], max_len=MAX_LEN) -> List[str]:
        content = []
        # detokenize = TreebankWordDetokenizer().detokenize

        for i in range(max_len):
            token = self.generate_token(model, text_seed=text_seed)
            if token == self.vocab_bos:
                # Shouldn't happen
                break
            if token == self.vocab_eos:
                break
            text_seed = text_seed[1:] + [token]
            content.append(token)
        # return detokenize(content)
        return content

    def generate_token(self, model: RNNLM, text_seed: List[str]) -> str:
        input_sent = [self.vocab_bos] + text_seed
        input_tensors = [torch.tensor([model.vocab.word2idx(w) for w in input_sent], dtype=torch.long, device=self.device)]
        input = torch.nn.utils.rnn.pad_sequence(input_tensors, padding_value=0)  # [seq, batch]
        lengths = torch.tensor([t.shape[0] for t in input_tensors], dtype=torch.long, device=self.device)
        logprobs = model(input, lengths)  # [seq, batch, vocab_size]
        prediction_idx = logprobs[-1, -1].argmax().item()
        return model.vocab.idx2word(prediction_idx)

    def main(self, save_dir: str, pa: str, ref_modelname: str):
        src_dict = load_data(pa, ref_modelname)
        if not os.path.exists(f"models/{save_dir}"):
            model = self.train(src_dict)
            self.save(model, save_dir)
        # end if
        model = self.load(save_dir)
        preds_test, tgts_test = self.test(src_dict,
            model,
            max_len=MAX_LEN,
            mode="test")
        write_results(tgts_test,
            preds_test,
            Path(f"tests/{save_dir}"),
            mode="test")

        preds_val, tgts_val = self.test(src_dict,
            model,
            max_len=MAX_LEN,
            mode="val")
        write_results(tgts_val,
            preds_val,
            Path(f"tests/{save_dir}"),
            mode="val")
        return


def main():
    ref_modelname = args.ref_modelname
    order = args.order
    pa = args.pa
    if args.mode:
        performance_over_ngram_order(pa, ref_modelname)
    else:
        pas = "1" if pa==1 else f"1-{pa}"
        if not args.rnn:
            # Use N-gram
            if args.modelname == "":
                modelname = f"LM-{order}gram" if pa==0 else f"LM-{order}gram+PA{pas}"
            else:
                modelname = args.modelname
            # end if
            main_ngram(modelname, order, pa, ref_modelname)
        else:
            # Use RNNLM
            if args.modelname == "":
                modelname = f"LM-RNN" if pa==0 else f"LM-RNN+PA{pas}"
            else:
                modelname = args.modelname
            # end if
            rnnlm_trainer = RNNLMTrainer()
            rnnlm_trainer.main(modelname, pa, ref_modelname)
        # end if
    # end if
    return


if __name__=="__main__":
    main()
