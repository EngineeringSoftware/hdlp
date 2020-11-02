#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is for preprocessing data and building vocabulary for
# S2S-LHS and S2S-LHS+PA{pa_index}{+Type} models.

import os
import codecs
import glob
import sys
import gc
import torch
from functools import partial
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)

        
def build_save_dataset_ms(corpus_type, fields, src_reader, tgt_reader, opt):
    abs_path = os.path.dirname(opt.train_src[0])
    from onmt.inputters.MultiSourceInputter import MultiSourceInputter
    from onmt.inputters.MultiSourceDataset import MultiSourceDataset

    assert corpus_type in ['train', 'valid']
    traineval = "train" if corpus_type == "train" else "val"
    raw_data_keys = ["src.{}".format(src_type) for src_type in opt.src_types] + (["tgt"])
    raw_data_paths = {
        k: "{0}/{1}.{2}.txt".format(abs_path, k, traineval)
        for k in raw_data_keys
    }

    if corpus_type == 'train':  counters = defaultdict(Counter)

    # for src, tgt, maybe_id in zip(srcs, tgts, ids):
    logger.info("Reading source and target files: {}".format(raw_data_paths.values()))

    raw_data_shards = {
        k: list(split_corpus(p, opt.shard_size))
        for k, p in raw_data_paths.items()
    }

    dataset_paths = []
    if (corpus_type == "train" or opt.filter_valid):
        filter_pred = partial(MultiSourceInputter.filter_example,
            src_types=opt.src_types,
            use_src_len=opt.data_type == "text",
            max_src_len=opt.src_seq_length,
            max_tgt_len=opt.tgt_seq_length,
        )
    else:
        filter_pred = None

    if corpus_type == "train":
        existing_fields = None
        if opt.src_vocab != "":
            try:
                logger.info("Using existing vocabulary...")
                existing_fields = torch.load(opt.src_vocab)
            except torch.serialization.pickle.UnpicklingError:
                logger.info("Building vocab from text file...")
                src_vocab, src_vocab_size = MultiSourceInputter.load_vocab(opt.src_vocab, "src", counters, opt.src_words_min_frequency)
            # end try
        else:
            src_vocab = None
        # end if

        if opt.tgt_vocab != "":
            tgt_vocab, tgt_vocab_size = MultiSourceInputter.load_vocab(opt.tgt_vocab, "tgt", counters, opt.tgt_words_min_frequency)
        else:
            tgt_vocab = None
        # end if
    # end if

    for i in range(len(list(raw_data_shards.values())[0])):
    # for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
    #     assert len(src_shard) == len(tgt_shard)
        logger.info("Building shard %d." % i)
        dataset = MultiSourceDataset(
            opt.src_types,
            fields,
            readers=([src_reader] * len(opt.src_types) + ([tgt_reader] if tgt_reader else [])),
            data=[(k, raw_data_shards[k][i]) for k in raw_data_keys],
            dirs=[None] * len(raw_data_keys),
            sort_key=inputters.str2sortkey[opt.data_type],
            filter_pred=filter_pred,
            can_copy=True if opt.use_copy else False,
        )
        if corpus_type == "train" and existing_fields is None:
            for ex in dataset.examples:
                for name, field in fields.items():
                    try:
                        f_iter = iter(field)
                    except TypeError:
                        f_iter = [(name, field)]
                        all_data = [getattr(ex, name, None)]
                    else:
                        all_data = getattr(ex, name)
                    # end try
                    for (sub_n, sub_f), fd in zip(f_iter, all_data):
                        has_vocab = (sub_n == 'src' and src_vocab is not None) or \
                                    (sub_n == 'tgt' and tgt_vocab is not None)
                        if (hasattr(sub_f, 'sequential') and sub_f.sequential and not has_vocab):
                            val = fd
                            counters[sub_n].update(val)
                        # end if
                    # end for
                # end for
            # end for
        # end if

        # if maybe_id:
        #     shard_base = corpus_type + "_" + maybe_id
        # else:
        shard_base = corpus_type
        # end if
        data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, shard_base, i)
        dataset_paths.append(data_path)

        logger.info(" * saving %sth %s data shard to %s." % (i, shard_base, data_path))

        dataset.save(data_path)
        
        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()
    # end for

    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        if existing_fields is None:
            fields = MultiSourceInputter.build_fields_vocab(
                opt.src_types,
                fields, counters,
                opt.share_vocab, opt.vocab_size_multiple,
                opt.src_vocab_size, opt.src_words_min_frequency,
                opt.tgt_vocab_size, opt.tgt_words_min_frequency)
        else:
            fields = existing_fields
        # end if
        torch.save(fields, vocab_path)
    # end if
    #return


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main_ms(opt):
    #ArgumentParser.validate_preprocess_args(opt)
    from onmt.inputters.MultiSourceInputter import MultiSourceInputter

    torch.manual_seed(opt.seed)
    if not(opt.overwrite):
        check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    abs_path = os.path.dirname(opt.train_src[0])
    src_nfeats = 0
    tgt_nfeats = 0

    logger.info("Building `Fields` object...")
    fields = MultiSourceInputter.get_fields(
        opt.src_types,
        src_nfeats,
        tgt_nfeats,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc)

    src_reader = inputters.str2reader["text"].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset_ms(
        'train', fields, src_reader, tgt_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset_ms('valid', fields, src_reader, tgt_reader, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')
    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main_ms(opt)
