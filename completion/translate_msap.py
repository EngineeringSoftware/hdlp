#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is for testing S2S-LHSPA1-{2,3,4,5}{+Type} models.

from __future__ import unicode_literals
from typing import *

import os
from translate import _get_parser
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.utils.parse import ArgumentParser

from onmt.translate.MultiSourceAPTranslator import MultiSourceAPTranslator
from onmt.translate.MultiSourceAPTypeAppendedTranslator import MultiSourceAPTypeAppendedTranslator


def main(opt):
    ArgumentParser.validate_translate_opts(opt)

    logger = init_logger(opt.log_file)
    abs_path = os.path.dirname(opt.src)
    src_mode = opt.data_mode
    candidates_logprobs: List[List[Tuple[List[str], float]]] = list()

    if "patype0" in opt.src_types:
        translator = MultiSourceAPTypeAppendedTranslator.build_translator(opt.src_types, opt, report_score=True)
    else:
        translator = MultiSourceAPTranslator.build_translator(opt.src_types, opt, report_score=True)
    raw_data_keys = ["src.{}".format(src_type) for src_type in opt.src_types] + (["tgt"])
    raw_data_paths: Dict[str, str] = {
        k: "{0}/{1}.{2}.txt".format(abs_path, k, src_mode)
        for k in raw_data_keys
    }
    raw_data_shards: Dict[str, list] = {
        k: list(split_corpus(p, opt.shard_size))
        for k, p in raw_data_paths.items()
    }

    for i in range(len(list(raw_data_shards.values())[0])):
        logger.info("Translating shard %d." % i)
        _, _, candidates_logprobs_shard = translator.translate(
            {k: v[i] for k, v in raw_data_shards.items()},
            True,
            src_dir=None,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
        )
        candidates_logprobs.extend(candidates_logprobs_shard)

    # Reformat candidates
    candidates_logprobs: List[List[Tuple[str, float]]] = [[("".join(c), l) for c, l in cl] for cl in candidates_logprobs]
    return candidates_logprobs


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
