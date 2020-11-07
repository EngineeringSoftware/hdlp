#!/usr/bin/env python
"""Training on a single process."""
import os # DELETE

import torch

from onmt.inputters.inputter import load_old_vocab, old_style_vocab
from onmt.utils.optimizers import Optimizer
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.train_single import configure_process, _tally_parameters, _check_save_model_path

from onmt.inputters.MultiSourceInputter import MultiSourceInputter
from onmt.MultiSourceModelBuilder import MultiSourceModelBuilder
from onmt.MultiSourceTypeAppendedModelBuilder import MultiSourceTypeAppendedModelBuilder
from onmt.MultiSourceTrainer import MultiSourceTrainer
from onmt.MultiSourceTypeAppendedTrainer import MultiSourceTypeAppendedTrainer
from onmt.models.MultiSourceModelSaver import MultiSourceModelSaver


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    data_keys = [f"src.{src_type}" for src_type in opt.src_types] + ["tgt"]
    for side in data_keys:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    logger.info('Building model...')
    if opt.type_append:
        model = MultiSourceTypeAppendedModelBuilder.build_model(opt.src_types, model_opt, opt, fields, checkpoint)
    else:
        model = MultiSourceModelBuilder.build_model(opt.src_types, model_opt, opt, fields, checkpoint)
        
    logger.info(model)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = MultiSourceModelSaver.build_model_saver(
            opt.src_types, model_opt, opt, model, fields, optim)
    if opt.consist_reg:
        trainer = MultiSourceCRTrainer.build_trainer(
            opt.src_types, opt, device_id, model, fields, optim, model_saver=model_saver)
    elif opt.type_append:
        trainer = MultiSourceTypeAppendedTrainer.build_trainer(
            opt.src_types, opt, device_id, model, fields, optim, model_saver=model_saver)
    else:
        trainer = MultiSourceTrainer.build_trainer(
            opt.src_types, opt, device_id, model, fields, optim, model_saver=model_saver)

    if batch_queue is None:
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = MultiSourceInputter.build_dataset_iter_multiple(opt.src_types, train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            train_iter = MultiSourceInputter.build_dataset_iter(opt.src_types, shard_base, fields, opt)
    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                yield batch

        train_iter = _train_iter()

    valid_iter = MultiSourceInputter.build_dataset_iter(opt.src_types, "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()
