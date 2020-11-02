#!/usr/bin/env python

"""Train MultiSourceNMT models."""

import torch
import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.train_single_s2s import main as single_main
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple

from train import _get_parser
from train import ErrorHandler, batch_producer
from onmt.inputters.MultiSourceInputter import MultiSourceInputter


def main_ms(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    if len(opt.data_ids) > 1:
        train_shards = []
        for train_id in opt.data_ids:
            shard_base = "train_" + train_id
            train_shards.append(shard_base)
        train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
    else:
        if opt.data_ids[0] is not None:
            shard_base = "train_" + opt.data_ids[0]
        else:
            shard_base = "train"
        train_iter = MultiSourceInputter.build_dataset_iter(opt.src_types, shard_base, fields, opt)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, q, semaphore), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producer = mp.Process(target=batch_producer,
                              args=(train_iter, queues, semaphore, opt,),
                              daemon=True)
        producer.start()
        error_handler.add_child(producer.pid)

        for p in procs:
            p.join()
        producer.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)

        
def run(opt, device_id, error_queue, batch_queue, semaphore):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in Distributed initialization")

        self.train_single(opt, device_id, batch_queue, semaphore)

    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    main_ms(opt)
        

if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main_ms(opt)
