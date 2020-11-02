from typing import *

from onmt.decoders.decoder import DecoderBase
from onmt.encoders.encoder import EncoderBase
import torch
import torch.nn as nn

from seutil import LoggingUtils


class MultiSourceTypeAppendedModel(nn.Module):

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self,
                 encoders: Dict[str, EncoderBase],
                 decoder: DecoderBase,
    ):
        super().__init__()
        self.encoders = encoders
        for enc_i, (enc_type, encoder) in enumerate(self.encoders.items()):  self.add_module(f"encoder-{enc_i}", encoder)
        self.decoder = decoder
        return
    
    def forward(self,
            src_list: Dict[str, torch.Tensor],
            tgt: torch.LongTensor,
            lengths_list: Dict[str, torch.LongTensor],
            bptt: bool = False,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        
        enc_state_list: List = list()
        memory_bank_list: List = list()
        new_lengths_list: List = list()
        for enc_type, encoder in self.encoders.items():
            if enc_type=="l":
                enc_state, memory_bank, lengths = encoder(src_list[enc_type], src_list["type"], lengths_list[enc_type])
            else:
                enc_state, memory_bank, lengths = encoder(src_list[enc_type], src_list["patype"], lengths_list[enc_type])
                # enc_state, memory_bank, lengths = encoder(src_list[enc_type], lengths_list[enc_type])
            enc_state_list.append(enc_state)
            memory_bank_list.append(memory_bank)
            new_lengths_list.append(lengths)
        # end for

        if bptt is False:
            self.decoder.init_state(src_list, memory_bank_list, enc_state_list)
        # end if
        dec_out, attns = self.decoder(tgt, memory_bank_list, memory_lengths_list=new_lengths_list)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
