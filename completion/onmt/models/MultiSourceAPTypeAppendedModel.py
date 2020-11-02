from typing import *

from onmt.decoders.decoder import DecoderBase
from onmt.encoders.encoder import EncoderBase

import torch
import torch.nn as nn

from seutil import LoggingUtils


class MultiSourceAPTypeAppendedModel(nn.Module):

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self,
            encoders: List[EncoderBase],
            decoder: DecoderBase,
            ap_tool_dict: Dict,
            src_types,
            num_pas,
            model_opt,
    ):
        super().__init__()
        self.encoders = encoders
        for enc_type, encoder in self.encoders.items():
            self.add_module(f"encoder-{enc_type}", encoder)
        self.decoder = decoder
        self.enc_rnn_size = model_opt.enc_rnn_size
        self.num_pas = num_pas
        self.ap_tool_dict = ap_tool_dict
        self.src_types = src_types
        return

    def forward(self,
            src_list: Dict[str, torch.Tensor],
            tgt: torch.LongTensor,
            lengths_list: Dict[str, torch.LongTensor],
            bptt: bool = False,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state_list: List = list()
        memory_bank_list: List = list()
        new_lengths_list: List = list()
        for enc_type, encoder in self.encoders.items():
            if enc_type=="l":
                enc_state, memory_bank, lengths = encoder(src_list[enc_type], src_list["type"], lengths_list[enc_type])
            elif "prevassign" in enc_type:
                pa_indx = enc_type.split("prevassign")[-1]
                enc_state, memory_bank, lengths = encoder(src_list[enc_type],
                                                          src_list[f"patype{pa_indx}"],
                                                          lengths_list[enc_type])
            enc_state_list.append(enc_state)
            memory_bank_list.append(memory_bank)
            new_lengths_list.append(lengths)
        # end for
        
        if bptt is False:
            self.decoder.init_state(src_list, memory_bank_list, enc_state_list)
        # end if
        dec_out, attns = self.decoder(tgt, memory_bank_list, memory_lengths_list=new_lengths_list)
        return dec_out, attns

    def swich_dim(self, tensor, dim1, dim2):
        return torch.transpose(tensor, dim1, dim2)

    def search_token_index(self, tensor, token_indx):
        return torch.squeeze((tensor[:,0] == token_indx).nonzero(), dim=-1)

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
