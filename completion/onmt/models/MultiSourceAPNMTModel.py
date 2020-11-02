from typing import *

from onmt.decoders.decoder import DecoderBase
from onmt.encoders.encoder import EncoderBase
import torch
import torch.nn as nn

from seutil import LoggingUtils


class MultiSourceAPNMTModel(nn.Module):

    logger = LoggingUtils.get_logger(__name__)

    def __init__(self,
            encoders: List[EncoderBase],
            decoder: DecoderBase,
            ap_tool_dict: Dict,
            src_types,
            model_opt,
    ):
        super().__init__()
        self.encoders = encoders
        for enc_i, encoder in enumerate(self.encoders):  self.add_module(f"encoder-{enc_i}", encoder)
        self.decoder = decoder
        self.comb_operation = nn.AdaptiveMaxPool1d(1)
        self.pool_method = model_opt.pool_method
        self.enc_rnn_size = model_opt.enc_rnn_size
        self.num_pas = model_opt.num_pas
        if self.pool_method=="avg":
            self.comb_operation = nn.AdaptiveAvgPool1d(1)
        elif self.pool_method=="fc":
            self.comb_operation = nn.Linear(self.enc_rnn_size*self.num_pas, self.enc_rnn_size)
        self.ap_tool_dict = ap_tool_dict
        assert "prevassign" in src_types
        self.src_types = src_types
        self.rhsonly = model_opt.rhsonly
        return

    def forward(self,
            src_list: List[torch.Tensor],
            tgt: torch.LongTensor,
            lengths_list: List[torch.LongTensor],
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
        all_lengths_list: List = list()
        for enc_i, (encoder, src_type) in enumerate(zip(self.encoders, self.src_types)):
            if "prevassign" in src_type:
                # src_list[enc_i]: (src_len, batch, feat_len)
                # src_pas: (batch, src_len, feat_len)
                src_pas = self.swich_dim(src_list[enc_i], 0, 1)
                batch_size = src_pas.size()[0]
                feat_len = src_list[enc_i].size()[-1]

                src_pa_list: List = list()
                src_pa_lengths_list: List = list()
                src_sep_pa_nums: List = list()  # number of separate pas in each one single pa in batch
                for b_i in range(batch_size):
                    src_sep_pa_list: List = list()  # list of separate pas from one single pa in batch
                    src_sep_pa_lengths: List = list()  # list of length of separate pas from one single pa in batch
                    sep_indx_list = self.search_token_index(src_pas[b_i], self.ap_tool_dict["sep"])
                    src_sep_pa_nums.append(sep_indx_list.nelement())
                    frm = 0
                    if sep_indx_list.nelement()>0:
                        for i in sep_indx_list:
                            _len = i.tolist()-frm
                            # src_pas[b_i]: (src_len, feat_len)
                            # src_pas[b_i].narrow(0, frm, i): (_len, feat_len)
                            sep_pa_str = src_pas[b_i].narrow(0, frm, _len)
                            if self.rhsonly:
                                asn_indx = self.search_token_index(sep_pa_str, self.ap_tool_dict["asn"])
                                sep_pa_str = sep_pa_str.narrow(0, asn_indx+1, len(sep_pa_str)-asn_indx-1)
                            src_sep_pa_list.append(sep_pa_str)
                            src_sep_pa_lengths.append(_len)
                            frm = _len+1
                    else:
                        src_sep_pa_list.append(torch.tensor([[self.ap_tool_dict["emp"]]], device="cuda"))
                        src_sep_pa_lengths.append(1)
                    src_pa_list.append(src_sep_pa_list)
                    src_pa_lengths_list.append(src_sep_pa_lengths)

                sep_src_enc_state_list: List = list()
                memory_bank_pa_list: List = list()
                lengths_pa_list: List = list()
                rng = max(self.num_pas, 1)
                for p_i in range(-1, -rng-1, -1):
                    #print(src_pa_lengths_list)
                    sep_pas = [ sp[p_i] if len(sp)>0 and len(sp)>-p_i else torch.tensor([[self.ap_tool_dict["emp"]]], device="cuda") for sp in src_pa_list ]
                    sep_pas_lengths = [sp[p_i].size(0) if len(sp)>0 and len(sp)>-p_i else 1 for sp in src_pa_list]
                    sep_pas_maxlen = max(sep_pas_lengths)
                    sep_src = torch.full((batch_size, sep_pas_maxlen, feat_len), self.ap_tool_dict["pad"])
                    assert batch_size==len(sep_pas)
                    assert batch_size==len(sep_pas_lengths)
                    sep_pas_lengths = torch.tensor(sep_pas_lengths, device="cuda")
                    for i, sp in enumerate(sep_pas):
                        sep_src[i,:sp.size()[0],:sp.size()[1]] = sp

                    if torch.cuda.is_available():
                        sep_src = self.swich_dim(sep_src, 0, 1).type(torch.cuda.LongTensor)
                    else:
                        sep_src = self.swich_dim(sep_src, 0, 1).type(torch.cuda.LongTensor)

                    # enc_state: (num_dir, batch, hidden_size)
                    # memory_bank: (seq_len, batch, hidden_size)
                    # lengths: (batch)
                    enc_state, memory_bank, lengths = encoder(sep_src, sep_pas_lengths)
                    sep_src_enc_state_list.append(enc_state) #list of tensor(seq_len, batch, feat_len)
                    memory_bank_pa_list.append(memory_bank)
                    lengths_pa_list.append(lengths)

                if self.pool_method!="fc":
                    all_lengths_list += lengths_pa_list                   
                    sep_src_enc_state_list = torch.stack(sep_src_enc_state_list, dim=-1) # sep_src_enc_state_list: (num_dir, batch, hidden_size, num_pa)
                    # temp: (num_dir*batch, hidden_size, num_pa)
                    temp = sep_src_enc_state_list.view(sep_src_enc_state_list.shape[0]*sep_src_enc_state_list.shape[1],
                                                       sep_src_enc_state_list.shape[2],
                                                       sep_src_enc_state_list.shape[3])
                    enc_state = self.comb_operation(temp) # (num_dir*batch, hidden_size)                   
                    enc_state = enc_state.view(sep_src_enc_state_list.shape[0],
                                               sep_src_enc_state_list.shape[1],
                                               sep_src_enc_state_list.shape[2]) # (num_dir, batch, hidden_size)
                else:
                    # sep_src_enc_state_list: [num_pa * (num_dir, batch, hidden_size)] -> (num_dir, batch, hidden_size*num_pa)
                    sep_src_enc_state_list = torch.cat(sep_src_enc_state_list, dim=-1)
                    enc_state = self.comb_operation(sep_src_enc_state_list) # (num_dir, batch, hidden_size*num_pa) -> (num_dir, batch, hidden_size)
                    comb_memory_bank_list: List  = list()
                    comb_lengths_list: List  = list()
                    for b in range(enc_state.size(1)):
                        temp_comb_mb: List  = list()
                        temp_comb_sl: List  = list()
                        for p in range(self.num_pas):
                            sl = lengths_pa_list[p][b]
                            mb = memory_bank_pa_list[p][:sl,b,:]
                            temp_comb_mb.append(mb)
                            temp_comb_sl.append(sl)
                        new_mb = torch.cat(temp_comb_mb, dim=0)
                        new_sl = torch.stack(temp_comb_sl, dim=0).sum(dim=0)
                        comb_memory_bank_list.append(new_mb)
                        comb_lengths_list.append(new_sl)
                    new_lengths = torch.tensor(comb_lengths_list, device="cuda")
                    new_memory_bank = torch.full((max(comb_lengths_list), enc_state.size(1), enc_state.size(2)), self.ap_tool_dict["pad"], device="cuda")
                    for b in range(enc_state.size(1)):
                        new_memory_bank[:comb_lengths_list[b],b,:] = comb_memory_bank_list[b]
                    memory_bank_list.append(new_memory_bank)
                    all_lengths_list.append(new_lengths)
            else:
                enc_state, memory_bank, lengths = encoder(src_list[enc_i], lengths_list[enc_i])
                memory_bank_list.append(memory_bank)
                all_lengths_list.append(lengths)
                #lengths_list[enc_i] = lengths
            enc_state_list.append(enc_state)

        # end for

        if bptt is False:
            self.decoder.init_state(src_list, memory_bank_list, enc_state_list)
        # end if
        dec_out, attns = self.decoder(tgt, memory_bank_list, memory_lengths_list=all_lengths_list)
        return dec_out, attns

    def swich_dim(self, tensor, dim1, dim2):
        return torch.transpose(tensor, dim1, dim2)

    def search_token_index(self, tensor, token_indx):
        return torch.squeeze((tensor[:,0] == token_indx).nonzero(), dim=-1)

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
