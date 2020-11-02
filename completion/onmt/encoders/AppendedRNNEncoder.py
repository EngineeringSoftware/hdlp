from typing import *

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import torch.nn as nn
import torch.nn.functional as F
import torch

VHDL_TYPE_INDX = dict()

class AppendedRNNEncoder(EncoderBase):
    
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(AppendedRNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size+len(VHDL_TYPE_INDX)-1,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings, type_token_indx=None):
        """Alternate constructor."""
        global VHDL_TYPE_INDX
        VHDL_TYPE_INDX = type_token_indx
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge)
    
    def forward(self, src, src_type, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        # src_type: (seq_len, batch, 1)
        # type_onehot_emb: (seq_len, batch, len(VHDL_TYPE_INDX)-1)
        #                  or (1, batch, len(VHDL_TYPE_INDX)-1)
        type_onehot_emb = self.get_onehot_vector(src_type)

        if type_onehot_emb.size(0)==1:
            # type_onehot_emb: (s_len, batch, len(VHDL_TYPE)-1)
            type_onehot_emb = type_onehot_emb.repeat(emb.size(0), 1, 1)
        
        assert emb.size(0)==type_onehot_emb.size(0)
        # emb: (s_len, batch, emb_dim+len(VHDL_TYPE_INDX))
        emb = torch.cat((emb, type_onehot_emb), dim=-1)
        
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            # PN: allow non-sorted
            packed_emb = pack(emb, lengths_list, enforce_sorted=False)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths

    def get_onehot_vector(self, src_type):
        # (seq_len, batch, 1) -> (seq_len, batch, len(VHDL_TYPE_INDX)-1)
        src_type = self.convert_vocab_indx_to_type_indx(src_type)
        seq_len = src_type.size(0)
        batch_size = src_type.size(1)
        res_vec = torch.zeros(seq_len, batch_size, len(VHDL_TYPE_INDX)-1)
        for step_i in range(seq_len):
            for batch_i in range(batch_size):
                if src_type[step_i, batch_i,:]<len(VHDL_TYPE_INDX)-1:
                    res_vec[step_i, batch_i, src_type[step_i, batch_i,:]] = 1
        return res_vec.cuda()

    def convert_vocab_indx_to_type_indx(self, src_type):
        vhdl_type_indx_tensor = torch.tensor(list(VHDL_TYPE_INDX.values())).cuda()
        for step_i in range(src_type.size(0)):
            for batch_i in range(src_type.size(1)):
                indx = (vhdl_type_indx_tensor==src_type[step_i, batch_i, :]).nonzero()
                if len(indx)==0:
                    src_type[step_i, batch_i, :] = len(VHDL_TYPE_INDX)-1
                else:
                    src_type[step_i, batch_i, :] = indx.squeeze()
        return src_type

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
