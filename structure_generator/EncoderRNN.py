import torch.nn as nn
import torch
import sys
from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size=0, embedding=None, hidden_size=0, pos_size=0, pemsize=0, embed_size=0, fdsize=0,
                 attn_src='emb', input_dropout_p=0, dropout_p=0, n_layers=1, rnn_cell='gru', directions=2, 
                 variable_lengths=True, field_concat_pos=False,
                 field_embedding=None):
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.attn_src = attn_src
        self.embed_size = embed_size
        self.fdsize = fdsize
        self.pemsize = pemsize
        self.variable_lengths = variable_lengths
        self.field_concat_pos = field_concat_pos
        self.pos_embedding = nn.Embedding(pos_size, self.pemsize, padding_idx=0)
        self.embedding = embedding
        self.field_embedding = field_embedding
        self.rnn = self.rnn_cell(self.embed_size + self.fdsize + self.pemsize * 2, self.hidden_size,
                                 n_layers,
                                 batch_first=True,
                                 bidirectional=(directions == 2),
                                 dropout=dropout_p)

    def forward(self, batch_s, batch_f, batch_pf, batch_pb, input_lengths=None):

        # get mask for location of PAD
        enc_mask = batch_s.eq(0).detach()

        embed_input = self.embedding(batch_s)
        if self.field_embedding is not None:
            embed_field = self.field_embedding(batch_f)
        else:
            embed_field = self.embedding(batch_f)
            
        embed_pf = self.pos_embedding(batch_pf)
        embed_pb = self.pos_embedding(batch_pb)
        embed_pos = torch.cat((embed_pf, embed_pb), dim=2)
        embed_field_pos = torch.cat((embed_field, embed_pos), dim=2)
        embedded = torch.cat((embed_input, embed_field_pos), dim=2)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        enc_hidden, enc_state = self.rnn(embedded)

        if self.attn_src == 'emb':
            enc_outputs = None
        else:
            enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_hidden, batch_first=True)
            enc_outputs = enc_outputs.contiguous()

        if self.field_concat_pos:
            return enc_outputs, embed_input, embed_field_pos, embed_pos, enc_state, enc_mask
        else:
            return enc_outputs, embed_input, embed_field, embed_pos, enc_state, enc_mask
