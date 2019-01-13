import torch.nn as nn
import torch
import sys
from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, embedding, hidden_size, pos_size, pemsize, hidden_type='emb', input_dropout_p=0,
                 dropout_p=0, n_layers=1, bidirectional=True, rnn_cell='gru', variable_lengths=True):
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.hidden_type = hidden_type
        self.variable_lengths = variable_lengths
        self.pos_embedding = nn.Embedding(pos_size, pemsize, padding_idx=0)
        self.embedding = embedding
        self.rnn = self.rnn_cell((hidden_size + pemsize) * 2, hidden_size, n_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, batch_s, batch_f, batch_pf, batch_pb, input_lengths=None):

        # get mask for location of PAD
        mask = batch_s.eq(0).detach()

        embed_input = self.embedding(batch_s)
        embed_field = self.embedding(batch_f)
        embed_pf = self.pos_embedding(batch_pf)
        embed_pb = self.pos_embedding(batch_pb)
        embed_pos = torch.cat((embed_pf, embed_pb), dim=2)
        embed = torch.cat((embed_input, embed_field, embed_pos), dim=2)
        embed = self.input_dropout(embed)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embed, input_lengths, batch_first=True)

        enc_hidden, enc_state = self.rnn(embedded)

        if not self.hidden_type == 'emb':
            enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_hidden, batch_first=True)
        else:
            enc_outputs = None

        return enc_outputs, embed_input, embed_field, embed_pos, enc_state, mask




