import torch.nn as nn
import torch
import sys
from .baseRNN import BaseRNN
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size=0, embedding=None,
                 eos_id=2, mask_id=1,
                 hidden_size=0, posit_size=0, embed_size=0, fdsize=0, dec_size=0, attn_size=0,
                 attn_src='emb', dropout_p=0, n_layers=1, rnn_cell='gru', directions=2,
                 variable_lengths=True, field_cat_pos=False,
                 field_embedding=None, pos_embedding=None, dataset_type=0, enc_type='rnn'):

        self.rnn_type = rnn_cell.lower()
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, dropout_p, n_layers)

        self.attn_src = attn_src
        self.embed_size = embed_size
        self.fdsize = fdsize
        self.posit_size = posit_size
        self.variable_lengths = variable_lengths
        self.field_cat_pos = field_cat_pos
        self.pos_embedding = pos_embedding
        self.embedding = embedding
        self.field_embedding = field_embedding
        self.dataset_type = dataset_type
        self.enc_type = enc_type
        self.dec_size = dec_size
        self.attn_size = attn_size
        self.eos_id = eos_id
        self.mask_id = mask_id

        self.input_size = self.embed_size + self.fdsize + self.posit_size

        if self.enc_type == 'fc':
            self.fc = nn.Sequential(nn.Linear(self.input_size, self.dec_size), nn.ReLU())
            if self.attn_size > 0:
                self.attn_query = nn.Sequential(nn.Linear(self.dec_size, self.attn_size), nn.ELU(0.1))
                self.attn_linear = nn.Linear(self.attn_size, self.attn_size, bias=False)
            self.linear_out = nn.Linear(self.dec_size*2, self.dec_size, bias=False)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.rnn = self.rnn_cell(self.input_size,
                                     self.hidden_size,
                                     n_layers,
                                     batch_first=True,
                                     bidirectional=(directions == 2),
                                     dropout=self.dropout_p)

    def forward(self, batch_s, batch_f, batch_pf, batch_pb, input_lengths=None):

        # get mask for location of PAD
        # print('batch_s: {}'.format(batch_s))
        enc_mask = batch_s.lt(self.mask_id).detach()
        enc_non_stop_mask = batch_s.eq(self.eos_id).detach()

        embed_input = self.embedding(batch_s)
        if self.field_embedding is not None:
            embed_field = self.field_embedding(batch_f)
        else:
            embed_field = self.embedding(batch_f)
            
        if self.dataset_type == 3:
            rcd_embedding, ha_embedding = self.pos_embedding
            embed_pf = rcd_embedding(batch_pf)
            embed_pb = ha_embedding(batch_pb)
        else:
            embed_pf = self.pos_embedding(batch_pf)
            embed_pb = self.pos_embedding(batch_pb)
        embed_pos = torch.cat((embed_pf, embed_pb), dim=2)
        embed_field_pos = torch.cat((embed_field, embed_pos), dim=2)
        embedded = torch.cat((embed_input, embed_field_pos), dim=2)

        if self.enc_type == 'rnn':
            if self.variable_lengths:
                embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
            enc_hidden, enc_state = self.rnn(embedded)

            if self.attn_src == 'emb':
                enc_outputs = None
            else:
                enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_hidden, batch_first=True)
                enc_outputs = enc_outputs.contiguous()

        elif self.enc_type == 'fc':
            batch, sourceL, dim = embedded.size()
            mask = enc_mask.unsqueeze(1)
            mask = mask.repeat(1, sourceL, 1)
            mask_self_index = list(range(sourceL))
            mask[:, mask_self_index, mask_self_index] = 1
            # print('mask: {}'.format(mask.size()))

            r = self.fc(embedded)
            r = self.dropout(r)
            if self.attn_size > 0:
                r_query = self.attn_query(r)
                r_key = self.attn_linear(r_query)
            else:
                r_query = r
                r_key = r
            # print('r_query: {}'.format(r_query.size()))
            # print('r_key: {}'.format(r_key.size()))

            r_query_t = r_query.transpose(1, 2)
            # print('r_query_t: {}'.format(r_query_t.size()))

            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            align = torch.bmm(r_key, r_query_t)
            align.masked_fill_(mask.data.byte(), -1e10)
            # print('align: {}'.format(align.size()))

            weights = self.softmax(align)
            # print('weights: {}'.format(weights.size()))
            c = torch.bmm(weights, r)
            # print('c: {}'.format(c.size()))
            r_att = self.linear_out(torch.cat([c, r], 2))
            # print('r_att: {}'.format(r_att.size()))
            enc_outputs = torch.sigmoid(r_att).mul(r)
            # print('enc_outputs: {}'.format(enc_outputs.size()))
            mean = torch.mean(enc_outputs, dim=1).unsqueeze(0)
            # print('mean: {}'.format(mean.size()))

            # enc_state_h = self.bridge_h(mean)
            if self.rnn_type.lower() == 'lstm':
                # enc_state_c = self.bridge_c(mean)
                enc_state = (mean, mean)
            else:
                enc_state = mean


        if self.field_cat_pos:
            return enc_outputs, embed_input, embed_field_pos, embed_pos, enc_state, (enc_mask, enc_non_stop_mask)
        else:
            return enc_outputs, embed_input, embed_field, embed_pos, enc_state, (enc_mask, enc_non_stop_mask)
