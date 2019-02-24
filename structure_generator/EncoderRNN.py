import torch.nn as nn
import torch
import sys
from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size=0, embedding=None,
                 hidden_size=0, posit_size=0, embed_size=0, fdsize=0, dec_size=0,
                 attn_src='emb', input_dropout_p=0, dropout_p=0, n_layers=1, rnn_cell='gru', directions=2, 
                 variable_lengths=True, field_concat_pos=False,
                 field_embedding=None, pos_embedding=None, dataset_type=0, enc_type='rnn'):

        self.rnn_type = rnn_cell.lower()
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers)

        self.attn_src = attn_src
        self.embed_size = embed_size
        self.fdsize = fdsize
        self.posit_size = posit_size
        self.variable_lengths = variable_lengths
        self.field_concat_pos = field_concat_pos
        self.pos_embedding = pos_embedding
        self.embedding = embedding
        self.field_embedding = field_embedding
        self.dataset_type = dataset_type
        self.enc_type = enc_type
        self.input_size = self.embed_size + self.fdsize + self.posit_size

        if self.enc_type == 'fc':
            self.fc = nn.Sequential(nn.Linear(self.input_size, self.input_size), nn.ReLU())
            # self.bridge_h = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.ReLU())
            # if isinstance(self.rnn_cell, nn.LSTM):
            #     self.bridge_c = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.ReLU())
            self.linear = nn.Linear(self.input_size*2, self.input_size, bias=False)
            self.softmax = nn.Softmax(dim=2)
        elif self.enc_type == 'rnn':
            self.rnn = self.rnn_cell(self.input_size,
                                     self.hidden_size,
                                     n_layers,
                                     batch_first=True,
                                     bidirectional=(directions == 2),
                                     dropout=dropout_p)
        else:
            raise ValueError("{} enc_type is not supported".format(enc_type))

    def forward(self, batch_s, batch_f, batch_pf, batch_pb, input_lengths=None):

        # get mask for location of PAD
        enc_mask = batch_s.eq(0).detach()

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
        embedded = self.input_dropout(embedded)

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

            r = self.fc(embedded)
            rt = r.transpose(1, 2)

            align = torch.bmm(r, rt)
            align.masked_fill_(mask.data.byte(), -1e10)
            weights = self.softmax(align)
            c = torch.bmm(weights, r)
            r_att = self.linear(torch.cat([c, r], 2))
            enc_outputs = torch.sigmoid(r_att).mul(r)
            mean = torch.mean(enc_outputs, dim=1).unsqueeze(0)

            # enc_state_h = self.bridge_h(mean)
            if self.rnn_type.lower() == 'lstm':
                # enc_state_c = self.bridge_c(mean)
                enc_state = (mean, mean)
            else:
                enc_state = mean

            # print('mask: {}'.format(mask.size()))
            # print('mean: {}'.format(mean.size()))
            # print('r: {}'.format(r.size()))
            # print('rt: {}'.format(rt.size()))
            # print('align: {}'.format(align.size()))
            # print('weights: {}'.format(weights.size()))
            # print('c: {}'.format(c.size()))
            # print('r_att: {}'.format(r_att.size()))
            # print('enc_outputs: {}'.format(enc_outputs.size()))

        if self.field_concat_pos:
            return enc_outputs, embed_input, embed_field_pos, embed_pos, enc_state, enc_mask
        else:
            return enc_outputs, embed_input, embed_field, embed_pos, enc_state, enc_mask
