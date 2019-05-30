import torch
import torch.nn as nn
import sys
from .baseRNN import BaseRNN
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

class EncoderRNN(BaseRNN):
    def __init__(self,
                 vocab_size=0,
                 eos_id=2, mask_id=1,
                 hidden_size=0, posit_size=0, embed_size=0, fdsize=0, dec_size=0, attn_size=0,
                 attn_src='emb', attn_level=2, attn_type='',
                 dropout_p=0, n_layers=1, rnn_cell='gru', directions=2,
                 variable_lengths=True,
                 field_cat_pos=False, field_self_att=False,
                 embedding=None, field_embedding=None, pos_embedding=None,
                 dataset_type=0, enc_type='rnn'):

        self.rnn_type = rnn_cell.lower()
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, dropout_p, n_layers)

        self.attn_src = attn_src
        self.attn_level = attn_level
        self.attn_type = attn_type
        self.embed_size = embed_size
        self.fdsize = fdsize
        self.posit_size = posit_size  # including forward and backward positions/rcd and ha
        self.variable_lengths = variable_lengths
        self.field_cat_pos = field_cat_pos
        self.field_self_att = field_self_att
        self.embedding = embedding
        self.field_embedding = field_embedding
        self.dataset_type = dataset_type
        self.enc_type = enc_type
        self.dec_size = dec_size
        self.attn_size = attn_size
        self.eos_id = eos_id
        self.mask_id = mask_id
        self.directions = directions

        # ----------------- word and feature embeddings ----------------- #
        if self.dataset_type == 3:
            self.pos_embedding, self.rpos_embedding = pos_embedding
        else:
            self.pos_embedding = pos_embedding
            self.rpos_embedding = self.pos_embedding

        # ----------------- encoder layer ----------------- #
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
                                     bidirectional=(self.directions == 2),
                                     dropout=self.dropout_p)

        # ----------------- params for encoder memory keys ----------------- #
        enc_hidden_size = self.hidden_size * self.directions
        field_input_size = self.fdsize
        # print('input_size: {}'.format(self.input_size))
        if self.attn_level > 1 and self.field_cat_pos:
            field_input_size += self.posit_size
        # print('field_input_size: {}'.format(field_input_size))
        if self.enc_type == 'rnn' and self.attn_type == 'cat':
            self.Wf = nn.Linear(field_input_size, self.hidden_size, bias=False)  # field embeddings to keys
            if self.attn_level == 3:
                self.We = nn.Linear(self.embed_size, self.hidden_size, bias=False)  # word embeddings to keys
                self.Wr = nn.Linear(enc_hidden_size, self.hidden_size, bias=False)  # encoder hidden states to keys
            elif self.attn_level == 2:
                if self.attn_src == 'emb':
                    self.We = nn.Linear(self.embed_size, self.hidden_size, bias=False)
                elif self.attn_src == 'rnn':
                    self.Wr = nn.Linear(enc_hidden_size, self.hidden_size, bias=False)
            else:
                # NOTE: assume to use encoder rnn hidden states when attn_level == 1
                self.Wr = nn.Linear(enc_hidden_size, self.hidden_size, bias=False)
        else:
            self.We = None
            self.Wf = None
            self.Wr = None

        # ----------------- parameters for self attention ----------------- #
        if self.field_self_att:
            self.Win = nn.Linear(self.posit_size, self.posit_size)
            self.Wout = nn.Linear(self.posit_size, self.posit_size)
            self.Wg = nn.Linear(self.posit_size, self.posit_size)

        # ----------------- params for directions ----------------- #
        # TODO: this bridge should have Relu/Elu
        if self.directions == 2:
            self.W_enc_state = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.W_enc_state = nn.Linear(hidden_size, hidden_size)

    def _get_enc_keys(self, enc_hidden, enc_input, enc_field, batch_size, max_enc_len):
        """
        project encoder memories to attention keys
        :param enc_hidden: hidden states
        :param enc_input:  embeddings
        :param enc_field:  field embeddings
        :return: FC layer outputs, self.Wr for hidden, self.We for input, self.Wf for field
        """

        if self.attn_level == 3:
            enc_hidden_flat = enc_hidden.view(batch_size * max_enc_len, -1)
            enc_hidden_keys = self.Wr(enc_hidden_flat).view(batch_size, max_enc_len, -1)

            enc_input_flat = enc_input.view(batch_size * max_enc_len, -1)
            enc_input_keys = self.We(enc_input_flat).view(batch_size, max_enc_len, -1)

        elif self.attn_level == 2:
            if self.attn_src == 'rnn':
                enc_hidden_flat = enc_hidden.view(batch_size * max_enc_len, -1)
                enc_hidden_keys = self.Wr(enc_hidden_flat).view(batch_size, max_enc_len, -1)
                enc_input_keys = None
            elif self.attn_src == 'emb':
                enc_hidden_keys = None
                enc_input_flat = enc_input.view(batch_size * max_enc_len, -1)
                enc_input_keys = self.We(enc_input_flat).view(batch_size, max_enc_len, -1)

        else:
            enc_hidden_flat = enc_hidden.view(batch_size * max_enc_len, -1)
            enc_hidden_keys = self.Wr(enc_hidden_flat).view(batch_size, max_enc_len, -1)
            enc_input_keys = None

        if self.attn_level > 1:
            enc_field_flat = enc_field.view(batch_size * max_enc_len, -1)
            enc_field_keys = self.Wf(enc_field_flat).view(batch_size, max_enc_len, -1)
        else:
            enc_field_keys = None

        return enc_hidden_keys, enc_input_keys, enc_field_keys

    def _pos_self_attn(self, enc_pos, enc_hidden, enc_input, enc_field, enc_seq_mask):
        """ compute the self-attentive encoder output and field encodings"""

        enc_mask_float = enc_seq_mask.unsqueeze(2).float()
        enc_mask_2d = enc_mask_float.bmm(enc_mask_float.transpose(1, 2))

        gin = torch.tanh(self.Win(enc_pos))
        gout = torch.tanh(self.Wout(enc_pos))

        f = gin.bmm(self.Wg(gout).transpose(1, 2))
        f.masked_fill_(enc_mask_2d.data.byte(), -np.inf)  # mask to -inf before applying softmax
        f_matrix = torch.nn.functional.softmax(f, dim=2)

        if self.attn_level == 3:
            enc_hidden_selfatt = torch.bmm(f_matrix, enc_hidden)
            enc_input_selfatt = torch.bmm(f_matrix, enc_input)
            enc_field_selfatt = torch.bmm(f_matrix, enc_field)
        elif self.attn_level == 2:
            enc_field_selfatt = torch.bmm(f_matrix, enc_field)
            if self.attn_src == 'rnn':
                enc_hidden_selfatt = torch.bmm(f_matrix, enc_hidden)
                enc_input_selfatt = enc_input
            elif self.attn_src == 'emb':
                enc_hidden_selfatt = enc_hidden
                enc_input_selfatt = torch.bmm(f_matrix, enc_input)
        else:
            enc_hidden_selfatt = torch.bmm(f_matrix, enc_hidden)
            enc_input_selfatt = enc_input
            enc_field_selfatt = enc_field

        enc_selfatt = (enc_hidden_selfatt, enc_input_selfatt, enc_field_selfatt)
        return f_matrix, enc_selfatt

    def _build_memory_key_values(self, enc_outputs):
        enc_hidden, enc_input, enc_field, enc_pos, enc_state, enc_masks = enc_outputs
        enc_seq_mask, _ = enc_masks

        batch_size, max_enc_len, _ = enc_hidden.size()

        # print('enc_hidden: {}'.format(enc_hidden.size()))
        if self.enc_type == 'rnn' and self.attn_type == 'cat':
            enc_keys = self._get_enc_keys(enc_hidden, enc_input, enc_field, batch_size, max_enc_len)
        else:
            enc_keys = (enc_hidden, enc_input, enc_field)

        # get position self-attention scores
        if self.field_self_att:
            f_matrix, enc_vals = self._pos_self_attn(enc_pos, enc_hidden, enc_input, enc_field, enc_seq_mask)
        else:
            f_matrix = None
            enc_vals = (enc_hidden, enc_input, enc_field)

        return enc_keys, enc_vals, f_matrix

    def _state_bridge(self, enc_state):
        """ Initialize the encoder hidden state. """
        if enc_state is None:
            return None
        if isinstance(enc_state, tuple):
            enc_state = tuple([self._cat_directions(h) for h in enc_state])
        else:
            enc_state = self._cat_directions(enc_state)
        return enc_state

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.directions == 2:
            fw = h[0:h.size(0):2]
            bw = h[1:h.size(0):2]
            h = torch.cat([fw, bw], 2)
            h = self.W_enc_state(h)
        return h

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
            
        embed_pf = self.pos_embedding(batch_pf)
        embed_pb = self.rpos_embedding(batch_pb)
        embed_pos = torch.cat((embed_pf, embed_pb), dim=2)
        embed_field_pos = torch.cat((embed_field, embed_pos), dim=2)
        embedded = torch.cat((embed_input, embed_field_pos), dim=2)

        if self.enc_type == 'rnn':
            if self.variable_lengths:
                embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
            enc_hidden, enc_state = self.rnn(embedded)

            if self.attn_src == 'emb':
                enc_hidden = None
            else:
                enc_hidden, _ = nn.utils.rnn.pad_packed_sequence(enc_hidden, batch_first=True)
                enc_hidden = enc_hidden.contiguous()
            # print('enc_hidden: {}'.format(enc_hidden.size()))

        elif self.enc_type == 'fc':
            batch, source_len, _ = embedded.size()
            mask = enc_mask.unsqueeze(1)
            mask = mask.repeat(1, source_len, 1)
            mask_self_index = list(range(source_len))
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
            enc_hidden = torch.sigmoid(r_att).mul(r)
            # print('enc_hidden: {}'.format(enc_hidden.size()))
            mean = torch.mean(enc_hidden, dim=1).unsqueeze(0)
            # mean = self.W_enc_state(mean)
            # print('mean: {}'.format(mean.size()))

            # enc_state_h = self.bridge_h(mean)
            if self.rnn_type.lower() == 'lstm':
                # enc_state_c = self.bridge_c(mean)
                enc_state = (mean, mean)
            else:
                enc_state = mean

        if self.field_cat_pos:
            embed_field = embed_field_pos

        enc_masks = (enc_mask, enc_non_stop_mask)
        enc_outputs = (enc_hidden, embed_input, embed_field, embed_pos, enc_state, enc_masks)

        enc_keys, enc_vals, f_matrix = self._build_memory_key_values(enc_outputs)
        dec_state = self._state_bridge(enc_state)

        return enc_outputs, enc_keys, enc_vals, f_matrix, dec_state
