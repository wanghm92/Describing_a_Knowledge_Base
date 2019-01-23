import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN
# self.word2idx['<UNK>'] = 1


class DecoderRNN(BaseRNN):

    def __init__(self, dec_type='pg', vocab_size=0, embedding=None, embed_size=0, hidden_size=0, fdsize=0, pemsize=0,
                 sos_id=3, eos_id=2, unk_id=1,
                 rnn_cell='gru', directions=2,
                 attn_src='emb', attn_type='concat', attn_fuse='sum', attn_level=2,
                 use_cov_loss=True, use_cov_attn=True, cov_in_pgen=False,
                 field_self_att=False, field_concat_pos=False, field_context=False, context_mlp=False,
                 mask=False, use_cuda=True, unk_gen=False,
                 n_layers=1, input_dropout_p=0, dropout_p=0, max_len=100, lmbda=1.5):

        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.decoder_type = dec_type
        self.attn_src = attn_src
        self.attn_type = attn_type
        self.attn_fuse = attn_fuse
        self.attn_level = attn_level
        self.directions = directions
        self.use_cov_loss = use_cov_loss
        self.use_cov_attn = use_cov_attn
        self.cov_in_pgen = cov_in_pgen
        self.field_self_att = field_self_att
        self.field_concat_pos = field_concat_pos
        self.field_context = field_context
        self.context_mlp = context_mlp
        self.output_size = vocab_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.mask = mask
        self.unk_gen = unk_gen
        self.embedding = embedding
        self.lmbda = lmbda
        self.use_cuda = use_cuda

        # ----------------- params for rnn cell ----------------- #
        # TODO: input feeding
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        # ----------------- params for directions ----------------- #
        if self.directions == 2:
            self.W_enc_state = nn.Linear(hidden_size * 2, hidden_size)

        # ----------------- parameters for self attention ----------------- #
        self_size = pemsize * 2
        if self.field_self_att:
            self.Win = nn.Linear(self_size, self_size)
            self.Wout = nn.Linear(self_size, self_size)
            self.Wg = nn.Linear(self_size, self_size)

        # ----------------- params for attention score ----------------- #
        field_input_size = fdsize
        # print('enc_input_size: {}'.format(enc_input_size))
        if self.attn_level > 1 and self.field_concat_pos:
            field_input_size += self_size
        # print('field_input_size: {}'.format(field_input_size))

        if self.attn_fuse == 'concat':
            self.v = nn.Linear(hidden_size, 1)
            self.Wd = nn.Linear(hidden_size, hidden_size)  # e_t decoder current state
        else:
            self.v_hidden = None
            self.v_input = None
            self.v_field = None

            if self.attn_level == 3:
                if self.attn_type != 'dot':
                    self.v_hidden = nn.Linear(hidden_size, 1)
                    self.v_input = nn.Linear(hidden_size, 1)
                    self.v_field = nn.Linear(hidden_size, 1)
                self.Wd_hidden = nn.Linear(hidden_size, hidden_size)
                self.Wd_input = nn.Linear(hidden_size, hidden_size)
                self.Wd_field = nn.Linear(hidden_size, hidden_size)

            elif self.attn_level == 2:
                if self.attn_type != 'dot':
                    self.v_field = nn.Linear(hidden_size, 1)
                self.Wd_field = nn.Linear(hidden_size, hidden_size)

                if self.attn_src == 'emb':
                    if self.attn_type != 'dot':
                        self.v_input = nn.Linear(hidden_size, 1)
                    self.Wd_input = nn.Linear(hidden_size, hidden_size)

                elif self.attn_src == 'rnn':
                    if self.attn_type != 'dot':
                        self.v_hidden = nn.Linear(hidden_size, 1)
                    self.Wd_hidden = nn.Linear(hidden_size, hidden_size)

            else:
                if self.attn_type != 'dot':
                    self.v_hidden = nn.Linear(hidden_size, 1)
                self.Wd_hidden = nn.Linear(hidden_size, hidden_size)

        # ----------------- params for encoder memory keys ----------------- #
        self.Wf = nn.Linear(field_input_size, hidden_size)

        enc_hidden_size = hidden_size * self.directions
        if self.attn_level == 3:
            self.We = nn.Linear(embed_size, hidden_size)  # e_t: word embeddings to keys
            self.Wr = nn.Linear(enc_hidden_size, hidden_size)  # e_t: encoder hidden states to keys
        elif self.attn_level == 2:
            if self.attn_src == 'emb':
                self.We = nn.Linear(embed_size, hidden_size)
            elif self.attn_src == 'rnn':
                self.Wr = nn.Linear(enc_hidden_size, hidden_size)
        else:
            # NOTE: assume to use encoder rnn hidden states when attn_level == 1
            self.Wr = nn.Linear(enc_hidden_size, hidden_size)

        if self.use_cov_attn:
            self.Wc = nn.Linear(1, hidden_size)  # e_t: coverage vector

        # ----------------- params for output ----------------- #
        if self.decoder_type != 'pt':
            output_layer_input_size = 0  # decoder state size
            if self.attn_level == 3:
                output_layer_input_size += (hidden_size + embed_size)
                if self.field_context:
                    output_layer_input_size += field_input_size
            elif self.attn_level == 2:
                if self.field_context:
                    output_layer_input_size += field_input_size
                if self.attn_src == 'emb':
                    output_layer_input_size += hidden_size
                elif self.attn_src == 'rnn':
                    output_layer_input_size += enc_hidden_size
            else:
                output_layer_input_size += enc_hidden_size

            # print('output_layer_input_size: {}'.format(output_layer_input_size))
            if self.context_mlp:
                self.V1 = nn.Linear(output_layer_input_size, hidden_size)
                self.V2 = nn.Linear(hidden_size*2, self.output_size)
            else:
                self.V = nn.Linear(output_layer_input_size + hidden_size, self.output_size)

        # ----------------- parameters for p_gen ----------------- #
        if self.decoder_type == 'pg':
            self.w_r = nn.Linear(enc_hidden_size, 1)    # encoder hidden context
            self.w_e = nn.Linear(embed_size, 1)         # encoder word context
            self.w_f = nn.Linear(field_input_size, 1)   # encoder field context
            self.w_d = nn.Linear(hidden_size, 1)        # decoder hidden state
            self.w_y = nn.Linear(embed_size,  1)        # decoder input word embedding

    def _pos_self_attn(self, enc_pos, enc_hidden, enc_input, enc_field, enc_mask):
        """ compute the self-attentive encoder output and field encodings"""

        enc_mask_float = enc_mask.unsqueeze(2).float()
        enc_mask_2d = enc_mask_float.bmm(enc_mask_float.transpose(1, 2))

        gin = torch.tanh(self.Win(enc_pos))
        gout = torch.tanh(self.Wout(enc_pos))

        f = gin.bmm(self.Wg(gout).transpose(1, 2))
        f.masked_fill_(enc_mask_2d.data.byte(), -np.inf)  # mask to -inf before applying softmax
        f_matrix = F.softmax(f, dim=2)

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

        return f_matrix, enc_hidden_selfatt, enc_input_selfatt, enc_field_selfatt

    def _get_enc_keys(self, enc_hidden, enc_input, enc_field, batch_size, max_enc_len):

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

    def _softmax(self, et, mask):
        """ apply mask to logits and compute softmax"""
        et.masked_fill_(mask.data.byte(), -np.inf)  # mask to -inf before applying softmax
        score = F.softmax(et, dim=1)  # along direction of sequence length
        return score

    def _attn_score_concat(self, batch_size, max_enc_len, vt, dec_query, enc_keys, cov_vector, enc_mask):

        attn_src = dec_query.unsqueeze(1).expand_as(enc_keys) + enc_keys
        # print('attn_src: {}'.format(attn_src.size()))

        if cov_vector is not None:
            # print('cov_vector: {}'.format(cov_vector.size()))
            attn_src += cov_vector

        et = vt(torch.tanh(attn_src).view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len)

        score = self._softmax(et, enc_mask)

        return score

    def _attn_score_dot(self, dec_query, enc_keys, enc_mask):

        et = dec_query.unsqueeze(1).bmm(enc_keys.transpose(1, 2)).squeeze(1)
        score = self._softmax(et, enc_mask)

        return score

    def _attn_score(self, batch_size, max_enc_len, vt, dec_query, enc_keys, cov_vector, enc_mask, attn_type='concat'):

        if attn_type == 'concat':
            return self._attn_score_concat(batch_size, max_enc_len, vt, dec_query, enc_keys, cov_vector, enc_mask)
        else:
            return self._attn_score_dot(dec_query, enc_keys, enc_mask)

    def _get_attn_score_fuse_concat(self, batch_size, max_enc_len, cov_vector, enc_mask,
                                    dec_hidden, enc_hidden_keys, enc_input_keys, enc_field_keys):

        dec_query = self.Wd(dec_hidden)

        if self.attn_level == 3:
            enc_keys = enc_hidden_keys + enc_input_keys + enc_field_keys
        elif self.attn_level == 2:
            if self.attn_src == 'emb':
                enc_keys = enc_input_keys + enc_field_keys
            elif self.attn_src == 'rnn':
                enc_keys = enc_hidden_keys + enc_field_keys
        else:
            enc_keys = enc_hidden_keys

        score = self._attn_score(batch_size, max_enc_len, self.v, dec_query, enc_keys, cov_vector, enc_mask)

        return score, score, score

    def _normalize(self, t, mask):
        # t.masked_fill_(mask.data.byte(), sys.float_info.epsilon)  # mask to epsilon before normalization
        normalizer = t.sum(dim=-1, keepdim=True).add_(sys.float_info.epsilon)
        return torch.div(t, normalizer)

    def _get_attn_score_fuse_hierarchical(self, batch_size, max_enc_len, cov_vector, enc_mask, dec_hidden,
                                          enc_hidden_keys, enc_input_keys, enc_field_keys, attn_type='concat'):

        attn_score_top = None
        attn_score_mid = None
        attn_score_btm = None

        if self.attn_level == 3:
            dec_query_hidden = self.Wd_hidden(dec_hidden)
            dec_query_input = self.Wd_input(dec_hidden)
            dec_query_field = self.Wd_field(dec_hidden)

            attn_score_hidden = self._attn_score(batch_size, max_enc_len, self.v_hidden, dec_query_hidden, 
                                                 enc_hidden_keys, cov_vector, enc_mask, attn_type=attn_type)
            attn_score_input = self._attn_score(batch_size, max_enc_len, self.v_input, dec_query_input,
                                                enc_input_keys, None, enc_mask, attn_type=attn_type)
            attn_score_field = self._attn_score(batch_size, max_enc_len, self.v_field, dec_query_field, 
                                                enc_field_keys, None, enc_mask, attn_type=attn_type)

            attn_score_btm = attn_score_field
            if self.attn_fuse == 'prod':
                attn_score_mid = self._normalize(torch.mul(attn_score_input, attn_score_btm), enc_mask)
                attn_score_top = self._normalize(torch.mul(attn_score_hidden, attn_score_btm), enc_mask)
            else:
                attn_score_mid = attn_score_input
                attn_score_top = attn_score_hidden

        elif self.attn_level == 2:
            dec_query_field = self.Wd_field(dec_hidden)

            attn_score_field = self._attn_score(batch_size, max_enc_len, self.v_field, dec_query_field, 
                                                enc_field_keys, None, enc_mask, attn_type=attn_type)
            attn_score_btm = attn_score_field

            if self.attn_src == 'emb':
                dec_query_input = self.Wd_input(dec_hidden)

                attn_score_input = self._attn_score(batch_size, max_enc_len, self.v_input, dec_query_input, 
                                                    enc_input_keys, cov_vector, enc_mask, attn_type=attn_type)
                if self.attn_fuse == 'prod':
                    attn_score_top = self._normalize(torch.mul(attn_score_input, attn_score_btm), enc_mask)
                else:
                    attn_score_top = attn_score_input

            elif self.attn_src == 'rnn':
                dec_query_hidden = self.Wd_hidden(dec_hidden)

                attn_score_hidden = self._attn_score(batch_size, max_enc_len, self.v_hidden, dec_query_hidden, 
                                                     enc_hidden_keys, cov_vector, enc_mask, attn_type=attn_type)
                if self.attn_fuse == 'prod':
                    attn_score_top = self._normalize(torch.mul(attn_score_hidden, attn_score_btm), enc_mask)
                else:
                    attn_score_top = attn_score_hidden

        else:
            dec_query_hidden = self.Wd_hidden(dec_hidden)

            attn_score_hidden = self._attn_score(batch_size, max_enc_len, self.v_hidden, dec_query_hidden, 
                                                 enc_hidden_keys, cov_vector, enc_mask, attn_type=attn_type)
            attn_score_top = attn_score_hidden

        return attn_score_top, attn_score_mid, attn_score_btm

    def _get_attn_scores(self, batch_size, max_enc_len, coverage, enc_mask, dec_hidden, 
                         enc_hidden_keys, enc_input_keys, enc_field_keys, attn_type):

        if self.use_cov_attn:
            cov_vector = self.Wc(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
        else:
            cov_vector = None

        if self.attn_fuse == 'concat':
            return self._get_attn_score_fuse_concat(batch_size, max_enc_len, cov_vector, enc_mask,
                                                    dec_hidden, enc_hidden_keys, enc_input_keys, enc_field_keys)
        else:
            return self._get_attn_score_fuse_hierarchical(batch_size, max_enc_len, cov_vector, enc_mask, dec_hidden,
                                                          enc_hidden_keys, enc_input_keys, enc_field_keys, attn_type)

    def _get_contexts(self, attn_scores, enc_hidden_vals, enc_input_vals, enc_field_vals):
        enc_context_proj = None
        if self.attn_level == 3:
            enc_hidden_context = attn_scores[0].unsqueeze(1).bmm(enc_hidden_vals).squeeze(1)
            enc_input_context = attn_scores[1].unsqueeze(1).bmm(enc_input_vals).squeeze(1)
            # output
            enc_output_context = torch.cat((enc_hidden_context, enc_input_context), 1)

            enc_field_context = attn_scores[2].unsqueeze(1).bmm(enc_field_vals).squeeze(1)
            if self.field_context:
                enc_output_context = torch.cat((enc_output_context, enc_field_context), 1)

            # p_gen
            if self.decoder_type == 'pg':
                enc_context_proj = self.w_r(enc_hidden_context) + \
                                   self.w_e(enc_input_context) + \
                                   self.w_f(enc_field_context)

        elif self.attn_level == 2:
            if self.attn_src == 'emb':
                # output
                enc_output_context = attn_scores[0].unsqueeze(1).bmm(enc_input_vals).squeeze(1)
                # p_gen
                if self.decoder_type == 'pg':
                    enc_context_proj = self.w_e(enc_output_context)

            elif self.attn_src == 'rnn':
                # output
                enc_output_context = attn_scores[0].unsqueeze(1).bmm(enc_hidden_vals).squeeze(1)
                # p_gen
                if self.decoder_type == 'pg':
                    enc_context_proj = self.w_r(enc_output_context)

            enc_field_context = attn_scores[2].unsqueeze(1).bmm(enc_field_vals).squeeze(1)
            if self.decoder_type == 'pg':
                enc_context_proj += self.w_f(enc_field_context)
            if self.field_context:
                enc_output_context = torch.cat((enc_output_context, enc_field_context), 1)

        else:
            # output
            enc_output_context = attn_scores[0].unsqueeze(1).bmm(enc_hidden_vals).squeeze(1)
            # p_gen
            if self.decoder_type == 'pg':
                enc_context_proj = self.w_r(enc_output_context)

        return enc_output_context, enc_context_proj

    def _decode_step(self,
                     batch_size, input_ids, coverage, max_source_oov,
                     dec_hidden, decoder_input,
                     enc_mask, max_enc_len,
                     enc_hidden_keys, enc_input_keys, enc_field_keys,
                     enc_hidden_vals, enc_input_vals, enc_field_vals
                     ):
        # print('input_ids: {}'.format(input_ids.size()))

        attn_scores = self._get_attn_scores(batch_size, max_enc_len, coverage, enc_mask, dec_hidden,
                                            enc_hidden_keys, enc_input_keys, enc_field_keys, self.attn_type)

        enc_output_context, enc_context_proj = self._get_contexts(attn_scores, enc_hidden_vals, enc_input_vals, enc_field_vals)
        # print('enc_output_context: {}'.format(enc_output_context.size()))
        # print('enc_context_proj: {}'.format(enc_context_proj.size()))

        if self.context_mlp:
            enc_output_context_rd = self.V1(enc_output_context)
            out_vec = torch.cat((dec_hidden, enc_output_context_rd), 1)
            out_vec = self.V2(out_vec)
        else:
            out_vec = self.V(torch.cat((dec_hidden, enc_output_context), 1))
        p_vocab = F.softmax(out_vec, dim=1)
        # print('p_vocab: {}'.format(p_vocab.size()))

        attn_weights = attn_scores[0]

        if self.decoder_type == 'pg':
            if self.cov_in_pgen:
                # print('coverage: {}'.format(coverage))
                cov_mean = coverage.mean(dim=-1, keepdim=True)
                # print('cov_mean: {}'.format(cov_mean))
                enc_context_proj += cov_mean
            p_gen_logits = enc_context_proj + self.w_d(dec_hidden) + self.w_y(decoder_input)
            # print('p_gen_logits: {}'.format(p_gen_logits.size()))
            p_gen = torch.sigmoid(p_gen_logits).view(-1, 1)
            # print('p_gen: {}'.format(p_gen.size()))

            weighted_Pvocab = p_vocab * p_gen
            # print('weighted_Pvocab: {}'.format(weighted_Pvocab.size()))

            # print('attn_weights: {}'.format(attn_weights.size()))

            weighted_attn = (1-p_gen) * attn_weights
            # print('weighted_attn: {}'.format(weighted_attn.size()))

            # print('max_source_oov: {}'.format(max_source_oov))
            if max_source_oov > 0:
                # create OOV (but in-article) zero vectors
                ext_vocab = torch.zeros(batch_size, max_source_oov)
                if self.use_cuda:
                    ext_vocab=ext_vocab.cuda()
                # print('ext_vocab: {}'.format(ext_vocab.size()))
                combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
                del ext_vocab
            else:
                combined_vocab = weighted_Pvocab
            del weighted_Pvocab       # 'Recheck OOV indexes!'

            # scatter article word probs to combined vocab prob.
            src_prob = combined_vocab.gather(1, input_ids)
            combined_vocab = combined_vocab.scatter_add(1, input_ids, weighted_attn)
            # print('combined_vocab: {}'.format(combined_vocab.size()))
            return combined_vocab, attn_weights, (p_gen, src_prob)

        elif self.decoder_type == 'seq':
            return p_vocab, attn_weights, (None, None)

    def forward(self, max_source_oov=0, targets=None, targets_id=None, input_ids=None,
                enc_hidden=None, enc_input=None, enc_state=None, enc_mask=None, enc_field=None, enc_pos=None,
                teacher_forcing_ratio=None, w2fs=None, fig=False):
        """
            target=batch_t, target_id=batch_o_t, input_ids=batch_o_s
        """

        targets, batch_size, max_length, max_enc_len = self._validate_args(targets, enc_state, enc_input, teacher_forcing_ratio)
        decoder_hidden_init = self._init_state(enc_state)
        if self.use_cov_loss or self.use_cov_attn:
            coverage = torch.zeros(batch_size, max_enc_len)
            # coverage_norm = torch.zeros(batch_size, max_enc_len)
            if self.use_cuda:
                coverage = coverage.cuda()
                # coverage_norm = coverage_norm.cuda()
        else:
            coverage = None
            # coverage_norm = None

        enc_hidden_keys, enc_input_keys, enc_field_keys = \
            self._get_enc_keys(enc_hidden, enc_input, enc_field, batch_size, max_enc_len)

        # get position self-attention scores
        if self.field_self_att:
            f_matrix, enc_hidden_vals, enc_input_vals, enc_field_vals\
                = self._pos_self_attn(enc_pos, enc_hidden, enc_input, enc_field, enc_mask)
        else:
            f_matrix = None
            enc_hidden_vals = enc_hidden
            enc_input_vals = enc_input
            enc_field_vals = enc_field

        if teacher_forcing_ratio:
            lm_loss, cov_loss = [], []
            dec_lens = (targets > 0).float().sum(1)

            embedded = self.embedding(targets)
            decoder_inputs = self.input_dropout(embedded)

            hidden, _ = self.rnn(decoder_inputs, decoder_hidden_init)

            # step through decoder hidden states
            for step in range(max_length):
                target_id = targets_id[:, step+1].unsqueeze(1)  # 0th is <SOS>, [batch] of ids of next word

                dec_hidden = hidden[:, step, :]
                decoder_input = decoder_inputs[:, step, :]

                combined_vocab, attn_weights, _ = self._decode_step(batch_size, input_ids, coverage, max_source_oov,
                                                                    dec_hidden, decoder_input,
                                                                    enc_mask, max_enc_len,
                                                                    enc_hidden_keys, enc_input_keys, enc_field_keys,
                                                                    enc_hidden_vals, enc_input_vals, enc_field_vals)
                # mask the output to account for PAD
                target_mask_0 = target_id.ne(0).detach()
                output = combined_vocab.gather(1, target_id).add_(sys.float_info.epsilon)
                # print('output: {}'.format(output.size()))
                _lm_loss = output.log().mul(-1) * target_mask_0.float()
                # print('_lm_loss: {}'.format(_lm_loss.size()))
                lm_loss.append(_lm_loss)

                if self.use_cov_loss:
                    coverage = coverage + attn_weights
                    # coverage_norm = coverage/(step + 1)
                    # print('coverage: {}'.format(coverage.size()))
                    # take minimum across both attn_weights and coverage
                    _cov_loss, _ = torch.stack((coverage, attn_weights), 2).min(2)
                    # _cov_loss, _ = torch.stack((coverage_norm, attn_weights), 2).min(2)
                    # print('_cov_loss: {}'.format(_cov_loss.size()))
                    cov_loss.append(_cov_loss.sum(1))

            # NOTE: loss is normalized by length, use sum of loss leads to faster conversion
            total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens)
            # total_masked_loss = torch.cat(lm_loss, 1).sum(1)
            if self.use_cov_loss:
                total_masked_loss = total_masked_loss + self.lmbda * torch.stack(cov_loss, 1).sum(1).div(dec_lens)

            return total_masked_loss
        else:
            return self.evaluate(targets, batch_size, max_length, max_source_oov,
                                 f_matrix, decoder_hidden_init, enc_mask, input_ids, coverage,
                                 enc_hidden_keys, enc_input_keys, enc_field_keys,
                                 enc_hidden_vals, enc_input_vals, enc_field_vals,
                                 max_enc_len, w2fs, fig)

    def evaluate(self, targets, batch_size, max_length, max_source_oov,
                 f_matrix, decoder_hidden_init, enc_mask, input_ids, coverage,
                 enc_hidden_keys, enc_input_keys, enc_field_keys,
                 enc_hidden_vals, enc_input_vals, enc_field_vals,
                 max_enc_len, w2fs, fig):

        lengths = np.array([max_length] * batch_size)
        finished = np.array([False] * batch_size)
        losses = []
        decoded_outputs = []
        src_probs = [] if self.decoder_type == 'pg' else None
        p_gens = [] if self.decoder_type == 'pg' else None
        if fig:
            attn = []
        decoder_input = self.embedding(targets)
        # step through decoder hidden states
        for step in range(max_length):
            dec_hidden, _c = self.rnn(decoder_input, decoder_hidden_init)
            combined_vocab, attn_weights, (p_gen, src_prob) = self._decode_step(batch_size, input_ids, coverage,
                                                                                max_source_oov,
                                                                                dec_hidden.squeeze(1),
                                                                                decoder_input.squeeze(1),
                                                                                enc_mask, max_enc_len,
                                                                                enc_hidden_keys, enc_input_keys,
                                                                                enc_field_keys,
                                                                                enc_hidden_vals, enc_input_vals,
                                                                                enc_field_vals)

            if not self.unk_gen:
                combined_vocab[:, self.unk_id] = 0  # NOTE: not allow decoder to output UNK
            probs, symbols = combined_vocab.topk(1)  # greedy decoding: get word indices and probs
            if self.mask:
                tmp_mask = torch.zeros_like(enc_mask, dtype=torch.uint8)
                for i in range(symbols.size(0)):
                    pos = (input_ids[i] == symbols[i]).nonzero()
                    if pos.size(0) != 0:
                        tmp_mask[i][pos] = 1
                enc_mask = torch.where(enc_mask > tmp_mask, enc_mask, tmp_mask)

            if fig:
                attn.append(attn_weights)
            decoded_outputs.append(symbols.clone())
            eos_batch = symbols.data.eq(self.eos_id)
            if eos_batch.dim() > 0:
                eos_batch = eos_batch.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batch) != 0
                lengths[update_idx] = len(decoded_outputs)
            # replace oov with the corresponding field embedding
            if self.decoder_type == 'pg':
                for i in range(symbols.size(0)):
                    w2f = w2fs[i]
                    if symbols[i].item() > self.vocab_size-1:
                        symbols[i] = w2f[symbols[i].item()]
            # symbols.masked_fill_((symbols > self.vocab_size-1), self.unk_id)
            decoder_input = self.embedding(symbols)
            decoder_hidden_init = _c
            if self.use_cov_loss:
                coverage = coverage + attn_weights

            # record eval loss
            target_mask_step = symbols.ne(self.eos_id).detach()
            probs = probs.add_(sys.float_info.epsilon)
            logits = probs.log()
            nll = logits.mul(-1)
            batch_loss = nll * target_mask_step.float()
            losses.append(batch_loss.mean().item())

            if self.decoder_type == 'pg':
                p_gens.append(p_gen)
                src_probs.append(src_prob)

            # check if all samples finished at the eos token
            finished_step = np.logical_not(np.array(target_mask_step.view(-1), dtype=bool))
            finished = np.logical_or(finished, finished_step)
            # stop if all finished
            if all(finished): break

        if self.decoder_type == 'pg':
            p_gens = torch.stack(p_gens, 1).squeeze(2)

        if fig:
            self_matrix = f_matrix if self.field_self_att else None
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), losses, p_gens, self_matrix, \
                   torch.stack(attn, 1).squeeze(2)
        else:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), losses, p_gens

    def _init_state(self, enc_state):
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

    def _validate_args(self, targets, enc_state, encoder_outputs, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        else:
            max_enc_len = encoder_outputs.size(1)
        # inference batch size
        if targets is None and enc_state is None:
            batch_size = 1
        else:
            if targets is not None:
                batch_size = targets.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = enc_state[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = enc_state.size(1)

        # set default targets and max decoding length
        if targets is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")
            # torch.set_grad_enabled(False)
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if self.use_cuda:
                targets = targets.cuda()
            max_length = self.max_length
        else:
            max_length = targets.size(1) - 1     # minus the start of sequence symbol

        return targets, batch_size, max_length, max_enc_len