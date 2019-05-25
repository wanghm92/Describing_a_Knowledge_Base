import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN
# self.word2idx['<UNK>'] = 1
# np.set_printoptions(threshold=np.nan)

class DecoderRNN(BaseRNN):

    def __init__(self,
                 dec_type='pg', dataset_type=0,
                 ptr_input='emb', ptr_feat_merge='mlp',
                 vocab_size=0, embed_size=0, hidden_size=0, fdsize=0, posit_size=0,
                 pad_id=0, sos_id=1, eos_id=2, unk_id=3,
                 rnn_cell='gru', directions=2,
                 attn_src='emb', attn_type='cat', attn_fuse='sum', attn_level=2,
                 ptr_dec_feat=False, input_feeding=False,
                 use_cov_loss=True, use_cov_attn=True, cov_in_pgen=False,
                 field_self_att=False, field_cat_pos=False,
                 field_context=False, context_mlp=False,
                 mask=False, use_cuda=True, unk_gen=False,
                 max_len=100, min_len=20, n_layers=1, dropout_p=0, lmbda=1.5,
                 embedding=None, field_embedding=None, pos_embedding=None):

        self.rnn_type = rnn_cell.lower()
        super(DecoderRNN, self).__init__(vocab_size, hidden_size, dropout_p, n_layers)

        self.decoder_type = dec_type
        self.ptr_input = ptr_input
        self.ptr_feat_merge = ptr_feat_merge
        self.attn_src = attn_src
        self.attn_type = attn_type
        self.attn_fuse = attn_fuse
        self.attn_level = attn_level
        self.directions = directions
        self.use_cov_loss = use_cov_loss
        self.use_cov_attn = use_cov_attn
        self.cov_in_pgen = cov_in_pgen
        self.field_self_att = field_self_att
        self.field_cat_pos = field_cat_pos
        self.field_context = field_context
        self.context_mlp = context_mlp
        self.output_size = vocab_size
        self.max_length = max_len
        self.min_length = min_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.mask = mask
        self.unk_gen = unk_gen
        self.embedding = embedding
        self.field_embedding = field_embedding
        self.dataset_type = dataset_type
        self.ptr_dec_feat = ptr_dec_feat
        self.input_feeding = input_feeding
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        if self.dataset_type == 3:
            self.pos_embedding, self.rpos_embedding = pos_embedding
        else:
            self.pos_embedding = pos_embedding
            self.rpos_embedding = self.pos_embedding
        self.lmbda = lmbda
        self.use_cuda = use_cuda
        if self.decoder_type != 'pg':
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        self_size = posit_size
        enc_hidden_size = hidden_size * self.directions

        # ----------------- params for attention score ----------------- #
        field_input_size = fdsize
        # print('enc_input_size: {}'.format(enc_input_size))
        if self.attn_level > 1 and self.field_cat_pos:
            field_input_size += self_size
        # print('field_input_size: {}'.format(field_input_size))

        if self.attn_fuse == 'cat':
            self.v = nn.Linear(hidden_size, 1, bias=False)
            self.Wd = nn.Linear(hidden_size, hidden_size)  # e_t decoder current state
        else:
            self.v_hidden = None
            self.v_input = None
            self.v_field = None

            if self.attn_level == 3:
                if self.attn_type != 'dot':
                    self.v_hidden = nn.Linear(hidden_size, 1, bias=False)
                    self.v_input = nn.Linear(hidden_size, 1, bias=False)
                    self.v_field = nn.Linear(hidden_size, 1, bias=False)
                self.Wd_hidden = nn.Linear(hidden_size, hidden_size)
                self.Wd_input = nn.Linear(hidden_size, hidden_size)
                self.Wd_field = nn.Linear(hidden_size, hidden_size)

            elif self.attn_level == 2:
                if self.attn_type != 'dot':
                    self.v_field = nn.Linear(hidden_size, 1, bias=False)
                self.Wd_field = nn.Linear(hidden_size, hidden_size)

                if self.attn_src == 'emb':
                    if self.attn_type != 'dot':
                        self.v_input = nn.Linear(hidden_size, 1, bias=False)
                    self.Wd_input = nn.Linear(hidden_size, hidden_size)

                elif self.attn_src == 'rnn':
                    if self.attn_type != 'dot':
                        self.v_hidden = nn.Linear(hidden_size, 1, bias=False)
                    self.Wd_hidden = nn.Linear(hidden_size, hidden_size)

            else:
                if self.attn_type != 'dot':
                    self.v_hidden = nn.Linear(hidden_size, 1, bias=False)
                self.Wd_hidden = nn.Linear(hidden_size, hidden_size)

        if self.use_cov_attn:
            self.Wc = nn.Linear(1, hidden_size)  # e_t: coverage vector

        # ----------------- params for output ----------------- #
        # if not (self.decoder_type == 'pt' and not self.input_feeding):
        output_layer_input_size = 0  # decoder state size
        if self.attn_level == 3:
            output_layer_input_size += (enc_hidden_size + embed_size)
            if self.field_context:
                output_layer_input_size += field_input_size
        elif self.attn_level == 2:
            if self.field_context:
                output_layer_input_size += field_input_size
            if self.attn_src == 'emb':
                output_layer_input_size += embed_size
            elif self.attn_src == 'rnn':
                output_layer_input_size += enc_hidden_size
        else:
            output_layer_input_size += enc_hidden_size

        # print('output_layer_input_size: {}'.format(output_layer_input_size))
        if self.context_mlp:
            self.V1 = nn.Linear(output_layer_input_size, hidden_size)
            self.V = nn.Linear(hidden_size*2, self.output_size)
        else:
            self.V = nn.Linear(output_layer_input_size + hidden_size, self.output_size)

        # ----------------- parameters for p_gen ----------------- #
        if self.decoder_type == 'pg':
            self.w_r = nn.Linear(enc_hidden_size, 1)    # encoder hidden context
            self.w_e = nn.Linear(embed_size, 1)         # encoder word context
            self.w_f = nn.Linear(field_input_size, 1)   # encoder field context
            self.w_d = nn.Linear(hidden_size, 1)        # decoder hidden state
            self.w_y = nn.Linear(embed_size,  1)        # decoder input word embedding

        # ----------------- params for rnn cell ----------------- #
        self.input_size = embed_size
        if self.decoder_type == 'pt':
            if self.ptr_input == 'emb':
                if self.ptr_dec_feat:
                    if self.ptr_feat_merge == 'mlp':
                        # TODO: try share the same fc/linear layer with encoder
                        self.input_mlp = nn.Sequential(nn.Linear(embed_size + fdsize + posit_size, embed_size), nn.ReLU())
                    elif self.ptr_feat_merge == 'cat':
                        self.input_size += (fdsize + posit_size)
                    else:
                        raise ValueError("{} feat_merge type not supported".format(self.ptr_feat_merge))
            elif self.ptr_input == 'hid':
                self.input_size = hidden_size  # TODO: same hidden size for encoder and decoder for now, change

        if self.input_feeding:
            self.input_size += hidden_size
        self.rnn = self.rnn_cell(self.input_size, hidden_size, n_layers,
                                 batch_first=True, dropout=self.dropout_p)

        if self.input_feeding and self.attn_level > 1 and not self.context_mlp:
            raise ValueError("input_feeding = {}, attn_level = {}, context_mlp = {} is NOT ALLOWED"
                             .format(self.input_feeding, self.attn_level, self.context_mlp))

    def _attn_score_cat(self, batch_size, max_enc_len, vt, dec_query, enc_keys, cov_vector, enc_mask, no_dup_mask=None):
        """
        attention score in the form e = v` tanh(Wx+b)
        :param dec_query:  attention query vectors
        :param enc_keys:   output from self._get_enc_keys
        :param cov_vector: coverage vector
        """

        attn_src = dec_query.unsqueeze(1).expand_as(enc_keys) + enc_keys
        # print('attn_src: {}'.format(attn_src.size()))

        if cov_vector is not None:
            # print('cov_vector: {}'.format(cov_vector.size()))
            attn_src += cov_vector

        et = vt(torch.tanh(attn_src).view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len)
        et.masked_fill_(enc_mask.data.byte(), -1e10)  # mask to -1e10 before applying softmax
        if no_dup_mask is not None:
            et.masked_fill_(no_dup_mask.data.byte(), -1e10)  # mask to -1e10 before applying softmax

        if not self.decoder_type == 'pt':
            score = F.softmax(et, dim=1)  # along direction of sequence length
            return score
        else:
            return et

    def _attn_score_dot(self, dec_query, enc_keys, enc_mask, no_dup_mask=None):
        """
        attention score in the form e = x*y
        :param dec_query:  attention query vectors
        :param enc_keys:   output from self._get_enc_keys
        :param cov_vector: coverage vector
        """

        et = dec_query.unsqueeze(1).bmm(enc_keys.transpose(1, 2)).squeeze(1)
        et.masked_fill_(enc_mask.data.byte(), -1e10)
        if no_dup_mask is not None:
            et.masked_fill_(no_dup_mask.data.byte(), -1e10)  # mask to -1e10 before applying softmax

        if not self.decoder_type == 'pt':
            score = F.softmax(et, dim=1)  # along direction of sequence length
            return score
        else:
            # NOTE: here et is masked to -1e10 at paddings
            return et

    def _attn_score(self, batch_size, max_enc_len, vt, dec_query, enc_keys, cov_vector, enc_mask,
                    attn_type='cat', no_dup_mask=None):
        """ Wrapper for two types of attention scores: cat and dot"""

        if attn_type == 'cat':
            return self._attn_score_cat(batch_size, max_enc_len, vt, dec_query, enc_keys,
                                        cov_vector, enc_mask, no_dup_mask=no_dup_mask)
        else:
            return self._attn_score_dot(dec_query, enc_keys, enc_mask, no_dup_mask=no_dup_mask)

    def _attn_score_fuse_cat(self, batch_size, max_enc_len, cov_vector, enc_mask, dec_hidden,
                             enc_hidden_keys, enc_input_keys, enc_field_keys, no_dup_mask=None):
        """ normal multi-source attention score using cat"""

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

        logit_or_score = self._attn_score(batch_size, max_enc_len, self.v, dec_query, enc_keys,
                                          cov_vector, enc_mask, no_dup_mask=no_dup_mask)

        return logit_or_score, logit_or_score, logit_or_score

    def _normalize(self, t):
        # t.masked_fill_(mask.data.byte(), sys.float_info.epsilon)  # mask to epsilon before normalization
        normalizer = t.sum(dim=-1, keepdim=True).add_(sys.float_info.epsilon)
        return torch.div(t, normalizer)

    def _attn_score_fuse_hier(self, batch_size, max_enc_len, cov_vector, enc_mask, dec_hidden,
                              enc_hidden_keys, enc_input_keys, enc_field_keys, attn_type='cat', no_dup_mask=None):
        """ aggregated attention score with normalization from lower layers"""

        attn_score_top = None
        attn_score_mid = None
        attn_score_btm = None

        if self.attn_level == 3:
            dec_query_hidden = self.Wd_hidden(dec_hidden)
            dec_query_input = self.Wd_input(dec_hidden)
            dec_query_field = self.Wd_field(dec_hidden)

            attn_score_hidden = self._attn_score(batch_size, max_enc_len, self.v_hidden, dec_query_hidden, 
                                                 enc_hidden_keys, cov_vector, enc_mask, attn_type=attn_type,
                                                 no_dup_mask=no_dup_mask)
            attn_score_input = self._attn_score(batch_size, max_enc_len, self.v_input, dec_query_input,
                                                enc_input_keys, None, enc_mask, attn_type=attn_type,
                                                no_dup_mask=no_dup_mask)
            attn_score_field = self._attn_score(batch_size, max_enc_len, self.v_field, dec_query_field, 
                                                enc_field_keys, None, enc_mask, attn_type=attn_type,
                                                no_dup_mask=no_dup_mask)

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
                                                enc_field_keys, None, enc_mask, attn_type=attn_type,
                                                no_dup_mask=no_dup_mask)
            attn_score_btm = attn_score_field

            if self.attn_src == 'emb':
                dec_query_input = self.Wd_input(dec_hidden)

                attn_score_input = self._attn_score(batch_size, max_enc_len, self.v_input, dec_query_input, 
                                                    enc_input_keys, cov_vector, enc_mask, attn_type=attn_type,
                                                    no_dup_mask=no_dup_mask)
                if self.attn_fuse == 'prod':
                    attn_score_top = self._normalize(torch.mul(attn_score_input, attn_score_btm), enc_mask)
                else:
                    attn_score_top = attn_score_input

            elif self.attn_src == 'rnn':
                dec_query_hidden = self.Wd_hidden(dec_hidden)

                attn_score_hidden = self._attn_score(batch_size, max_enc_len, self.v_hidden, dec_query_hidden, 
                                                     enc_hidden_keys, cov_vector, enc_mask, attn_type=attn_type,
                                                     no_dup_mask=no_dup_mask)
                if self.attn_fuse == 'prod':
                    attn_score_top = self._normalize(torch.mul(attn_score_hidden, attn_score_btm), enc_mask)
                else:
                    attn_score_top = attn_score_hidden

        else:
            dec_query_hidden = self.Wd_hidden(dec_hidden)

            attn_score_hidden = self._attn_score(batch_size, max_enc_len, self.v_hidden, dec_query_hidden, 
                                                 enc_hidden_keys, cov_vector, enc_mask, attn_type=attn_type,
                                                 no_dup_mask=no_dup_mask)
            attn_score_top = attn_score_hidden

        return attn_score_top, attn_score_mid, attn_score_btm # TODO: check for pointer-net

    def _get_attn_scores(self, batch_size, max_enc_len, coverage, enc_mask, dec_hidden, 
                         enc_hidden_keys, enc_input_keys, enc_field_keys, attn_type='cat', no_dup_mask=None):
        """ Meta wrapper for attention scores with and without coverage"""

        if self.use_cov_attn:
            cov_vector = self.Wc(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
        else:
            cov_vector = None

        if self.attn_fuse == 'cat':
            return self._attn_score_fuse_cat(batch_size, max_enc_len, cov_vector, enc_mask, dec_hidden,
                                             enc_hidden_keys, enc_input_keys, enc_field_keys,
                                             no_dup_mask=no_dup_mask)
        else:
            return self._attn_score_fuse_hier(batch_size, max_enc_len, cov_vector, enc_mask, dec_hidden,
                                              enc_hidden_keys, enc_input_keys, enc_field_keys, attn_type,
                                              no_dup_mask=no_dup_mask)

    def _get_contexts(self, attn_scores, enc_hidden_vals, enc_input_vals, enc_field_vals):
        """ project encoder memory bank to compute the source context vectors, later weighted by attention scores"""

        # ptr-net returns logits on encoder hidden state for calculating CE loss
        if self.decoder_type == 'pt':
            x, y, z = attn_scores
            attn_scores = (F.softmax(x, dim=1), F.softmax(y, dim=1), F.softmax(z, dim=1))

        enc_context_proj = None
        if self.attn_level == 3:
            enc_hidden_context = attn_scores[0].unsqueeze(1).bmm(enc_hidden_vals).squeeze(1)
            enc_input_context = attn_scores[1].unsqueeze(1).bmm(enc_input_vals).squeeze(1)
            # output
            enc_output_context = torch.cat((enc_hidden_context, enc_input_context), -1)

            enc_field_context = attn_scores[2].unsqueeze(1).bmm(enc_field_vals).squeeze(1)
            if self.field_context:
                enc_output_context = torch.cat((enc_output_context, enc_field_context), -1)

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
                enc_output_context = torch.cat((enc_output_context, enc_field_context), -1)

        else:
            # output
            enc_output_context = attn_scores[0].unsqueeze(1).bmm(enc_hidden_vals).squeeze(1)
            # p_gen
            if self.decoder_type == 'pg':
                enc_context_proj = self.w_r(enc_output_context)

        return enc_output_context, enc_context_proj

    def _decode_step(self,
                     batch_size, input_ids, coverage, max_tail_oov,
                     dec_hidden, decoder_input,
                     enc_mask, max_enc_len,
                     enc_hidden_keys, enc_input_keys, enc_field_keys,
                     enc_hidden_vals, enc_input_vals, enc_field_vals,
                     no_dup_mask=None
                     ):
        # print('input_ids: {}'.format(input_ids.size()))
        # coverage, weighted_coverage = coverage
        logit_or_attn_scores = self._get_attn_scores(batch_size, max_enc_len, coverage, enc_mask, dec_hidden,
                                                     enc_hidden_keys, enc_input_keys, enc_field_keys,
                                                     self.attn_type, no_dup_mask=no_dup_mask)

        enc_output_context, enc_context_proj = self._get_contexts(logit_or_attn_scores,
                                                                  enc_hidden_vals, enc_input_vals, enc_field_vals)
        # print('enc_output_context: {}'.format(enc_output_context.size()))

        if self.context_mlp:
            enc_output_context = self.V1(enc_output_context)

        if self.decoder_type == 'pt':
            return logit_or_attn_scores[0], None, (None, None), enc_output_context  # only use the top-most emit vector
        else:
            attn_scores = logit_or_attn_scores

            out_vec = self.V(torch.cat((dec_hidden, enc_output_context), 1))

            attn_weights = attn_scores[0]

            if self.decoder_type == 'pg':
                p_vocab = F.softmax(out_vec, dim=1)
                # print('p_vocab: {}'.format(p_vocab.size()))

                if self.cov_in_pgen:
                    # print('coverage: {}'.format(coverage))
                    cov_mean = coverage.mean(dim=-1, keepdim=True)
                    # print('cov_mean: {}'.format(cov_mean))
                    enc_context_proj += cov_mean
                # print('enc_context_proj: {}'.format(enc_context_proj.size()))
                # print('dec_hidden: {}'.format(dec_hidden.size()))
                # print('decoder_input: {}'.format(decoder_input.size()))
                p_gen_logits = enc_context_proj + self.w_d(dec_hidden) + self.w_y(decoder_input)
                # print('p_gen_logits: {}'.format(p_gen_logits.size()))
                p_gen = torch.sigmoid(p_gen_logits).view(-1, 1)
                # print('p_gen: {}'.format(p_gen.size()))

                weighted_Pvocab = p_vocab * p_gen
                # print('weighted_Pvocab: {}'.format(weighted_Pvocab.size()))

                # print('attn_weights: {}'.format(attn_weights.size()))

                weighted_attn = (1-p_gen) * attn_weights  # * (1-weighted_coverage.clamp(0, 1))
                # print('weighted_attn: {}'.format(weighted_attn.size()))

                # print('max_tail_oov: {}'.format(max_tail_oov))
                if max_tail_oov > 0:
                    # create OOV (but in-article) zero vectors
                    ext_vocab = torch.zeros(batch_size, max_tail_oov)
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

                # print('input_ids: {}'.format(input_ids.size()))
                # print('weighted_attn: {}'.format(weighted_attn.size()))
                combined_vocab = combined_vocab.scatter_add(1, input_ids, weighted_attn)
                # print('combined_vocab: {}'.format(combined_vocab.size()))
                return combined_vocab, attn_weights, (p_gen, src_prob), enc_output_context

            elif self.decoder_type == 'seq':
                return out_vec, attn_weights, (None, None), enc_output_context

    def forward(self, max_tail_oov=0, targets=None, targets_ids=None, input_ids=None, no_dup_mask=None,
                enc_outputs=None, enc_keys=None, enc_vals=None, f_matrix=None, dec_state=None, coverage=None,
                forward_mode='train', w2fs=None, fig=False, batch_size=0, max_enc_len=0,
                chunk_len=0, chunk_start=0, dec_inp_chunk=None):
        """
            targets=batch_t, targets_ids=batch_o_t, input_ids=batch_o_s
        """
        # unpack everything from encoder side
        enc_hidden, enc_input, enc_field, enc_pos, enc_state, enc_masks = enc_outputs
        enc_seq_mask, _ = enc_masks
        enc_hidden_keys, enc_input_keys, enc_field_keys = enc_keys
        enc_hidden_vals, enc_input_vals, enc_field_vals = enc_vals
        # print('dec_state: {}'.format([x.size() for x in dec_state]))
        # print('dec_inp_chunk: {}'.format(dec_inp_chunk.size()))

        if forward_mode == 'pred':
            return self.evaluate(batch_size, max_tail_oov,
                                 f_matrix, dec_state, enc_masks, input_ids, coverage,
                                 enc_hidden_keys, enc_input_keys, enc_field_keys,
                                 enc_hidden_vals, enc_input_vals, enc_field_vals,
                                 dec_inp_chunk, no_dup_mask,
                                 max_enc_len, w2fs, fig)

        else:

            step_losses, cov_losses = [], []
            dec_lens = (targets > self.pad_id).float().sum(1)

            # pre-compute hidden states if not input feeding
            # TODO: ptr-net: probably need self-attention (transformer ???)
            if self.input_feeding:
                # initial input feeding
                enc_output_context = torch.zeros([batch_size, self.hidden_size]).cuda()
                # print('enc_output_context: {}'.format(enc_output_context.size()))
            else:
                dec_outs, dec_state = self.rnn(dec_inp_chunk, dec_state)
            # step through decoder hidden states
            for step in range(chunk_len):

                target_id = targets_ids[:, step+chunk_start+1]  # 0th token is <SOS>, [batch] of ids of next word
                # print(target_id)
                target_step_mask = target_id.eq(self.pad_id).detach()  # non-padding tokens
                # print(target_step_mask)

                decoder_input = dec_inp_chunk[:, step, :]
                # picking one time-step a time
                if self.input_feeding:
                    # print('enc_output_context: {}'.format(enc_output_context.size()))
                    # print('decoder_input: {}'.format(decoder_input.size()))
                    decoder_input_rnn = torch.cat([decoder_input.unsqueeze(1), enc_output_context.unsqueeze(1)], dim=-1)
                    dec_hidden, dec_state = self.rnn(decoder_input_rnn, dec_state)
                    dec_hidden = dec_hidden.squeeze(1)
                else:
                    dec_hidden = dec_outs[:, step, :]

                no_dup_mask_tensor = torch.from_numpy(no_dup_mask).cuda() if self.decoder_type == 'pt' else None

                logits_or_probs, attn_weights, _, enc_output_context = \
                    self._decode_step(batch_size, input_ids, coverage, max_tail_oov,
                                      dec_hidden, decoder_input,
                                      enc_seq_mask, max_enc_len,
                                      enc_hidden_keys, enc_input_keys, enc_field_keys,
                                      enc_hidden_vals, enc_input_vals, enc_field_vals,
                                      no_dup_mask=no_dup_mask_tensor)

                if self.decoder_type == 'pg':
                    combined_vocab = logits_or_probs
                    output = combined_vocab.gather(1, target_id.unsqueeze(1)).add_(sys.float_info.epsilon)
                    _step_loss = output.log().mul(-1).squeeze(1)

                    if self.use_cov_loss:
                        coverage = coverage + attn_weights
                        # coverage_norm = coverage/(step + 1)
                        # print('coverage: {}'.format(coverage.size()))
                        # take minimum across both attn_weights and coverage
                        _cov_loss, _ = torch.stack((coverage, attn_weights), 2).min(2)
                        # _cov_loss, _ = torch.stack((coverage_norm, attn_weights), 2).min(2)
                        # print('_cov_loss: {}'.format(_cov_loss.size()))
                        cov_losses.append(_cov_loss.sum(1))
                else:
                    logits = logits_or_probs
                    # print('target_id: {}'.format(target_id.size()))
                    # print('logits: {}'.format(logits.size()))
                    _step_loss = self.criterion(logits, target_id)

                _step_loss.masked_fill_(target_step_mask.data.byte(), 0)
                step_losses.append(_step_loss)

                # mask the chosen positions before next step
                if self.decoder_type == 'pt':
                    for x, y in zip(range(batch_size), target_id.tolist()):
                        no_dup_mask[x][y] = 1
                        # del no_dup_mask_tensor

            chunk_masked_loss = torch.stack(step_losses, 1).sum(1).mean()  # sum over tgt length, mean over batch
            # print('total_masked_loss: {}'.format(total_masked_loss.size()))
            if self.use_cov_loss:
                # TODO: change coverage loss to be the same as nll loss ???
                chunk_masked_loss = chunk_masked_loss + self.lmbda * torch.stack(cov_losses, 1).sum(1).div(dec_lens)
            return chunk_masked_loss, coverage, dec_state

    def evaluate(self, batch_size, max_tail_oov,
                 f_matrix, dec_state, enc_masks, input_ids, coverage,
                 enc_hidden_keys, enc_input_keys, enc_field_keys,
                 enc_hidden_vals, enc_input_vals, enc_field_vals,
                 decoder_input, no_dup_mask,
                 max_enc_len, w2fs, fig):

        # decoder_input here is dec_inp_chunk, which is [batch, 1, emb_size] during evaluation
        max_length = self.max_length
        enc_seq_mask, enc_non_stop_mask = enc_masks
        lengths = np.array([max_length] * batch_size)
        finished = np.array([False] * batch_size)
        losses = []
        decoded_outputs = []
        locations = [] if self.decoder_type == 'pt' else None
        src_probs = [] if self.decoder_type == 'pg' else None
        p_gens = [] if self.decoder_type == 'pg' else None
        attn = []

        if self.input_feeding:
            # initial input feeding
            enc_output_context = torch.zeros([batch_size, self.hidden_size]).cuda()
        else:
            decoder_input_rnn = decoder_input

        # step through decoder hidden states
        for step in range(max_length):
            if self.input_feeding:
                # print('enc_output_context: {}'.format(enc_output_context.size()))
                # print('decoder_input: {}'.format(decoder_input.size()))
                decoder_input_rnn = torch.cat([decoder_input, enc_output_context.unsqueeze(1)], dim=-1)
                # print('decoder_input_rnn: {}'.format(decoder_input_rnn.size()))

            dec_hidden, dec_state = self.rnn(decoder_input_rnn, dec_state)

            no_dup_mask_tensor = torch.from_numpy(no_dup_mask).cuda() if self.decoder_type == 'pt' else None

            logits_or_prob, attn_weights, (p_gen, src_prob), enc_output_context = \
                self._decode_step(batch_size, input_ids, coverage,
                                  max_tail_oov,
                                  dec_hidden.squeeze(1),
                                  decoder_input.squeeze(1),
                                  enc_seq_mask, max_enc_len,
                                  enc_hidden_keys, enc_input_keys,
                                  enc_field_keys,
                                  enc_hidden_vals, enc_input_vals,
                                  enc_field_vals,
                                  no_dup_mask=no_dup_mask_tensor)

            if self.decoder_type != 'pg':
                vocab_probs = F.softmax(logits_or_prob, dim=1)
                if self.decoder_type == 'pt':
                    attn_weights = vocab_probs.detach()
            else:
                vocab_probs = logits_or_prob

            if self.decoder_type != 'pt' and not self.unk_gen:
                vocab_probs[:, self.unk_id] = 0  # NOTE: not allow decoder to output UNK

            # mask the probability of <eos> when shorter than min_length
            if step < self.min_length:
                if self.decoder_type == 'pt':
                    vocab_probs.masked_fill_(enc_non_stop_mask.data.byte(), 0.0)
                else:
                    vocab_probs[:, 2] = 0

            probs, symbols_or_positions = vocab_probs.topk(1)  # greedy decoding: get word indices and probs

            # mask the chosen positions before next step
            if self.decoder_type == 'pt':
                for x, y in zip(range(batch_size), symbols_or_positions.squeeze(-1).tolist()):
                    no_dup_mask[x][y] = 1

            if self.decoder_type == 'pt':
                # print('input_ids: {}'.format(input_ids.size()))
                symbols = input_ids.gather(1, symbols_or_positions)
                positions = symbols_or_positions
            else:
                symbols = symbols_or_positions

            # [deprecated] from original source: not allow decoder to output UNK
            if self.mask:
                tmp_mask = torch.zeros_like(enc_mask, dtype=torch.uint8)
                for i in range(symbols.size(0)):
                    pos = (input_ids[i] == symbols[i]).nonzero()
                    if pos.size(0) != 0:
                        tmp_mask[i][pos] = 1
                enc_mask = torch.where(enc_mask > tmp_mask, enc_mask, tmp_mask)

            attn.append(attn_weights)
            decoded_outputs.append(symbols.clone())
            eos_batch = symbols.data.eq(self.eos_id)
            if eos_batch.dim() > 0:
                eos_batch = eos_batch.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batch) != 0
                lengths[update_idx] = len(decoded_outputs)
            # replace oov with the corresponding field embedding
            if self.decoder_type == 'pg':
                # print('symbols: {}'.format(symbols))
                # print('w2fs: {}'.format(w2fs))
                # print('w2fs: {}'.format(len(w2fs)))
                # print('symbols: {}'.format(symbols.size()))
                for i in range(symbols.size(0)):
                    w2f = w2fs[i]
                    if symbols[i].item() > self.vocab_size-1:
                        # symbols[i] = w2f[symbols[i].item()]
                        symbols[i] = self.unk_id  # TODO: w2f[symbols[i].item()] is UNK anyway, change if disallow UNK

            # symbols.masked_fill_((symbols > self.vocab_size-1), self.unk_id)
            if self.decoder_type == 'pt':
                locations.append(positions.clone())
                if self.ptr_input == 'hid':
                    word_indices = symbols_or_positions.unsqueeze(-1).expand(batch_size, 1, enc_hidden_vals.size(-1))
                    decoder_input = enc_hidden_vals.gather(1, word_indices)
                else:
                    word_indices = symbols_or_positions.unsqueeze(-1).expand(batch_size, 1, enc_input_vals.size(-1))
                    dec_word_input = enc_input_vals.gather(1, word_indices)
                    feat_indices = symbols_or_positions.unsqueeze(-1).expand(batch_size, 1, enc_field_vals.size(-1))
                    dec_field_input = enc_field_vals.gather(1, feat_indices)
                    decoder_input = torch.cat((dec_word_input, dec_field_input), dim=2)
                    if self.ptr_dec_feat and self.ptr_feat_merge == 'mlp':
                        decoder_input = self.dropout(self.input_mlp(decoder_input))

                # print('enc_input_vals: {}'.format(enc_input_vals.size()))
                # print('enc_field_vals: {}'.format(enc_field_vals.size()))
                # print('symbols_or_positions: {}'.format(symbols_or_positions.size()))
                # print('indices: {}'.format(indices.size()))
                # print('dec_word_input: {}'.format(dec_word_input.size()))
                # print('dec_field_input: {}'.format(dec_field_input.size()))
            else:
                locations = None
                decoder_input = self.embedding(symbols)

            if self.use_cov_loss:
                # weighted_coverage = weighted_coverage + attn_weights * p_gen
                coverage = coverage + attn_weights

            # record eval loss
            target_mask_step = symbols.ne(self.eos_id).squeeze(1).detach()
            probs = probs.add_(sys.float_info.epsilon)
            logits = probs.log()
            nll = logits.mul(-1)
            batch_loss = torch.masked_select(nll.squeeze(1), target_mask_step)
            losses.append(batch_loss)
            if self.decoder_type == 'pg':
                p_gens.append(p_gen.squeeze(1))
                src_probs.append(src_prob)

            # check if all samples finished at the eos token
            finished_step = np.logical_not(np.array(target_mask_step.cpu(), dtype=bool))
            finished = np.logical_or(finished, finished_step)
            # stop if all finished
            if all(finished): break

        if self.decoder_type == 'pg':
            p_gens = torch.stack(p_gens, 1)

        locations = torch.stack(locations, 1).squeeze(2) if locations is not None else None
        losses = torch.cat(losses)
        self_matrix = f_matrix if self.field_self_att and fig else None
        return torch.stack(decoded_outputs, 1).squeeze(2), locations, lengths.tolist(), losses.tolist(), p_gens, \
               self_matrix, torch.stack(attn, 1).squeeze(2)