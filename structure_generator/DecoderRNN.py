import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN
# self.word2idx['<UNK>'] = 1


class DecoderRNN(BaseRNN):

    def __init__(self, vocab_size, embedding, embed_size, pemsize, sos_id, eos_id, unk_id, 
                 rnn_cell='gru', directions=2,
                 attn_src='emb', attn_type='concat', attn_fuse='sum', attn_level=2,
                 use_cov_loss=True, use_cov_attn=True, field_self_att=False, field_concat_pos=False, mask=False,
                 use_cuda=True, n_layers=1, input_dropout_p=0, dropout_p=0, max_len=100, lmbda=1.5):
        
        hidden_size = embed_size

        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.attn_src = attn_src
        self.attn_type = attn_type
        self.attn_fuse = attn_fuse # TODO: unused
        self.attn_level = attn_level
        self.directions = directions
        self.use_cov_loss = use_cov_loss
        self.use_cov_attn = use_cov_attn
        self.field_self_att = field_self_att
        self.field_concat_pos = field_concat_pos
        # TODO: input feeding
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.mask = mask
        self.embedding = embedding
        self.lmbda = lmbda
        self.use_cuda = use_cuda

        # ----------------- params for directions ----------------- #
        self.W_enc_state = nn.Linear(hidden_size * 2, hidden_size)

        # ----------------- parameters for self attention ----------------- #
        self_size = pemsize * 2
        # print('self_size: {}'.format(self_size))
        self.Win = nn.Linear(self_size, self_size)
        self.Wout = nn.Linear(self_size, self_size)
        self.Wg = nn.Linear(self_size, self_size)

        # ----------------- params for attention ----------------- #
        enc_input_size = hidden_size * self.directions
        field_input_size = hidden_size
        # print('enc_input_size: {}'.format(enc_input_size))
        if self.attn_level > 1 and self.field_concat_pos:
            field_input_size += self_size
        # print('field_input_size: {}'.format(field_input_size))

        self.v = nn.Linear(hidden_size, 1)
        self.Wd = nn.Linear(hidden_size, hidden_size)  # e_t decoder current state
        self.Wf = nn.Linear(field_input_size, hidden_size)

        if self.attn_level == 3:
            self.We = nn.Linear(hidden_size, hidden_size)  # e_t: word embeddings
            self.Wr = nn.Linear(enc_input_size, hidden_size)  # e_t: encoder hidden states
        elif self.attn_level == 2:
            if self.attn_src == 'emb':
                self.We = nn.Linear(hidden_size, hidden_size)
            elif self.attn_src == 'rnn':
                self.Wr = nn.Linear(enc_input_size, hidden_size)
        else:
            # NOTE: assume to use encoder rnn hidden states when attn_level == 1
            self.Wr = nn.Linear(enc_input_size, hidden_size)

        if self.use_cov_loss:
            self.Wc = nn.Linear(1, hidden_size)  # e_t: coverage vector

        # ----------------- params for output ----------------- #
        output_layer_input_size = hidden_size  # decoder state size
        if self.attn_level == 3:
            output_layer_input_size += (hidden_size + enc_input_size + field_input_size)
        elif self.attn_level == 2:
            output_layer_input_size += field_input_size
            if self.attn_src == 'emb':
                output_layer_input_size += hidden_size
            if self.attn_src == 'rnn':
                output_layer_input_size += enc_input_size
        else:
            output_layer_input_size += enc_input_size

        # print('output_layer_input_size: {}'.format(output_layer_input_size))
        self.V = nn.Linear(output_layer_input_size, self.output_size)

        # ----------------- parameters for p_gen ----------------- #
        self.w_r = nn.Linear(enc_input_size, 1)     # encoder hidden context
        self.w_e = nn.Linear(hidden_size, 1)        # encoder word context
        self.w_f = nn.Linear(field_input_size, 1)   # encoder field context
        self.w_d = nn.Linear(hidden_size, 1)        # decoder hidden state
        self.w_y = nn.Linear(embed_size,  1)        # decoder input word embedding


    def _pos_self_attn(self, enc_pos, enc_hidden, enc_input, enc_field):
        """ compute the self-attentive encoder output and field encodings"""

        gin = torch.tanh(self.Win(enc_pos))
        gout = torch.tanh(self.Wout(enc_pos))
        f = gin.bmm(self.Wg(gout).transpose(1, 2))
        f_matrix = F.softmax(f, dim=2)

        enc_hidden_selfatt = torch.bmm(f_matrix, enc_hidden)
        enc_input_selfatt = torch.bmm(f_matrix, enc_input)
        enc_field_selfatt = torch.bmm(f_matrix, enc_field)

        return f_matrix, enc_hidden_selfatt, enc_input_selfatt, enc_field_selfatt

    def _get_enc_keys(self, enc_hidden, enc_input, enc_field, batch_size, max_enc_len):

        # TODO: simplify this chunk
        if self.attn_level == 3:
            enc_hidden_flat = enc_hidden.contiguous().view(batch_size * max_enc_len, -1)
            # print('enc_hidden_flat: {}'.format(enc_hidden_flat.size()))
            enc_hidden_keys = self.Wr(enc_hidden_flat).view(batch_size, max_enc_len, -1)
            # print('enc_hidden_keys: {}'.format(enc_hidden_keys.size()))

            enc_input_flat = enc_input.view(batch_size * max_enc_len, -1)
            # print('enc_input_flat: {}'.format(enc_input_flat.size()))
            enc_input_keys = self.We(enc_input_flat).view(batch_size, max_enc_len, -1)
            # print('enc_input_keys: {}'.format(enc_input_keys.size()))

        elif self.attn_level == 2:
            if self.attn_src == 'rnn':
                enc_hidden_flat = enc_hidden.contiguous().view(batch_size * max_enc_len, -1)
                # print('enc_hidden_flat: {}'.format(enc_hidden_flat.size()))
                enc_hidden_keys = self.Wr(enc_hidden_flat).view(batch_size, max_enc_len, -1)
                # print('enc_hidden_keys: {}'.format(enc_hidden_keys.size()))
                enc_input_keys = None
            elif self.attn_src == 'emb':
                enc_hidden_keys = None
                enc_input_flat = enc_input.view(batch_size * max_enc_len, -1)
                # print('enc_input_flat: {}'.format(enc_input_flat.size()))
                enc_input_keys = self.We(enc_input_flat).view(batch_size, max_enc_len, -1)
                # print('enc_input_keys: {}'.format(enc_input_keys.size()))

        else:
            enc_hidden_flat = enc_hidden.contiguous().view(batch_size * max_enc_len, -1)
            # print('enc_hidden_flat: {}'.format(enc_hidden_flat.size()))
            enc_hidden_keys = self.Wr(enc_hidden_flat).view(batch_size, max_enc_len, -1)
            # print('enc_hidden_keys: {}'.format(enc_hidden_keys.size()))
            enc_input_keys = None

        if self.attn_level > 1:
            enc_field_flat = enc_field.view(batch_size * max_enc_len, -1)
            # print('enc_field_flat: {}'.format(enc_field_flat.size()))
            enc_field_keys = self.Wf(enc_field_flat).view(batch_size, max_enc_len, -1)
            # print('enc_field_keys: {}'.format(enc_field_keys.size()))
        else:
            enc_field_keys = None

        return enc_hidden_keys, enc_input_keys, enc_field_keys

    def _attn_score(self, batch_size, max_enc_len, coverage, enc_mask,
                    dec_hidden, enc_hidden_keys, enc_input_keys, enc_field_keys):
        
        dec_query = self.Wd(dec_hidden)
        # print('dec_query [before]: {}'.format(dec_query.size()))
        if self.attn_src == 'emb':
            dec_query = dec_query.unsqueeze(1).expand_as(enc_input_keys)
        elif self.attn_src == 'rnn':
            dec_query = dec_query.unsqueeze(1).expand_as(enc_hidden_keys)

        # print('dec_query [after]: {}'.format(dec_query.size()))

        if self.attn_level == 3:
            enc_keys = enc_hidden_keys + enc_input_keys + enc_field_keys
        elif self.attn_level == 2:
            if self.attn_src == 'emb':
                enc_keys = enc_input_keys + enc_field_keys
            elif self.attn_src == 'rnn':
                enc_keys = enc_hidden_keys + enc_field_keys
        else:
            enc_keys = enc_hidden_keys
        # print('enc_keys: {}'.format(enc_keys.size()))

        attention_source = dec_query + enc_keys

        if self.use_cov_attn:
            cov_vector = self.Wc(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
            # print('cov_vector: {}'.format(cov_vector.size()))
            attention_source += cov_vector

        # print('attention_source: {}'.format(attention_source.size()))
        e_t = self.v(torch.tanh(attention_source).view(batch_size*max_enc_len, -1))
        # print('e_t: {}'.format(e_t.size()))

        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t

        # mask to -inf before applying softmax
        # print('attn_scores [before]: {}'.format(attn_scores))
        # print('enc_mask: {}'.format(enc_mask))
        attn_scores.data.masked_fill_(enc_mask.data.byte(), 0)
        attn_scores = F.softmax(attn_scores, dim=1)
        # print('attn_scores: {}'.format(attn_scores.size()))

        return attn_scores

    def _decode_step(self,
                     batch_size, input_ids, coverage, max_source_oov,
                     dec_hidden, decoder_input,
                     enc_mask, max_enc_len,
                     enc_hidden_keys, enc_input_keys, enc_field_keys,
                     enc_hidden_vals, enc_input_vals, enc_field_vals
                     ):
        # print('input_ids: {}'.format(input_ids.size()))

        attn_scores = self._attn_score(batch_size, max_enc_len, coverage, enc_mask,
                                       dec_hidden, enc_hidden_keys, enc_input_keys, enc_field_keys)

        if self.attn_level == 3:
            enc_hidden_context = attn_scores.unsqueeze(1).bmm(enc_hidden_vals).squeeze(1)
            enc_input_context = attn_scores.unsqueeze(1).bmm(enc_input_vals).squeeze(1)
            enc_field_context = attn_scores.unsqueeze(1).bmm(enc_field_vals).squeeze(1)
            # output
            enc_output_context = torch.cat((enc_hidden_context, enc_input_context, enc_field_context), 1)
            # p_gen
            enc_context_proj = self.w_r(enc_hidden_context) + self.w_e(enc_input_context) + self.w_f(enc_field_context)

        elif self.attn_level == 2:
            enc_field_context = attn_scores.unsqueeze(1).bmm(enc_field_vals).squeeze(1)
            if self.attn_src == 'emb':
                enc_input_context = attn_scores.unsqueeze(1).bmm(enc_input_vals).squeeze(1)
                # output
                enc_output_context = torch.cat((enc_input_context, enc_field_context), 1)
                # p_gen
                enc_context_proj = self.w_e(enc_input_context) + self.w_f(enc_field_context)

            elif self.attn_src == 'rnn':
                enc_hidden_context = attn_scores.unsqueeze(1).bmm(enc_hidden_vals).squeeze(1)
                # output
                enc_output_context = torch.cat((enc_hidden_context, enc_field_context), 1)
                # p_gen
                enc_context_proj = self.w_r(enc_hidden_context) + self.w_f(enc_field_context)

        else:
            # output
            enc_output_context = attn_scores.unsqueeze(1).bmm(enc_hidden_vals).squeeze(1)
            # p_gen
            enc_context_proj = self.w_r(enc_output_context)

        # print('enc_output_context: {}'.format(enc_output_context.size()))
        # print('enc_context_proj: {}'.format(enc_context_proj.size()))
        p_vocab = F.softmax(self.V(torch.cat((dec_hidden, enc_output_context), 1)), dim=1)
        # print('p_vocab: {}'.format(p_vocab.size()))

        p_gen = torch.sigmoid(enc_context_proj + self.w_d(dec_hidden) + self.w_y(decoder_input)).view(-1, 1)
        # print('p_gen: {}'.format(p_gen.size()))

        weighted_Pvocab = p_vocab * p_gen
        # print('weighted_Pvocab: {}'.format(weighted_Pvocab.size()))

        weighted_attn = (1-p_gen) * attn_scores
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
        combined_vocab = combined_vocab.scatter_add(1, input_ids, weighted_attn)
        # print('combined_vocab: {}'.format(combined_vocab.size()))

        return combined_vocab, attn_scores

    def forward(self, max_source_oov=0, targets=None, targets_id=None, input_ids=None,
                enc_hidden=None, enc_input=None, enc_state=None, enc_mask=None, enc_field=None, enc_pos=None,
                teacher_forcing_ratio=None, w2fs=None, fig=False):
        """
            target=batch_t, target_id=batch_o_t, input_ids=batch_o_s
        """

        targets, batch_size, max_length, max_enc_len = self._validate_args(targets, enc_state, enc_input, teacher_forcing_ratio)
        decoder_hidden_init = self._init_state(enc_state)
        if self.use_cov_loss:
            coverage = torch.zeros(batch_size, max_enc_len)
            if self.use_cuda:
                coverage = coverage.cuda()
        else:
            coverage = None

        enc_hidden_keys, enc_input_keys, enc_field_keys = \
            self._get_enc_keys(enc_hidden, enc_input, enc_field, batch_size, max_enc_len)

        # get position self-attention scores
        if self.field_self_att:
            f_matrix, enc_hidden_vals, enc_input_vals, enc_field_vals\
                = self._pos_self_attn(enc_pos, enc_hidden, enc_input, enc_field)
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
                # print('target_id: {}'.format(target_id.size()))

                dec_hidden = hidden[:, step, :]
                decoder_input = decoder_inputs[:, step, :]

                combined_vocab, attn_scores = self._decode_step(batch_size, input_ids, coverage, max_source_oov,
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
                    coverage = coverage + attn_scores
                    # print('coverage: {}'.format(coverage.size()))
                    # take minimum across both attn_scores and coverage
                    _cov_loss, _ = torch.stack((coverage, attn_scores), 2).min(2)
                    # print('_cov_loss: {}'.format(_cov_loss.size()))
                    cov_loss.append(_cov_loss.sum(1))

            # NOTE: loss is normalized by length
            # TODO: use sum of loss
            total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens)
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
        decoded_outputs = []
        if fig:
            attn = []
        decoder_input = self.embedding(targets)
        # step through decoder hidden states
        for step in range(max_length):
            dec_hidden, _c = self.rnn(decoder_input, decoder_hidden_init)
            combined_vocab, attn_scores = self._decode_step(batch_size, input_ids, coverage, max_source_oov,
                                                            dec_hidden.squeeze(1), decoder_input.squeeze(1),
                                                            enc_mask, max_enc_len,
                                                            enc_hidden_keys, enc_input_keys, enc_field_keys,
                                                            enc_hidden_vals, enc_input_vals, enc_field_vals)

            combined_vocab[:, self.unk_id] = 0  # NOTE: not allow decoder to output UNK
            symbols = combined_vocab.topk(1)[1]  # (values, indices)
            if self.mask:
                tmp_mask = torch.zeros_like(enc_mask, dtype=torch.uint8)
                for i in range(symbols.size(0)):
                    pos = (input_ids[i] == symbols[i]).nonzero()
                    if pos.size(0) != 0:
                        tmp_mask[i][pos] = 1
                enc_mask = torch.where(enc_mask > tmp_mask, enc_mask, tmp_mask)

            if fig:
                attn.append(attn_scores)
            decoded_outputs.append(symbols.clone())
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(decoded_outputs)
            # change unk to corresponding field
            for i in range(symbols.size(0)):
                w2f = w2fs[i]
                if symbols[i].item() > self.vocab_size-1:
                    symbols[i] = w2f[symbols[i].item()]
            # symbols.masked_fill_((symbols > self.vocab_size-1), self.unk_id)
            decoder_input = self.embedding(symbols)
            decoder_hidden_init = _c
            if self.use_cov_loss:
                coverage = coverage + attn_scores
        if fig:
            self_matrix = f_matrix[0] if self.field_self_att else None
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), self_matrix, \
                   torch.stack(attn, 1).squeeze(2)[0]
        else:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist()

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
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
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