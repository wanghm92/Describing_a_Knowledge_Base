import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN
# self.word2idx['<UNK>'] = 1


class DecoderRNN(BaseRNN):

    def __init__(self, vocab_size, embedding, embed_size, pemsize, sos_id, eos_id, unk_id,
                 hidden_type='emb', max_len=100, n_layers=1, rnn_cell='gru', bidirectional=True,
                 input_dropout_p=0, dropout_p=0, lmbda=1.5, USE_CUDA = torch.cuda.is_available(), mask=0):
        
        # NOTE
        hidden_size = embed_size
        
        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.hidden_type = hidden_type
        self.bidirectional_encoder = bidirectional
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
        self.USE_CUDA = USE_CUDA

        # ----------------- params for directions ----------------- #
        self.W_dir = nn.Linear(hidden_size * 2, hidden_size)

        # ----------------- params for output ----------------- #
        multiplier = 3
        if self.hidden_type == 'rnn':
            multiplier += 1
        if self.hidden_type == 'both':
            multiplier += 2
        self.V = nn.Linear(hidden_size * multiplier, self.output_size)

        # ----------------- params for attention ----------------- #
        # for obtaining e from encoder hidden
        input_shape = hidden_size
        if self.hidden_type != 'emb':
            if self.bidirectional_encoder:
                input_shape *= 2
            if self.hidden_type == 'both':
                input_shape += hidden_size
        self.We = nn.Linear(input_shape, hidden_size)

        self.Wf = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder field
        self.Wh = nn.Linear(hidden_size, hidden_size)  # for obtaining e from decoder current state
        self.Wc = nn.Linear(1, hidden_size)  # for obtaining e from context vector
        self.v = nn.Linear(hidden_size, 1)

        # ----------------- parameters for p_gen ----------------- #
        self.w_e = nn.Linear(input_shape, 1)    # for changing context vector into a scalar
        self.w_f = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_h = nn.Linear(hidden_size, 1)    # for changing hidden state into a scalar
        self.w_y = nn.Linear(embed_size, 1)     # for changing input embedding into a scalar

        # ----------------- parameters for self attention ----------------- #
        self_size = pemsize * 2  # hidden_size +
        self.Win = nn.Linear(self_size, self_size)
        self.Wout = nn.Linear(self_size, self_size)
        self.Wg = nn.Linear(self_size, self_size)

    def get_matrix(self, enc_pos):
        gin = torch.tanh(self.Win(enc_pos))
        gout = torch.tanh(self.Wout(enc_pos))
        f = gin.bmm(self.Wg(gout).transpose(1, 2))
        return F.softmax(f, dim=2)

    def self_attn(self, f_matrix, enc_output, enc_field):
        enc_output_selfatt = torch.bmm(f_matrix, enc_output)
        enc_field_selfatt = torch.bmm(f_matrix, enc_field)
        return enc_output_selfatt, enc_field_selfatt

    def decode_step(self, input_ids, coverage, dec_hidden, enc_proj, batch_size, max_enc_len,
                    enc_mask, enc_output_selfatt, enc_field_selfatt, embed_input, max_source_oov, f_matrix):
        dec_proj = self.Wh(dec_hidden).unsqueeze(1).expand_as(enc_proj)
        # print('dec_proj: {}'.format(dec_proj.size()))

        cov_proj = self.Wc(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
        e_t = self.v(torch.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))

        # mask to -INF before applying softmax
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
        attn_scores.data.masked_fill_(enc_mask.data.byte(), 0)
        attn_scores = F.softmax(attn_scores, dim=1)
        # print('attn_scores: {}'.format(attn_scores.size()))

        enc_output_context = attn_scores.unsqueeze(1).bmm(enc_output_selfatt).squeeze(1)
        enc_field_context = attn_scores.unsqueeze(1).bmm(enc_field_selfatt).squeeze(1)

        # output proj calculation
        p_vocab = F.softmax(self.V(torch.cat((dec_hidden, enc_output_context, enc_field_context), 1)), dim=1)
        # print('p_vocab: {}'.format(p_vocab.size()))
        # p_gen calculation
        p_gen = torch.sigmoid(self.w_e(enc_output_context) + self.w_f(enc_field_context) + self.w_h(dec_hidden) + self.w_y(embed_input))
        p_gen = p_gen.view(-1, 1)
        # print('p_gen: {}'.format(p_gen.size()))
        weighted_Pvocab = p_vocab * p_gen
        # print('weighted_Pvocab: {}'.format(weighted_Pvocab.size()))
        weighted_attn = (1-p_gen) * attn_scores
        # print('weighted_attn: {}'.format(weighted_attn.size()))

        if max_source_oov > 0:
            # create OOV (but in-article) zero vectors
            ext_vocab = torch.zeros(batch_size, max_source_oov)
            if self.USE_CUDA:
                ext_vocab=ext_vocab.cuda()
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        del weighted_Pvocab       # 'Recheck OOV indexes!'

        # scatter article word probs to combined vocab prob.
        # print('input_ids: () {}'.format(input_ids.size(), input_ids))
        combined_vocab = combined_vocab.scatter_add(1, input_ids, weighted_attn)
        # print('combined_vocab: {}'.format(combined_vocab.size()))

        return combined_vocab, attn_scores

    def forward(self, max_source_oov=0, targets=None, targets_id=None, input_ids=None,
                enc_hidden=None, enc_input=None, enc_state=None, enc_mask=None, enc_field=None, enc_pos=None,
                teacher_forcing_ratio=None, w2fs=None, fig=False):
        # TODO: add flag for optional field encodings to run pointer-generator baseline

        targets, batch_size, max_length, max_enc_len = self._validate_args(targets, enc_state, enc_input, teacher_forcing_ratio)
        decoder_hidden = self._init_state(enc_state)
        coverage = torch.zeros(batch_size, max_enc_len)
        if self.USE_CUDA:
            coverage = coverage.cuda()

        if self.hidden_type == 'emb':
            enc_output = enc_input
        elif self.hidden_type == 'rnn':
            enc_output = enc_hidden
        else:
            enc_output = torch.cat((enc_hidden, enc_input), -1)

        enc_output_proj = self.We(enc_output.contiguous().view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)

        enc_field_proj = self.Wf(enc_field.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        enc_proj = enc_output_proj + enc_field_proj

        # get link attention scores
        f_matrix = self.get_matrix(enc_pos)
        enc_output_selfatt, enc_field_selfatt = self.self_attn(f_matrix, enc_output, enc_field)

        if teacher_forcing_ratio:
            lm_loss, cov_loss = [], []
            dec_lens = (targets > 0).float().sum(1)

            embedded = self.embedding(targets)
            embed_inputs = self.input_dropout(embedded)

            hidden, _ = self.rnn(embed_inputs, decoder_hidden)

            # step through decoder hidden states
            for step in range(max_length):
                target_id = targets_id[:, step+1].unsqueeze(1)
                # print('target_id: {}'.format(target_id.size()))

                dec_hidden = hidden[:, step, :]
                embed_input = embed_inputs[:, step, :]

                combined_vocab, attn_scores = self.decode_step(input_ids, coverage, dec_hidden, enc_proj, batch_size,
                                                               max_enc_len, enc_mask, enc_output_selfatt, enc_field_selfatt,
                                                               embed_input, max_source_oov, f_matrix)
                # mask the output to account for PAD
                target_mask_0 = target_id.ne(0).detach()
                output = combined_vocab.gather(1, target_id).add_(sys.float_info.epsilon)
                # print('output: {}'.format(output.size()))
                _lm_loss = output.log().mul(-1) * target_mask_0.float()
                # print('_lm_loss: {}'.format(_lm_loss.size()))
                lm_loss.append(_lm_loss)

                coverage = coverage + attn_scores
                # print('coverage: {}'.format(coverage.size()))
                # Coverage Loss
                # take minimum across both attn_scores and coverage
                _cov_loss, _ = torch.stack((coverage, attn_scores), 2).min(2)
                # print('_cov_loss: {}'.format(_cov_loss.size()))
                cov_loss.append(_cov_loss.sum(1))

            # add individual losses
            total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens) + self.lmbda * \
                torch.stack(cov_loss, 1).sum(1).div(dec_lens)
            return total_masked_loss
        else:
            return self.evaluate(targets, batch_size, max_length, max_source_oov, enc_output_selfatt, enc_field_selfatt,
                                 f_matrix, decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig)

    def evaluate(self, targets, batch_size, max_length, max_source_oov, enc_output_selfatt, enc_field_selfatt, f_matrix,
                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig):
        lengths = np.array([max_length] * batch_size)
        decoded_outputs = []
        if fig:
            attn = []
        embed_input = self.embedding(targets)
        # step through decoder hidden states
        for step in range(max_length):
            dec_hidden, _c = self.rnn(embed_input, decoder_hidden)
            combined_vocab, attn_scores = self.decode_step(input_ids, coverage,
                                                           dec_hidden.squeeze(1), enc_proj, batch_size, max_enc_len,
                                                           enc_mask, enc_output_selfatt, enc_field_selfatt,
                                                           embed_input.squeeze(1), max_source_oov, f_matrix)
            # NOTE: not allow decoder to output UNK
            combined_vocab[:, self.unk_id] = 0
            symbols = combined_vocab.topk(1)[1]
            if self.mask == 1:
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
            embed_input = self.embedding(symbols)
            decoder_hidden = _c
            coverage = coverage + attn_scores
        if fig:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), f_matrix[0], \
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
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            h = self.W_dir(h)
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
            if self.USE_CUDA:
                targets = targets.cuda()
            max_length = self.max_length
        else:
            max_length = targets.size(1) - 1     # minus the start of sequence symbol

        return targets, batch_size, max_length, max_enc_len