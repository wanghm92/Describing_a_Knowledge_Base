import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.miscs import Scheduler, detach_state

class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder, config):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)
        self.scheduler = Scheduler(config, self.optimizer)
        self.tbptt = config.tbptt
        self.max_grad_norm = config.max_grad_norm
        self.batch_size = 1
        self.max_enc_len = 0
        self.max_length = 1
        self.decoder_type = self.decoder.decoder_type

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def unpack_batch_data(self, data_packages, remaining, forward_mode='train'):
        source_package, _, _, summary_package = data_packages
        source_len, _, summary_len, max_tail_oov, w2fs, _ = remaining

        batch_s, batch_o_s, batch_f, batch_pf, batch_pb = source_package
        if forward_mode != 'pred':
            batch_sum, batch_o_sum = summary_package
            self.max_length = batch_sum.size(1) - 1  # minus the <SOS>
        else:
            batch_sum, batch_o_sum = None, None
            self.max_length = 1

        self.batch_size, self.max_enc_len = batch_s.size()

        return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_sum, batch_o_sum, source_len, max_tail_oov, w2fs

    def coverage_init(self):
        if self.decoder.use_cov_loss or self.decoder.use_cov_attn:
            coverage = torch.zeros(self.batch_size, self.max_enc_len).cuda()
        else:
            coverage = None
        return coverage

    def targets_init(self):
        return torch.LongTensor([self.decoder.sos_id] * self.batch_size).view(self.batch_size, 1).cuda()

    def forward(self, data_packages, remaining, forward_mode='train', fig=False):
        """
            batch_data: input word feature index tensors and output word index features
            remaining: source_len, outline_len, summary_len, max_tail_oov, w2fs, batch_idx2oov
        """

        batch_s, batch_o_s, batch_f, batch_pf, batch_pb, \
        batch_sum, batch_o_sum, \
        source_len, max_tail_oov, w2fs = \
            self.unpack_batch_data(data_packages, remaining, forward_mode)

        enc_outputs, enc_keys, enc_vals, f_matrix, dec_state = self.encoder(batch_s, batch_f, batch_pf, batch_pb,
                                                                            input_lengths=source_len)

        coverage = self.coverage_init()
        decoder_inputs = self.decoder.embedding(batch_sum) if forward_mode != 'pred' else None
        tbptt = self.tbptt if forward_mode == 'train' and self.tbptt > 0 else self.max_length

        total_norm = 0.0
        chunk_losses = []
        for batch, chunk_start in enumerate(range(0, self.max_length, tbptt)):
            if forward_mode != 'pred':
                chunk_len = min(tbptt, self.max_length - chunk_start)
                targets = batch_sum[:, chunk_start:chunk_start + chunk_len]
                dec_inp_chunk = decoder_inputs[:, chunk_start:chunk_start + chunk_len, :]
                dec_outs, dec_state = self.decoder.rnn(dec_inp_chunk, dec_state)
            else:
                chunk_len = 0
                targets = self.targets_init()
                dec_inp_chunk = self.decoder.embedding(targets)
                dec_outs, dec_state = None, None

            result = self.decoder(max_tail_oov=max_tail_oov,
                                  targets=targets,
                                  targets_ids=batch_o_sum,
                                  input_ids=batch_o_s,
                                  enc_outputs=enc_outputs,
                                  enc_keys=enc_keys,
                                  enc_vals=enc_vals,
                                  f_matrix=f_matrix,
                                  dec_state=dec_state,
                                  coverage=coverage,
                                  forward_mode=forward_mode,
                                  w2fs=w2fs,
                                  fig=fig,
                                  batch_size=self.batch_size,
                                  max_enc_len=self.max_enc_len,
                                  chunk_start=chunk_start,
                                  chunk_len=chunk_len,
                                  dec_inp_chunk=dec_inp_chunk,
                                  dec_outs=dec_outs)

            if forward_mode != 'pred':
                dec_state = detach_state(dec_state)
                mean_batch_loss, coverage = result
                chunk_losses.append(mean_batch_loss)
                if forward_mode == 'train':
                    total_norm = self._backprop(mean_batch_loss, total_norm)

        if forward_mode != 'pred':
            result = (sum(chunk_losses), total_norm)
        return {'{}'.format(self.decoder_type): result}

    def _backprop(self, loss, total_norm):
        self.optimizer.zero_grad()
        loss.backward()
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return total_norm