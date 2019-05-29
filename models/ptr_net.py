import torch
import numpy as np
from models.seq2seq import Seq2seq

class PointerNet(Seq2seq):

    def unpack_batch_data(self, data_packages, remaining, forward_mode='train'):
        source_package, outline_package, _, _ = data_packages
        source_len, outline_len, _, max_tail_oov, w2fs, _ = remaining

        batch_s, batch_o_s, batch_f, batch_pf, batch_pb = source_package
        if forward_mode != 'pred':
            batch_t, _, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t = outline_package
            batch_t = (batch_t, batch_f_t, batch_pf_t, batch_pb_t)
            batch_o_t = batch_lab_t
            self.max_length = batch_lab_t.size(1) - 1  # minus the <SOS>
        else:
            batch_t, batch_o_t = None, None
            self.max_length = 0

        self.batch_size, self.max_enc_len = batch_s.size()

        return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_tail_oov, w2fs

    def targets_init(self):
        if self.decoder.ptr_input == 'hid':
            targets = [torch.LongTensor([0] * self.batch_size).view(self.batch_size, 1).cuda()]
        else:
            targets = torch.LongTensor([self.decoder.sos_id] * self.batch_size).view(self.batch_size, 1).cuda()
            if self.decoder.ptr_dec_feat:
                fields = torch.LongTensor([self.decoder.sos_id] * self.batch_size).view(self.batch_size, 1).cuda()
                pos = torch.LongTensor([self.decoder.sos_id] * self.batch_size).view(self.batch_size, 1).cuda()
                rpos = torch.LongTensor([self.decoder.sos_id] * self.batch_size).view(self.batch_size, 1).cuda()
                targets = [targets, fields, pos, rpos]

        return targets

    def forward(self, data_packages, remaining, forward_mode='train', fig=False, retain_graph=False, src='full'):
        """
            batch_data: input word feature index tensors and output word index features
            remaining: source_len, outline_len, summary_len, max_tail_oov, w2fs, batch_idx2oov
        """

        batch_s, batch_o_s, batch_f, batch_pf, batch_pb, \
        batch_t, batch_o_t, \
        source_len, max_tail_oov, w2fs = \
            self.unpack_batch_data(data_packages, remaining, forward_mode)

        enc_outputs, enc_keys, enc_vals, f_matrix, dec_state = self.encoder(batch_s, batch_f, batch_pf, batch_pb,
                                                                            input_lengths=source_len)

        no_dup_mask = np.zeros((self.batch_size, self.max_enc_len), dtype=np.float32)
        no_dup_mask[:, 0] = 1  # start from making the 0th <SOS> token

        if forward_mode == 'pred':
            targets = self.targets_init()
            targets_ids = targets[0]
            dec_outs = None
        else:
            targets = batch_t
            targets_ids = batch_o_t

        if self.decoder.ptr_input == 'hid':
            targets = targets[0]
            enc_hidden = enc_vals[0]
            tgt_indices = targets_ids.unsqueeze(-1).expand(self.batch_size, self.max_length + 1, enc_hidden.size(-1))
            decoder_inputs = enc_hidden.gather(1, tgt_indices)
        else:
            targets, f_t, pf_t, pb_t = targets
            embedded = self.decoder.embedding(targets)
            embed_field = self.decoder.field_embedding(f_t)
            embed_pf = self.decoder.pos_embedding(pf_t)
            embed_pb = self.decoder.rpos_embedding(pb_t)
            embed_pos = torch.cat((embed_pf, embed_pb), dim=2)
            embed_field_pos = torch.cat((embed_field, embed_pos), dim=2)
            decoder_inputs = torch.cat((embedded, embed_field_pos), dim=2)
            if self.decoder.ptr_dec_feat and self.decoder.ptr_feat_merge == 'mlp':
                decoder_inputs = self.decoder.dropout(self.decoder.input_mlp(decoder_inputs))

        # if forward_mode != 'pred':
        #     dec_outs, dec_state = self.decoder.rnn(decoder_inputs, dec_state)

        total_norm = 0.0
        result = self.decoder(max_tail_oov=max_tail_oov,
                              targets=targets,
                              targets_ids=targets_ids,
                              input_ids=batch_o_s,
                              no_dup_mask=no_dup_mask,
                              enc_outputs=enc_outputs,
                              enc_keys=enc_keys,
                              enc_vals=enc_vals,
                              f_matrix=f_matrix,
                              dec_state=dec_state,
                              forward_mode=forward_mode,
                              w2fs=w2fs,
                              fig=fig,
                              batch_size=self.batch_size,
                              max_enc_len=self.max_enc_len,
                              chunk_len=self.max_length,
                              dec_inp_chunk=decoder_inputs)

        if forward_mode != 'pred':
            mean_batch_loss, _, _ = result
            if forward_mode == 'train':
                total_norm = self._backprop(mean_batch_loss, total_norm, retain_graph=retain_graph)

        if forward_mode != 'pred':
            result = (mean_batch_loss, total_norm)

        return {'{}'.format(self.decoder_type): result}