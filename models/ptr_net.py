import torch.nn as nn

class PointerNet(nn.Module):

    def __init__(self, encoder, decoder):
        super(PointerNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert self.decoder.decoder_type == 'pt'

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def unpack_batch_data(self, data_packages, remaining, teacher_forcing_ratio=None):
        source_package, outline_package, _, _ = data_packages
        batch_s, batch_o_s, batch_f, batch_pf, batch_pb = source_package
        if teacher_forcing_ratio:
            batch_t, batch_o_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t = outline_package
            batch_t = (batch_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t)
        else:
            batch_t, batch_o_t = None, None

        source_len, outline_len, _, max_tail_oov, w2fs, _ = remaining

        return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_tail_oov, w2fs


    def forward(self, data_packages, remaining, teacher_forcing_ratio=None, fig=False):
        """
            batch_data: input word feature index tensors and output word index features
            remaining: source_len, outline_len, summary_len, max_tail_oov, w2fs, batch_idx2oov
        """

        batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_tail_oov, w2fs = \
            self.unpack_batch_data(data_packages, remaining, teacher_forcing_ratio)

        enc_hidden, enc_input, enc_field, enc_pos, enc_state, enc_masks = \
            self.encoder(batch_s, batch_f, batch_pf, batch_pb, input_lengths=source_len)

        result = self.decoder(max_tail_oov=max_tail_oov,
                              targets=batch_t,
                              targets_id=batch_o_t,
                              input_ids=batch_o_s,
                              enc_hidden=enc_hidden,
                              enc_input=enc_input,
                              enc_state=enc_state,
                              enc_masks=enc_masks,
                              enc_field=enc_field,
                              enc_pos=enc_pos,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              w2fs=w2fs,
                              fig=fig)
        return result
