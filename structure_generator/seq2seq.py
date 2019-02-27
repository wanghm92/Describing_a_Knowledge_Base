import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, batch_s, batch_o_s, batch_f, batch_pf, batch_pb,
                target=None, target_id=None,
                input_lengths=None, max_source_oov=0,
                teacher_forcing_ratio=0,
                w2fs=None, fig=False):
        # target=batch_t, target_id=batch_o_t

        # enc_hidden, embed_input, embed_field, embed_pos, enc_state, mask
        enc_hidden, enc_input, enc_field, enc_pos, enc_state, enc_masks = \
            self.encoder(batch_s, batch_f, batch_pf, batch_pb, input_lengths)

        result = self.decoder(max_source_oov=max_source_oov,
                              targets=target,
                              targets_id=target_id,
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
