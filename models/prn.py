import torch.nn as nn

class PRN(nn.Module):

    def __init__(self, encoder_all, encoder_otl, planner, realizer):
        super(PRN, self).__init__()
        self.encoder_all = encoder_all
        self.encoder_otl = encoder_otl
        self.planner = planner
        self.realizer = realizer

    def flatten_parameters(self):
        self.encoder_all.rnn.flatten_parameters()
        self.encoder_otl.rnn.flatten_parameters()
        self.planner.rnn.flatten_parameters()
        self.realizer.rnn.flatten_parameters()

    def unpack_batch_data(self, data_packages, remaining, teacher_forcing_ratio=None):
        source_package, outline_package, outline_pkg_rev, summary_package = data_packages
        batch_s, batch_o_s, batch_f, batch_pf, batch_pb = source_package
        source_len, outline_len, summary_len, max_tail_oov, w2fs, _ = remaining

        if teacher_forcing_ratio:
            batch_t, batch_o_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t = outline_package
            batch_t = (batch_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t)

            # batch_t_r, batch_o_t_r, batch_f_t_r, batch_pf_t_r, batch_pb_t_r = outline_pkg_rev
            # batch_t_r = (batch_t_r, batch_f_t_r, batch_pf_t_r, batch_pb_t_r)

            batch_sum, batch_o_sum = summary_package
        else:
            batch_t, batch_o_t = None, None
            # batch_t_r, batch_o_t_r = None, None
            batch_sum, batch_o_sum = None, None

        return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, \
               batch_t, batch_o_t, \
               batch_sum, batch_o_sum, \
               source_len, outline_len, summary_len, max_tail_oov, w2fs, _
               # batch_t_r, batch_o_t_r, \

    def forward(self, data_packages, remaining, teacher_forcing_ratio=None, fig=False):
        """
            batch_data: input word feature index tensors and output word index features
            remaining: source_len, outline_len, summary_len, max_tail_oov, w2fs, batch_idx2oov
        """
        # batch_t_r, batch_o_t_r, \
        batch_s, batch_o_s, batch_f, batch_pf, batch_pb, \
        batch_t, batch_o_t, \
        batch_sum, batch_o_sum, \
        source_len, outline_len, summary_len, max_tail_oov, w2fs, _ = self.unpack_batch_data(data_packages, remaining,
                                                                                             teacher_forcing_ratio)

        # Stage 1
        enc_hidden, enc_input, enc_field, enc_pos, enc_state, enc_masks = \
            self.encoder_all(batch_s, batch_f, batch_pf, batch_pb, input_lengths=source_len)

        planner_output = self.planner(max_tail_oov=0,
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
                                      fig=fig)

        if teacher_forcing_ratio:
            planner_loss = planner_output
            realizer_loss = self.cont_fn(batch_t[:-1], batch_sum, batch_o_sum, batch_o_t, teacher_forcing_ratio,
                                         max_tail_oov, outline_len, w2fs, fig)
            return planner_loss, realizer_loss
        else:
            return planner_output

    def cont_fn(self, batch_t, batch_sum=None, batch_o_sum=None, batch_o_t=None,
                teacher_forcing_ratio=None, max_tail_oov=None, outline_len=None, w2fs=None, fig=None):

        # Stage 2
        batch_t, batch_f_t, batch_pf_t, batch_pb_t = (x[:, :max(outline_len)] for x in batch_t)

        pln_hidden, pln_input, pln_field, pln_pos, pln_state, pln_masks = \
            self.encoder_otl(batch_t, batch_f_t, batch_pf_t, batch_pb_t, input_lengths=outline_len)

        # print('batch_t: {}'.format(batch_t.size()))
        # print('batch_f_t: {}'.format(batch_f_t.size()))
        # print('batch_pf_t: {}'.format(batch_pf_t.size()))
        # print('batch_pb_t: {}'.format(batch_pb_t.size()))
        # print('batch_lab_t: {}'.format(batch_lab_t.size()))
        # print('pln_hidden: {}'.format(pln_hidden.size()))
        # print('pln_input: {}'.format(pln_input.size()))
        # print('pln_field: {}'.format(pln_field.size()))
        # print('pln_pos: {}'.format(pln_pos.size()))
        # for x in pln_state:
        #     print('x: {}'.format(x.size()))
        # for y in pln_masks:
        #     print('y: {}'.format(y.size()))

        batch_o_t = batch_o_t[:, :max(outline_len)]

        realizer_output = self.realizer(max_tail_oov=max_tail_oov,
                                        targets=batch_sum,
                                        targets_id=batch_o_sum,
                                        input_ids=batch_o_t,
                                        enc_hidden=pln_hidden,
                                        enc_input=pln_input,
                                        enc_state=pln_state,
                                        enc_masks=pln_masks,
                                        enc_field=pln_field,
                                        enc_pos=pln_pos,
                                        teacher_forcing_ratio=teacher_forcing_ratio,
                                        w2fs=w2fs,
                                        fig=fig)

        return realizer_output