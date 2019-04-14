import torch.nn as nn
import torch.optim as optim
from utils.miscs import Scheduler

class PRN(nn.Module):

    def __init__(self, planner, realizer, config):
        super(PRN, self).__init__()
        self.planner = planner
        self.realizer = realizer
        self.optimizer = (self.planner.optimizer, self.realizer.optimizer)
        self.scheduler = Scheduler(config, self.optimizer)
        self.batch_size = 1
        self.max_enc_len = 0
        self.max_length = 1
        self.decoder_type = 'prn'

    def flatten_parameters(self):
        self.planner.flatten_parameters()
        self.realizer.flatten_parameters()

    def unpack_batch_data(self, data_packages, remaining, forward_mode='train'):
        source_package, outline_package, outline_pkg_rev, summary_package = data_packages
        batch_s, batch_o_s, batch_f, batch_pf, batch_pb = source_package
        source_len, outline_len, summary_len, max_tail_oov, w2fs, _ = remaining

        if forward_mode != 'pred':
            batch_t, batch_o_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t = outline_package
            batch_t = (batch_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t)

            batch_sum, batch_o_sum = summary_package
        else:
            batch_t, batch_o_t = None, None
            batch_sum, batch_o_sum = None, None

        return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, \
               batch_t, batch_o_t, \
               batch_sum, batch_o_sum, \
               source_len, outline_len, summary_len, max_tail_oov, w2fs

    def forward(self, data_packages, remaining, forward_mode='train', fig=False, src=None):
        """
            batch_data: input word feature index tensors and output word index features
            remaining: source_len, outline_len, summary_len, max_tail_oov, w2fs, batch_idx2oov
        """

        planner_output = self.planner(data_packages, remaining, forward_mode=forward_mode)
        realizer_output = self.realizer(data_packages, remaining, forward_mode=forward_mode, src='outline')
        if forward_mode != 'pred':
            return {'prn-planner': list(planner_output.values())[0],
                    'prn-realizer': list(realizer_output.values())[0]}
