import torch, sys, os
from tqdm import tqdm

class Validator(object):

    def __init__(self, model, v_dataset, use_cuda):
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self.model.eval()  # switch to eval mode
        torch.set_grad_enabled(False)  # turn off gradient tracking
        self.v_dataset = v_dataset

    def valid_batch(self, batch_idx):
        """ Run forward pass on valid set"""
        data_packages, _, remaining = self.v_dataset.get_batch(batch_idx)
        batch_size = len(remaining[0])
        batch_output = self.model(data_packages, remaining, forward_mode='valid')
        return batch_output, batch_size

    def valid(self, epoch):

        if self.model.decoder_type == 'prn':
            valid_loss = {'prn-planner': 0.0, 'prn-realizer': 0.0}
        else:
            valid_loss = {'{}'.format(self.model.decoder_type): 0.0}

        num_valid_batch = len(self.v_dataset.corpus)
        num_valid_expls = self.v_dataset.len

        print("{} batches to be evaluated".format(num_valid_batch))
        for batch_idx in tqdm(range(num_valid_batch)):
            batch_output, batch_size = self.valid_batch(batch_idx)

            for mdl, outputs in batch_output.items():
                mean_batch_loss, _ = outputs  # total_norm is ignored
                valid_loss[mdl] += mean_batch_loss * batch_size

        return valid_loss, num_valid_expls
