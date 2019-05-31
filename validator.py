import torch, sys, os
from tqdm import tqdm

class Validator(object):

    def __init__(self, model, v_dataset, use_cuda):
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self.model.eval()  # switch to eval mode
        torch.set_grad_enabled(False)  # turn off gradient tracking
        self.v_dataset = v_dataset

    def valid_batch(self, batch_idx, src='full'):
        """ Run forward pass on valid set"""
        data_packages, _, remaining = self.v_dataset.get_batch(batch_idx)
        batch_size = len(remaining[0])
        batch_output = self.model(data_packages, remaining, forward_mode='valid', src=src)
        return batch_output, batch_size

    def valid(self, epoch, src='full'):

        if self.model.decoder_type == 'prn':
            valid_loss = {'prn-planner': (0.0, 0.0, 0.0), 'prn-realizer': (0.0, 0.0, 0.0)}
        else:
            valid_loss = {'{}'.format(self.model.decoder_type): (0.0, 0.0, 0.0)}

        num_valid_batch = len(self.v_dataset.corpus)
        num_valid_expls = self.v_dataset.len

        valid_losses = 0
        valid_switch_losses = 0
        valid_table_fill_losses = 0
        print("{} batches to be evaluated".format(num_valid_batch))
        for batch_idx in tqdm(range(num_valid_batch)):
            batch_output, batch_size = self.valid_batch(batch_idx, src=src)
            for mdl, outputs in batch_output.items():
                batch_loss_bundle, _ = outputs  # total_norm is ignored
                batch_loss, batch_switch_loss, batch_table_fill_loss = batch_loss_bundle
                valid_losses += batch_loss * batch_size
                if batch_switch_loss is not None:
                    valid_switch_losses += batch_switch_loss * batch_size
                if batch_table_fill_loss is not None:
                    valid_table_fill_losses += batch_table_fill_loss * batch_size

        valid_loss[mdl] = (valid_losses/num_valid_expls,
                           valid_switch_losses/num_valid_expls,
                           valid_table_fill_losses/num_valid_expls)

        return valid_loss
