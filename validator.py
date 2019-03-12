import torch, sys, os
from tqdm import tqdm

class Validator(object):

    def __init__(self, model, v_dataset, use_cuda, tfr):
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.v_dataset = v_dataset
        self.teacher_forcing_ratio = tfr

    def valid_batch(self, batch_idx):
        """ Run forward pass on valid set"""

        data_packages, _, remaining = self.v_dataset.get_batch(batch_idx)
        mean_batch_loss = self.model(data_packages, remaining, teacher_forcing_ratio=self.teacher_forcing_ratio)
        # TODO:
            # (1) use weighting factor
            # (2) return and log two losses
        if isinstance(mean_batch_loss, tuple):
            mean_batch_loss = mean_batch_loss[0] + mean_batch_loss[1]
        return mean_batch_loss.item(), len(remaining[0])

    def valid(self):
        torch.set_grad_enabled(False)

        valid_loss = 0.0
        num_valid_batch = len(self.v_dataset.corpus)
        num_valid_expls = self.v_dataset.len

        print("{} batches to be evaluated".format(num_valid_batch))
        for batch_idx in tqdm(range(num_valid_batch)):
            mean_batch_loss, batch_size = self.valid_batch(batch_idx)
            valid_loss += mean_batch_loss*batch_size
        valid_loss /= num_valid_expls

        return valid_loss
