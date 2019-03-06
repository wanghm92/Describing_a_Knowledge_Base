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
        source_len = remaining[0]
        losses = self.model(data_packages, remaining, teacher_forcing_ratio=self.teacher_forcing_ratio)
        batch_loss = losses.mean()
        return batch_loss.item(), len(source_len)

    def valid(self):
        torch.set_grad_enabled(False)

        valid_loss = 0.0
        total_batches = len(self.v_dataset.corpus)
        epoch_examples_total = self.v_dataset.len

        print("{} batches to be evaluated".format(total_batches))
        for batch_idx in tqdm(range(total_batches)):
            loss, num_examples = self.valid_batch(batch_idx)
            valid_loss += loss * num_examples
        valid_loss /= epoch_examples_total

        return valid_loss
