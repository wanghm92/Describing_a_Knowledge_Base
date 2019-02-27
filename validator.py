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
        batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_source_oov, \
        w2fs, sources, targets, fields, list_oovs = self.v_dataset.get_batch(batch_idx)

        losses = self.model(batch_s, batch_o_s, batch_f, batch_pf, batch_pb,
                            target=batch_t, target_id=batch_o_t,
                            input_lengths=source_len, max_source_oov=max_source_oov,
                            teacher_forcing_ratio=self.teacher_forcing_ratio)
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
