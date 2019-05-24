import numpy as np
import torch
from torch.optim.lr_scheduler import *
from pprint import pprint

DELIM = u"￨"

def print_save_metrics(args, config, metrics, epoch, dataset, save_file_dir,
                       cand, ref, others, live=True, save=True, mdl=''):

    sums_with_unks, sums_with_pgens, cands_ids, tgts_ids, srcs, feats, _ = others

    if args.verbose:
        print('\n[{}] source[1]: {}'.format(mdl, srcs[1]))
        for k, v in feats.items():
            print('\n[{}] {}[1]: {}'.format(mdl, k, v[1]))
        if sums_with_unks is not None:
            print('\n[{}] sums_with_unks[1]: {}'.format(mdl, sums_with_unks[1]))
        if sums_with_pgens is not None:
            print('\n[{}] sums_with_pgens[1]: {}'.format(mdl, sums_with_pgens[1]))
        if cands_ids is not None:
            print('\n[{}] cands_ids[1]: {}'.format(mdl, cands_ids[1]))
        if tgts_ids is not None:
            print('\n[{}] tgts_ids[1]: {}'.format(mdl, tgts_ids[1]))
        print('\n[{}] ref[1]: {}'.format(mdl, ref[1][0]))
        print('\n[{}] cand[1]: {}'.format(mdl, cand[1]))

    cands_ids_original = [cands_ids[i + 1] for i in np.argsort(dataset.sort_indices).tolist()] \
        if cands_ids is not None else None

    file_ext = '{}.'.format(mdl) if len(mdl) > 0 else ''
    if save:
        cand_file_out = "{}/evaluations/{}.epoch_{}.{}cand.txt".format(save_file_dir, args.dataset, epoch, file_ext)
        with open(cand_file_out, 'w+') as fout:
            for c in range(len(cand)):
                fout.write("{}\n".format(cand[c + 1]))

        if sums_with_pgens is not None:
            cand_pgen_file_out = "{}/evaluations/{}.epoch_{}.{}cand.pgen.txt".format(save_file_dir, args.dataset, epoch, file_ext)
            with open(cand_pgen_file_out, 'w+') as fout:
                for c in range(len(sums_with_pgens)):
                    fout.write("{}\n".format(sums_with_pgens[c + 1]))

        if sums_with_unks is not None:
            cand_unk_file_out = "{}/evaluations/{}.epoch_{}.{}cand.unk.txt".format(save_file_dir, args.dataset, epoch, file_ext)
            with open(cand_unk_file_out, 'w+') as fout:
                for c in range(len(sums_with_unks)):
                    fout.write("{}\n".format(sums_with_unks[c + 1]))

        if cands_ids_original is not None:
            eval_file_out_ids = "{}/evaluations/{}.epoch_{}.{}cand.ids.txt".format(save_file_dir, args.dataset, epoch, file_ext)
            with open(eval_file_out_ids, 'w+') as fout:
                for c in range(len(cands_ids_original)):
                    fout.write("{}\n".format(" ".join([str(x) for x in cands_ids_original[c]])))

        if not live:
            ref_file_out = "{}/evaluations/{}.{}ref.sum.txt".format(save_file_dir, args.dataset, file_ext)
            with open(ref_file_out, 'w+') as fout:
                for r in range(len(ref)):
                    fout.write("{}\n".format(ref[r + 1][0]))

            src_file_out = "{}/evaluations/{}.{}ref.tbv.txt".format(save_file_dir, args.dataset, file_ext)
            with open(src_file_out, 'w+') as fout:
                for s in range(len(srcs)):
                    fout.write("{}\n".format(srcs[s + 1]))

            if args.type == 3:
                src_file_onmt = "{}/evaluations/{}.{}ref.tb.onmt.txt".format(save_file_dir, args.dataset, file_ext)
                with open(src_file_onmt, 'w+') as fout:
                    for s in range(len(srcs)):
                        a = srcs[s + 1].split()
                        b = feats['fields'][s + 1].split()
                        c = feats['rcds'][s + 1].split()
                        d = feats['has'][s + 1].split()
                        records = [DELIM.join([h, i, j, k]) for h, i, j, k in zip(a, b, c, d)]
                        fout.write("{}\n".format(" ".join(records)))

            fd_file_out = "{}/evaluations/{}.{}ref.tbl.txt".format(save_file_dir, args.dataset, file_ext)
            with open(fd_file_out, 'w+') as fout:
                for f in range(len(feats['fields'])):
                    fout.write("{}\n".format(feats['fields'][f + 1]))

    final_scores = metrics.compute_metrics(live=live, cand=cand, ref=ref, epoch=epoch, dataset=dataset,
                                           cands_ids=cands_ids_original, tgts_ids=tgts_ids)

    return final_scores


class Scheduler(object):
    """ a simple wrapper over multiple schedulers"""
    def __init__(self, config, optimizer):

        if isinstance(optimizer, tuple):
            self.scheduler = tuple([self._build_scheduler(config, x) for x in optimizer])
        else:
            self.scheduler = self._build_scheduler(config, optimizer)

    def _build_scheduler(self, config, optimizer):
        if config.decay_rate < 1:
            if config.scheduler == 'exp':
                return ExponentialLR(optimizer, gamma=config.decay_rate)
            elif config.scheduler == 'plateau':
                return ReduceLROnPlateau(optimizer, 'min', patience=1, factor=config.decay_rate)
            elif config.scheduler == 'step':
                milestones = list(range(config.decay_start, config.epochs))
                return MultiStepLR(optimizer, milestones, gamma=config.decay_rate)
            else:
                raise ValueError("{} scheduler not supported".format(config.scheduler))
        else:
            return None

    def step(self, metrics=None):

        def run_step(scheduler, metrics=None):
            if isinstance(scheduler, ReduceLROnPlateau):
                assert metrics is not None
                scheduler.step(metrics)
            else:
                scheduler.step()

        if self.scheduler is None:
            pass
        else:
            if isinstance(self.scheduler, tuple):
                if isinstance(metrics, tuple):
                    for x, y in zip(self.scheduler, metrics):
                        run_step(x, y)
                else:
                    for x in self.scheduler:
                        run_step(x)
            else:
                run_step(self.scheduler, metrics)

def detach_state(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_state(v) for v in h)

def sanity_check(args, config):
    if args.dec_type == 'prn':
        if args.enc_type == 'rnn':
            raise ValueError("args.enc_type == rnn and args.dec_type == prn is not allowed")
        else:
            if config.directions == 2:
                print("[***WARNING***]"
                      "args.enc_type == fc and args.dec_type == prn and config.directions == 2 is not allowed"
                      "Changing it to 1")
                config.directions = 1

    print("\n***args: ")
    pprint(vars(args), indent=2)
    print("\n***config: ")
    pprint(vars(config), indent=2)