import torch
import sys, os, math, copy
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('Agg')  #TkAgg
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)

class Predictor(object):
    def __init__(self, model, vocab, use_cuda, decoder_type='pg', unk_gen=False, dataset_type=0, unk_id=3):
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab = vocab
        self.decoder_type = decoder_type
        self.unk_gen = unk_gen
        self.dataset_type = dataset_type
        self.unk_id = unk_id

    def predict(self, batch_s, batch_o_s, batch_f, batch_pf, batch_pb, max_source_oov, source_len, batch_idx2oov, w2fs):
        torch.set_grad_enabled(False)
        decoded_outputs, lengths = self.model(batch_s, batch_o_s, batch_f, batch_pf, batch_pb,
                                              input_lenghts=source_len, max_source_oov=max_source_oov, w2fs=w2fs)
        length = lengths[0]
        output = []
        # print(decoded_outputs)
        for i in range(length):
            symbol = decoded_outputs[0][i].item()
            if symbol < self.vocab.size:
                output.append(self.vocab.idx2word[symbol])
            else:
                output.append(batch_idx2oov[symbol-self.vocab.size])
        print(len(output))
        return ' '.join([i for i in output if i != '<PAD>' and i != '<EOS>' and i != '<SOS>'])

    def inference(self, dataset, fig=False, save_dir=None):
        """ wrapper for 2 type of decoding schemes"""
        if self.decoder_type == 'prn':
            return self.inference_prn(dataset, fig=fig, save_dir=None)
        else:
            return self.inference_seq2seq(dataset, fig=fig, save_dir=None)

    def inference_seq2seq(self, dataset, fig=False, save_dir=None):
        """ 1-pass decoding: seq2seq, ptr-net, ptr-gen"""
        torch.set_grad_enabled(False)
        ref = {}
        cand = {}
        if self.decoder_type == 'pt' and self.dataset_type == 3:
            cand_ids = {}
            tgt_ids = {}
            feats = {'fields': {}, 'rcds': {}, 'has': {}}
        else:
            cand_ids = None
            tgt_ids = None
            feats = {'fields': {}}
        sums_with_pgens = {} if self.decoder_type == 'pg' else None
        sums_with_unks = {} if self.decoder_type != 'pt' and self.unk_gen else None
        srcs = {}
        i = 0
        pred_loss = 0
        figs_per_batch = 1
        token_count = 0
        total_batches = len(dataset.corpus)

        print("{} batches to be evaluated".format(total_batches))
        for batch_idx in tqdm(range(total_batches)):
            data_packages, texts_package, remaining = dataset.get_batch(batch_idx)
            sources, fields, summaries, outlines = texts_package
            batch_idx2oov = remaining[-1]

            batch_pf = data_packages[0][-2]
            if self.decoder_type in ['pt', 'prn']:
                lab_t = data_packages[1][-1]
                targets = outlines
            else:
                lab_t = None
                targets = summaries

            dec_outs, locations, lens, losses, p_gens, selfatt, attns = self.model(data_packages, remaining, fig=fig)
            token_count += sum(lens)
            pred_loss += sum(losses)

            i, srcs, feats, ref, \
            cand, sums_with_pgens, sums_with_unks, \
            cand_ids, tgt_ids, \
            batch_outline_positions = self.parse_batch(i, lens, dec_outs, p_gens, selfatt, attns,
                                                       srcs, feats, ref,
                                                       cand, sums_with_pgens, sums_with_unks,
                                                       cand_ids, tgt_ids,
                                                       sources, fields, targets, batch_idx2oov,
                                                       locations, lab_t,
                                                       fig, figs_per_batch, batch_pf, batch_idx, save_dir)

        avg_len = float(token_count)/dataset.len
        others = (sums_with_unks, sums_with_pgens, cand_ids, tgt_ids, srcs, feats, avg_len)

        return cand, ref, math.exp(pred_loss/token_count), others

    def inference_prn(self, dataset, fig=False, save_dir=None):
        """ 2-stage decoding: planner-realizer network"""
        torch.set_grad_enabled(False)
        
        planner_srcs = {}
        planner_feats = {'fields': {}, 'rcds': {}, 'has': {}}
        planner_ref = {}

        realizer_srcs = {}
        realizer_feats = {'fields': {}, 'rcds': {}, 'has': {}}
        realizer_ref = {}

        outline_ids = {}
        outline_tgt_ids = {}

        batch_otls = {}
        batch_sums = {}
        sums_with_pgens = {}
        sums_with_unks = {}

        planner_counter = 0
        realizer_counter = 0
        planner_loss = 0
        realizer_loss = 0
        figs_per_batch = 1
        token_count_otl = 0
        token_count_sum = 0
        total_batches = len(dataset.corpus)

        print("{} batches to be evaluated".format(total_batches))
        for batch_idx in tqdm(range(total_batches)):

            # prepare data
            data_packages, texts_package, remaining = dataset.get_batch(batch_idx)

            batch_s, batch_o_s, batch_f, batch_pf, batch_pb,\
            _, _,\
            _, _, \
            _, outline_len, _, max_tail_oov, w2fs, batch_idx2oov = self.model.unpack_batch_data(data_packages, remaining)
            sources, fields, summaries, outlines = texts_package

            # stage 1 decoding
            lab_t = data_packages[1][-1]
            targets = outlines

            dec_outs, locations, outline_len, losses, _, _, attns = self.model(data_packages, remaining, fig=fig)
            token_count_otl += sum(outline_len)
            planner_loss += sum(losses)

            planner_counter, planner_srcs, planner_feats, planner_ref, \
            batch_otls, _, _, \
            outline_ids, outline_tgt_ids, \
            batch_outline_positions = self.parse_batch(planner_counter,
                                                       outline_len, dec_outs, None, None, attns,
                                                       planner_srcs, planner_feats, planner_ref,
                                                       batch_otls, None, None,
                                                       outline_ids, outline_tgt_ids,
                                                       sources, fields, targets, None,
                                                       locations, lab_t,
                                                       fig, figs_per_batch, batch_pf, batch_idx, save_dir)

            # stage 2 decoding

            outline_len = [x - 1 for x in outline_len]
            batch_otl_pos_tensor = self.pad_eos(batch_outline_positions, outline_len)
            batch_t = (batch_s.gather(1, batch_otl_pos_tensor),
                       batch_f.gather(1, batch_otl_pos_tensor),
                       batch_pf.gather(1, batch_otl_pos_tensor),
                       batch_pb.gather(1, batch_otl_pos_tensor))
            batch_o_t = batch_o_s.gather(1, batch_otl_pos_tensor)

            temp = copy.deepcopy(fields)
            for k, v in fields.items():
                for bth, seq in enumerate(v):
                    temp[k][bth] = np.array(seq)[[x-1 for x in batch_outline_positions[bth]]].tolist()
            fields = temp
            targets = summaries

            pred_summaries = self.model.cont_fn(batch_t, batch_o_t=batch_o_t,
                                                max_tail_oov=max_tail_oov, outline_len=outline_len, w2fs=w2fs, fig=fig)

            dec_outs, _, summary_lens, losses, p_gens, selfatt, attns = pred_summaries
            token_count_sum += sum(summary_lens)
            realizer_loss += sum(losses)

            realizer_counter, realizer_srcs, realizer_feats, realizer_ref, \
            batch_sums, sums_with_pgens, sums_with_unks, \
            _, _, _ = self.parse_batch(realizer_counter,
                                       summary_lens, dec_outs, p_gens, selfatt, attns,
                                       realizer_srcs, realizer_feats, realizer_ref,
                                       batch_sums, sums_with_pgens, sums_with_unks,
                                       None, None,
                                       outlines, fields, targets, batch_idx2oov,
                                       None, None,
                                       fig, figs_per_batch, batch_t[2], batch_idx, save_dir)

        # combine results from two stages
        others_1 = (None, None, outline_ids, outline_tgt_ids, planner_srcs, planner_feats, float(token_count_otl)/dataset.len)
        planner_output = (batch_otls, planner_ref, math.exp(planner_loss/token_count_otl), others_1)

        others_2 = (sums_with_unks, sums_with_pgens, None, None, realizer_srcs, realizer_feats, float(token_count_sum)/dataset.len)
        realizer_output = (batch_sums, realizer_ref, math.exp(realizer_loss/token_count_sum), others_2)

        return {'planner': planner_output, 'realizer': realizer_output}

    def pad_eos(self, batch_outline_positions, lens):
        out = []
        max_len = max(lens)
        for x, y in zip(batch_outline_positions, lens):
            out.append(x[:y]+[2]*(max_len-y))
        return torch.from_numpy(np.array(out)).to(self.device)

    def parse_batch(self, i,
                    lens, dec_outs, p_gens, selfatt, attns,
                    srcs, feats, ref,
                    cand, sums_with_pgens, sums_with_unks,
                    cand_ids, tgt_ids,
                    sources, fields, targets, batch_idx2oov,
                    locations, lab_t,
                    fig, figs_per_batch, batch_pf, batch_idx, save_dir):

        batch_outline_positions = []

        for j in range(len(lens)):
            i += 1

            # record original srcs, feats, and ref
            # TODO: do not need if all can be reversed back to the original order
            srcs[i] = ' '.join(['_'.join(x.split()) for x in sources[j]])

            # NOTE: fields is a dictionary of all feats for ptr-net
            for k, v in fields.items():
                feats[k][i] = ' '.join(v[j])

            ref[i] = [' '.join(targets[j])]

            # for ptr-net only
            if locations is not None:
                out_seq_ids = locations[j].tolist()
                out_seq_ids = out_seq_ids[:lens[j] - 1]
                cand_ids[i] = out_seq_ids
                batch_outline_positions.append(out_seq_ids)
                tgt_seq_ids = [x for x in lab_t[j].tolist() if x > 3]
                tgt_ids[i] = tgt_seq_ids

            # get tokens and replace OOVs
            out_seq_clean = []
            out_seq_unk = []
            for k in range(lens[j]):
                symbol = dec_outs[j][k].item()
                # seq2seq and ptr-net
                if symbol < self.vocab.size:
                    if symbol == self.unk_id and self.unk_gen:
                        replacement = sources[j][attns[j][k].argmax()]
                        out_seq_clean.append(replacement)
                    else:
                        out_seq_clean.append(self.vocab.idx2word[symbol])
                    out_seq_unk.append(self.vocab.idx2word[symbol])
                # ptr-gen decoder
                else:
                    oov = batch_idx2oov[j][symbol - self.vocab.size]
                    out_seq_clean.append(oov)
                    out_seq_unk.append(oov)

            # post-processing for text metrics
            pgen = p_gens[j] if p_gens is not None else None
            out, out_unk, out_with_gens = self.post_process(out_seq_clean, out_seq_unk, pgen)

            cand[i] = ' '.join(out)
            if sums_with_pgens is not None:
                sums_with_pgens[i] = ' '.join(out_with_gens)
            if sums_with_unks is not None:
                sums_with_unks[i] = ' '.join(out_unk)

            # make attention visualization figures
            if fig and len(out) > 0 and j < figs_per_batch:
                fmatrix = selfatt[j] if selfatt is not None else None
                self.make_figure(lens[j], out, fmatrix, attns[j], batch_pf[j], sources[j], batch_idx + j, save_dir)

        return i, srcs, feats, ref, cand, sums_with_pgens, sums_with_unks, cand_ids, tgt_ids, batch_outline_positions

    def post_process(self, sentence, sentence_unk, pgen=None):
        try:
            eos = sentence.index('<EOS>')
            sentence = sentence[:eos]
            sentence_unk = sentence_unk[:eos]
            if pgen is not None:
                pgen = pgen[:eos]
        except ValueError:
            pass

        if pgen is None:
            sentence_trim = [x for x in sentence if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
            sentence_unk_trim = [x for x in sentence_unk if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
            return sentence_trim, sentence_unk_trim, []
        else:
            token_pgens = [(x, y, z.item()) for x, y, z in zip(sentence, sentence_unk, pgen)
                           if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
            if len(token_pgens) > 0:
                out, out_unk, pgens_filtered = zip(*token_pgens)
                pgens_filtered = ["_%.3f"%x if x < 0.7 else '' for x in pgens_filtered]
                out_with_gens = ["{}{}".format('_'.join(x.split()), y) for x, y in zip(out, pgens_filtered)]
                return list(out), list(out_unk), out_with_gens
            else:
                return [], [], []

    def show_attention(self, input_words, output_words, attentions, attn='self', xlabel="", ylabel="", idx=0, dir=''):
        # Set up figure with colorbar
        plt.rcParams.update({'font.size': 18})
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap=plt.cm.Blues)
        if attn == 'self':
            fig.colorbar(cax, pad=0.03)
        else:
            fig.colorbar(cax, shrink=0.7, pad=0.03)

        # Set up axes
        ax.set_xticklabels([''] + input_words, rotation=90)
        ax.set_yticklabels([''] + output_words)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        fig.savefig(os.path.join(dir, "attention_figures/{}.{}.png".format(attn, idx)), bbox_inches='tight')
        plt.close(fig)

    def make_figure(self, length, out, fmatrix, attns, batch_pf, src_words, idx, savedir):
        pos = [str(i) for i in batch_pf.cpu().tolist() if i >0]
        combine = []
        for j in range(len(pos)):
            combine.append(src_words[j] + " : " + pos[j])

        if fmatrix is not None:
            self.show_attention(pos, combine, fmatrix.cpu(),
                               attn='self', xlabel='Table Position', ylabel='Table Position', idx=idx, dir=savedir)
        self.show_attention(combine, out, attns[:len(out)].cpu(),
                           attn='attn', xlabel='Structured KB', ylabel='Text Output', idx=idx, dir=savedir)
